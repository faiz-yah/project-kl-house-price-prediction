import pandas as pd
import numpy as np
import re
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error


def load_data(filepath):
    df = pd.read_excel(filepath)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def data_preprocessing(df):
    data = df.copy()

    # Standardise column names
    data.columns = data.columns.str.strip()

    def make_unique_columns(columns):
        seen = {}
        new_columns = []
        for col in columns:
            if col in seen:
                seen[col] += 1
                new_columns.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 1
                new_columns.append(col)
        return new_columns

    data.columns = make_unique_columns(data.columns)

    # Forward-fill repeated categorical values
    ffill_columns = [
        "Property Type",
        "District",
        "Mukim",
        "Scheme Name/Area",
        "Road Name",
        "Month, Year of Transaction Date",
        "Tenure"
    ]
    data[ffill_columns] = data[ffill_columns].ffill()

    # Clean numeric columns
    def clean_numeric(value):
        if pd.isna(value):
            return np.nan
        value = str(value).strip()
        if value in ("-", ""):
            return np.nan
        value = re.sub(r"[^0-9.]", "", value)
        return float(value) if value else np.nan

    data["Transaction_Price"] = data["Transaction Price"].apply(clean_numeric)
    data["Land_Parcel_Area"] = data["Land/Parcel Area"].apply(clean_numeric)
    data["Main_Floor_Area"] = data["Main Floor Area"].apply(clean_numeric)

    # Clean unit level
    def clean_unit_level(value):
        if pd.isna(value):
            return 0
        value = str(value).strip().upper()
        if value in ("", "-", "NAN"):
            return 0
        if value in ("G", "GROUND"):
            return 1
        number = re.search(r"\d+", value)
        return int(number.group()) + 1 if number else 0

    data["Unit_Level"] = data["Unit Level"].apply(clean_unit_level)

    # Convert transaction date to month and year
    data["Transaction_Date"] = pd.to_datetime(
        data["Month, Year of Transaction Date"],
        format="%B %Y",
        errors="coerce"
    )
    data["Transaction_Month"] = data["Transaction_Date"].dt.month
    data["Transaction_Year"] = data["Transaction_Date"].dt.year

    # Create property group
    high_rise_types = ["Condominium/Apartment", "Flat", "Low-Cost Flat"]
    landed_types = [
        "1 - 1 1/2 Storey Semi-Detached", "1 - 1 1/2 Storey Terraced",
        "2 - 2 1/2 Storey Semi-Detached", "2 - 2 1/2 Storey Terraced",
        "Cluster House", "Detached", "Low-Cost House", "Town House"
    ]

    def classify_property_group(property_type):
        if property_type in high_rise_types:
            return "High-Rise"
        elif property_type in landed_types:
            return "Landed"
        return "Others"

    data["Property_Group"] = data["Property Type"].apply(classify_property_group)

    # Select and rename columns
    selected_columns = [
        "Property Type", "Property_Group", "District", "Mukim",
        "Scheme Name/Area", "Road Name", "Tenure",
        "Transaction_Month", "Transaction_Year",
        "Land_Parcel_Area", "Main_Floor_Area", "Unit_Level", "Transaction_Price"
    ]
    clean_df = data[selected_columns].copy()
    clean_df = clean_df.dropna(subset=["Transaction_Price"])
    clean_df = clean_df[clean_df["Transaction_Price"] > 0]

    rename_map = {
        'Property Type': 'property_type',
        'Property_Group': 'property_group',
        'District': 'district',
        'Mukim': 'mukim',
        'Scheme Name/Area': 'scheme_name',
        'Road Name': 'road_name',
        'Tenure': 'tenure',
        'Transaction_Month': 'transaction_month',
        'Transaction_Year': 'transaction_year',
        'Land_Parcel_Area': 'land_parcel_area',
        'Main_Floor_Area': 'main_floor_area',
        'Unit_Level': 'unit_level',
        'Transaction_Price': 'transaction_price'
    }
    clean_df.columns = clean_df.columns.map(rename_map)

    # Drop main_floor_area (76% missing) and rows missing land_parcel_area
    modelling_df = clean_df[~clean_df['land_parcel_area'].isna()].drop(columns='main_floor_area')

    # Feature engineering
    modelling_df = modelling_df.copy()
    modelling_df['property_age'] = modelling_df['transaction_year'] - modelling_df['transaction_year'].min()

    # 70-15-15 split
    X = modelling_df.drop(columns='transaction_price')
    y = modelling_df['transaction_price']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # OHE for low-cardinality categoricals
    ohe_cols = ['property_type', 'property_group', 'district', 'mukim', 'tenure']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def apply_ohe(X_train, X_val, X_test):
        X_ohe_train = encoder.fit_transform(X_train[ohe_cols])
        ohe_feature_names = encoder.get_feature_names_out()

        result = {}
        for split_name, X in {'train': X_train, 'val': X_val, 'test': X_test}.items():
            X_ohe = encoder.transform(X[ohe_cols]) if split_name != 'train' else X_ohe_train
            X_ohe = pd.DataFrame(X_ohe, columns=ohe_feature_names, index=X.index)
            result[split_name] = pd.concat([X.drop(columns=ohe_cols), X_ohe], axis=1)

        return result['train'], result['val'], result['test']

    X_train, X_val, X_test = apply_ohe(X_train, X_val, X_test)

    # Target encoding for high-cardinality categoricals
    te_cols = ['scheme_name', 'road_name']
    target_encoder = TargetEncoder(target_type='continuous', smooth='auto')

    def apply_target_encoding(X_train, y_train, X_val, X_test):
        result = {}
        for split_name, X in {'train': X_train, 'val': X_val, 'test': X_test}.items():
            if split_name == 'train':
                X_te = target_encoder.fit_transform(X[te_cols], y_train)
            else:
                X_te = target_encoder.transform(X[te_cols])
            X_te = pd.DataFrame(X_te, columns=target_encoder.get_feature_names_out(), index=X.index)
            result[split_name] = pd.concat([X.drop(columns=te_cols), X_te], axis=1)
        return result['train'], result['val'], result['test']

    X_train, X_val, X_test = apply_target_encoding(X_train, y_train, X_val, X_test)

    # Standardise all features
    features = X_train.columns.tolist()
    scaler = StandardScaler()
    X_train[features] = scaler.fit_transform(X_train[features])
    X_val[features] = scaler.transform(X_val[features])
    X_test[features] = scaler.transform(X_test[features])

    y_train = y_train.values.ravel()
    y_val = y_val.values.ravel()
    y_test = y_test.values.ravel()

    return X_train, X_val, X_test, y_train, y_val, y_test


def model_development(X_train, y_train):
    # Best params from Optuna tuning (n_estimators=232, max_depth=16, min_samples_leaf=3, R²=0.8576)
    params = {
        'n_estimators': 232,
        'max_depth': 16,
        'min_samples_leaf': 3,
        'random_state': 42,
        'n_jobs': -1
    }

    print("Training Random Forest with best hyperparameters...")
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    print("Training complete.")

    return model, params


def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test):
    datasets = {
        'Train': (X_train, y_train),
        'Validation': (X_val, y_val),
        'Test': (X_test, y_test)
    }

    print("\n" + "=" * 60)
    print("Model Performance Evaluation")
    print("=" * 60)

    for name, (X, y) in datasets.items():
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        mape = mean_absolute_percentage_error(y, y_pred)

        print(f"\n{name} Set:")
        print(f"  R²:   {r2:.4f}")
        print(f"  RMSE: {rmse:,.2f}")
        print(f"  MAE:  {mae:,.2f}")
        print(f"  MAPE: {mape:.4f}")


def save_best_model_with_best_hyperparameter(model, params, save_dir='model'):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    model_path = save_dir / f'rf_model_{timestamp}.joblib'
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    params_path = save_dir / f'rf_params_{timestamp}.joblib'
    joblib.dump(params, params_path)
    print(f"Hyperparameters saved to: {params_path}")

    joblib.dump(model, save_dir / 'rf_model_latest.joblib')
    joblib.dump(params, save_dir / 'rf_params_latest.joblib')
    print(f"Latest version saved to: {save_dir / 'rf_model_latest.joblib'}")

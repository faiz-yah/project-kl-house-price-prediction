from preprocess_train import (
    load_data,
    data_preprocessing,
    evaluate_model,
    save_best_model_with_best_hyperparameter,
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from pathlib import Path


def main():
    PARENT_PATH = Path(__file__).parent.parent
    DATA_PATH = PARENT_PATH / 'data/7012 Raw Data.xlsx'
    MODEL_DIR = PARENT_PATH / 'model'

    print("=" * 60)
    print("Model Training Pipeline")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    df = load_data(DATA_PATH)

    print("\n[2/4] Preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = data_preprocessing(df)

    # ------------------------------------------------------------------
    # FULL PIPELINE: train all 3 models and pick the best automatically
    # ------------------------------------------------------------------
    # model_candidates = {
    #     'linear_regression': LinearRegression(),
    #     'random_forest':     RandomForestRegressor(n_estimators=232, max_depth=16, min_samples_leaf=3, random_state=42, n_jobs=-1),
    #     'xgboost':           XGBRegressor(n_estimators=325, max_depth=3, learning_rate=0.030790258303172606, subsample=0.9660678426558463, random_state=42),
    # }
    #
    # best_name, best_model, best_r2 = None, None, -float('inf')
    #
    # for name, m in model_candidates.items():
    #     m.fit(X_train, y_train)
    #     r2 = r2_score(y_val, m.predict(X_val))
    #     print(f"  {name} val R²: {r2:.4f}")
    #     if r2 > best_r2:
    #         best_name, best_model, best_r2 = name, m, r2
    #
    # print(f"\nBest model: {best_name} (val R² = {best_r2:.4f})")
    # params = model_candidates[best_name].get_params()
    # ------------------------------------------------------------------

    # Train Random Forest with Optuna best params (val R² = 0.8576)
    print("\n[3/4] Training Random Forest...")
    params = {'n_estimators': 232, 'max_depth': 16, 'min_samples_leaf': 3, 'random_state': 42, 'n_jobs': -1}
    best_model = RandomForestRegressor(**params)
    best_model.fit(X_train, y_train)

    print("\n[4/4] Evaluating model...")
    evaluate_model(best_model, X_train, X_val, X_test, y_train, y_val, y_test)

    save_best_model_with_best_hyperparameter(best_model, params, save_dir=MODEL_DIR)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

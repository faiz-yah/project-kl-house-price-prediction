import joblib
import pandas as pd
import numpy as np
from pathlib import Path

HIGH_RISE = ["Condominium/Apartment", "Flat", "Low-Cost Flat"]
LANDED = [
    "1 - 1 1/2 Storey Semi-Detached", "1 - 1 1/2 Storey Terraced",
    "2 - 2 1/2 Storey Semi-Detached", "2 - 2 1/2 Storey Terraced",
    "Cluster House", "Detached", "Low-Cost House", "Town House",
]
ALL_TYPES = HIGH_RISE + LANDED
ALL_MUKIMS = [
    "Kuala Lumpur Town Centre", "Mukim Ampang", "Mukim Batu", "Mukim Cheras",
    "Mukim Kuala Lumpur", "Mukim Petaling", "Mukim Setapak", "Mukim Ulu Kelang",
]
ALL_TENURES = ["Freehold", "Leasehold"]
AREA_SAMPLES = [50, 80, 100, 130, 160, 200, 280, 400, 600]


def classify_property_group(property_type: str) -> str:
    if property_type in HIGH_RISE:
        return "High-Rise"
    if property_type in LANDED:
        return "Landed"
    return "Others"


def load_artifacts(model_dir: Path):
    model = joblib.load(model_dir / "rf_model_latest.joblib")
    preprocessors = joblib.load(model_dir / "preprocessors_latest.joblib")
    return model, preprocessors


def _encode_and_predict(df_raw: pd.DataFrame, model, preprocessors) -> np.ndarray:
    df = df_raw.copy()
    df["property_age"] = df["transaction_year"] - preprocessors["min_year"]

    ohe = preprocessors["ohe"]
    ohe_cols = preprocessors["ohe_cols"]
    X_ohe = pd.DataFrame(ohe.transform(df[ohe_cols]),
                         columns=ohe.get_feature_names_out(), index=df.index)
    df = pd.concat([df.drop(columns=ohe_cols), X_ohe], axis=1)

    te = preprocessors["target_encoder"]
    te_cols = preprocessors["te_cols"]
    X_te = pd.DataFrame(te.transform(df[te_cols]),
                        columns=te.get_feature_names_out(), index=df.index)
    df = pd.concat([df.drop(columns=te_cols), X_te], axis=1)

    df = df[preprocessors["feature_names"]]
    return model.predict(preprocessors["scaler"].transform(df))


def predict(input_data: dict, model, preprocessors) -> float:
    row = {
        "property_type":     input_data["property_type"],
        "property_group":    classify_property_group(input_data["property_type"]),
        "district":          input_data["district"],
        "mukim":             input_data["mukim"],
        "scheme_name":       input_data.get("scheme_name") or "UNKNOWN",
        "road_name":         input_data.get("road_name")   or "UNKNOWN",
        "tenure":            input_data["tenure"],
        "transaction_month": int(input_data["transaction_month"]),
        "transaction_year":  int(input_data["transaction_year"]),
        "land_parcel_area":  float(input_data["land_parcel_area"]),
        "unit_level":        int(input_data["unit_level"]),
    }
    return float(_encode_and_predict(pd.DataFrame([row]), model, preprocessors)[0])


def build_grid(model, preprocessors) -> list:
    rows = []
    for prop_type in ALL_TYPES:
        levels = [0, 5, 10, 20, 35] if prop_type in HIGH_RISE else [0]
        for mukim in ALL_MUKIMS:
            for tenure in ALL_TENURES:
                for area in AREA_SAMPLES:
                    for level in levels:
                        rows.append({
                            "property_type":    prop_type,
                            "property_group":   classify_property_group(prop_type),
                            "district":         "Kuala Lumpur",
                            "mukim":            mukim,
                            "scheme_name":      "UNKNOWN",
                            "road_name":        "UNKNOWN",
                            "tenure":           tenure,
                            "transaction_month": 6,
                            "transaction_year":  2024,
                            "land_parcel_area":  float(area),
                            "unit_level":        level,
                        })

    prices = _encode_and_predict(pd.DataFrame(rows), model, preprocessors)

    grid = []
    for row, price in zip(rows, prices):
        grid.append({
            "property_type":    row["property_type"],
            "property_group":   row["property_group"],
            "mukim":            row["mukim"],
            "tenure":           row["tenure"],
            "land_parcel_area": int(row["land_parcel_area"]),
            "unit_level":       row["unit_level"],
            "predicted_price":  round(float(price)),
        })
    return grid

import sqlite3
import joblib
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from config import CITIES, DB_NAME


def run_training() -> None:
    with sqlite3.connect(DB_NAME) as conn:
        df = pd.read_sql("SELECT * FROM historical_weather", conn)

    df["city"] = pd.Categorical(df["city"], categories=sorted(CITIES.keys()))
    df["time"] = pd.to_datetime(df["time"], format="mixed")
    df = df.sort_values(by=["city", "time"]).reset_index(drop=True)

    days_span_7 = (
        df["time"] - df.groupby("city", observed=True)["time"].shift(7)
    ).dt.days
    is_continuous_7d = days_span_7 == 7

    feature_cols = [
        "temperature_2m_max",
        "temperature_2m_min",
    ]

    lag2_df = df.groupby("city", observed=True)[feature_cols].shift(2)
    lag2_df.columns = [f"{col}_lag2" for col in feature_cols]

    all_feature_cols = feature_cols + [
        "precipitation_sum",
        "wind_speed_10m_max",
        "surface_pressure_mean",
        "cloudcover_mean",
        "relative_humidity_2m_mean",
        "winddirection_10m_dominant",
    ]

    lag1_df = df.groupby("city", observed=True)[all_feature_cols].shift(1)
    lag1_df.columns = [f"{col}_lag1" for col in all_feature_cols]

    df["temp_max_rolling_mean_7"] = df.groupby("city", observed=True)[
        "temperature_2m_max"
    ].transform(lambda s: s.shift(1).rolling(7).mean())

    df = pd.concat([df, lag1_df, lag2_df], axis=1)
    df["temp_max_diff"] = (
        df["temperature_2m_max_lag1"] - df["temperature_2m_max_lag2"]
    )

    required_cols = (
        ["city", "time", "temperature_2m_max", "temp_max_rolling_mean_7"]
        + [f"{col}_lag1" for col in all_feature_cols]
        + [f"{col}_lag2" for col in feature_cols]
    )
    df = df[is_continuous_7d].dropna(subset=required_cols).reset_index(drop=True)

    if len(df) < 100:
        raise ValueError(
            f"Insufficient complete rows for training ({len(df)}). Aborting to avoid overwriting model."
        )

    days = df["time"].dt.dayofyear
    df["sin_day"] = np.sin(2 * np.pi * days / 365.25)
    df["cos_day"] = np.cos(2 * np.pi * days / 365.25)

    wind_rad = np.deg2rad(df["winddirection_10m_dominant_lag1"])
    df["wind_u_lag1"] = -df["wind_speed_10m_max_lag1"] * np.sin(wind_rad)
    df["wind_v_lag1"] = -df["wind_speed_10m_max_lag1"] * np.cos(wind_rad)

    df["temp_amplitude_lag1"] = (
        df["temperature_2m_max_lag1"] - df["temperature_2m_min_lag1"]
    )

    cols_to_drop = [
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
        "surface_pressure_mean",
        "cloudcover_mean",
        "relative_humidity_2m_mean",
        "winddirection_10m_dominant",
    ]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    target = "temperature_2m_max"
    features = [
        "city",
        "temp_max_rolling_mean_7",
        "temperature_2m_max_lag1",
        "temperature_2m_min_lag1",
        "precipitation_sum_lag1",
        "wind_speed_10m_max_lag1",
        "surface_pressure_mean_lag1",
        "cloudcover_mean_lag1",
        "relative_humidity_2m_mean_lag1",
        "winddirection_10m_dominant_lag1",
        "temperature_2m_max_lag2",
        "temperature_2m_min_lag2",
        "temp_max_diff",
        "sin_day",
        "cos_day",
        "wind_v_lag1",
        "wind_u_lag1",
        "temp_amplitude_lag1",
    ]

    X_train = df[features]
    y_train = df[target]

    lgbm = LGBMRegressor(
        random_state=42,
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=15,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        min_child_samples=30,
        verbose=-1,
    ).fit(X_train, y_train)

    importance = pd.DataFrame(
        {"feature": features, "importance": lgbm.feature_importances_}
    ).sort_values("importance", ascending=False)

    print("Feature importance:")
    print(importance.to_string(index=False))

    joblib.dump(lgbm, "weather_lgbm_model.joblib")
    joblib.dump(features, "model_features.joblib")
    print("\nModel and features saved successfully.")


if __name__ == "__main__":
    run_training()
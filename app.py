import os
import logging
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_demand_data():
    supabase = get_supabase()
    response = supabase.table("elexon_demand") \
        .select("start_time, transmission_system_demand, national_demand") \
        .order("start_time") \
        .execute()
    df = pd.DataFrame(response.data)
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df["start_time"] = df["start_time"].dt.tz_localize(None)
    df = df.rename(columns={"start_time": "ds", "transmission_system_demand": "y"})
    df = df.dropna(subset=["y"])
    df = df.sort_values("ds").reset_index(drop=True)
    logger.info(f"Fetched {len(df)} demand rows")
    return df

def fetch_weather_data():
    supabase = get_supabase()
    response = supabase.table("weather_forecast") \
        .select("timestamp, temperature_2m, windspeed_10m, shortwave_radiation") \
        .order("timestamp") \
        .execute()
    df = pd.DataFrame(response.data)
    if len(df) == 0:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def engineer_features(df):
    """Create time-series features for sklearn model."""
    df = df.copy()
    df["hour"] = df["ds"].dt.hour
    df["minute"] = df["ds"].dt.minute
    df["dayofweek"] = df["ds"].dt.dayofweek
    df["dayofyear"] = df["ds"].dt.dayofyear
    df["month"] = df["ds"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    # Settlement period (1-48 for half-hourly)
    df["settlement_period"] = df["hour"] * 2 + df["minute"] // 30 + 1
    # Lag features
    df["lag_48"] = df["y"].shift(48)   # 24hrs ago
    df["lag_96"] = df["y"].shift(96)   # 48hrs ago
    df["lag_336"] = df["y"].shift(336) # 1 week ago
    # Rolling mean
    df["rolling_mean_48"] = df["y"].rolling(window=48, min_periods=1).mean()
    df["rolling_std_48"] = df["y"].rolling(window=48, min_periods=1).std().fillna(0)
    return df

def merge_weather(demand_df, weather_df):
    """Merge weather on nearest hour."""
    if len(weather_df) == 0:
        demand_df["temperature_2m"] = 12.0
        demand_df["windspeed_10m"] = 10.0
        demand_df["shortwave_radiation"] = 0.0
        return demand_df
    demand_df["ds_hour"] = demand_df["ds"].dt.floor("H")
    weather_df = weather_df.rename(columns={"timestamp": "ds_hour"})
    merged = pd.merge_asof(
        demand_df.sort_values("ds_hour"),
        weather_df.sort_values("ds_hour"),
        on="ds_hour", direction="nearest"
    ).drop(columns=["ds_hour"])
    merged = merged.sort_values("ds").reset_index(drop=True)
    for col in ["temperature_2m", "windspeed_10m", "shortwave_radiation"]:
        merged[col] = merged[col].fillna(method="ffill").fillna(method="bfill").fillna(0)
    return merged

def train_and_forecast(demand_df, weather_df, horizon_periods=96):
    """Train GBM model and generate forecast."""
    df = merge_weather(demand_df, weather_df)
    df = engineer_features(df)

    feature_cols = [
        "hour", "minute", "dayofweek", "dayofyear", "month",
        "is_weekend", "settlement_period",
        "lag_48", "lag_96", "rolling_mean_48", "rolling_std_48",
        "temperature_2m", "windspeed_10m", "shortwave_radiation"
    ]

    # Train on rows where all lags are available
    train_df = df.dropna(subset=feature_cols)
    if len(train_df) < 10:
        raise ValueError(f"Insufficient training data after feature engineering: {len(train_df)} rows")

    X_train = train_df[feature_cols]
    y_train = train_df["y"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=5,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_scaled, y_train)
    logger.info(f"Model trained on {len(train_df)} rows")

    # Build future timestamps
    last_ts = df["ds"].max()
    future_timestamps = [last_ts + timedelta(minutes=30 * i) for i in range(1, horizon_periods + 1)]

    # Build future feature rows
    future_rows = []
    last_known = df["y"].values.tolist()

    for i, ts in enumerate(future_timestamps):
        row = {
            "ds": ts,
            "hour": ts.hour,
            "minute": ts.minute,
            "dayofweek": ts.dayofweek,
            "dayofyear": ts.dayofyear,
            "month": ts.month,
            "is_weekend": int(ts.weekday() >= 5),
            "settlement_period": ts.hour * 2 + ts.minute // 30 + 1,
        }
        # Lag features from known + predicted values
        all_vals = last_known + [r.get("y_pred", np.nan) for r in future_rows]
        n = len(all_vals)
        row["lag_48"] = all_vals[n - 48] if n >= 48 else np.mean(last_known[-48:])
        row["lag_96"] = all_vals[n - 96] if n >= 96 else np.mean(last_known[-96:])
        recent = all_vals[-48:] if len(all_vals) >= 48 else all_vals
        row["rolling_mean_48"] = np.mean(recent)
        row["rolling_std_48"] = np.std(recent) if len(recent) > 1 else 0

        # Weather — use last known values
        if len(weather_df) > 0:
            weather_row = weather_df[weather_df["timestamp"] <= ts]
            if len(weather_row) > 0:
                row["temperature_2m"] = float(weather_row.iloc[-1]["temperature_2m"])
                row["windspeed_10m"] = float(weather_row.iloc[-1]["windspeed_10m"])
                row["shortwave_radiation"] = float(weather_row.iloc[-1]["shortwave_radiation"])
            else:
                row["temperature_2m"] = 12.0
                row["windspeed_10m"] = 10.0
                row["shortwave_radiation"] = 0.0
        else:
            row["temperature_2m"] = 12.0
            row["windspeed_10m"] = 10.0
            row["shortwave_radiation"] = 0.0

        X_row = np.array([[row[c] for c in feature_cols]])
        X_row_scaled = scaler.transform(X_row)
        y_pred = float(model.predict(X_row_scaled)[0])
        row["y_pred"] = y_pred
        future_rows.append(row)

    # Build confidence intervals using training residuals
    train_preds = model.predict(X_scaled)
    residuals = y_train.values - train_preds
    std_residual = np.std(residuals)

    forecast_data = []
    forecasts = [r["y_pred"] for r in future_rows]
    for i, (ts, pred) in enumerate(zip(future_timestamps, forecasts)):
        forecast_data.append({
            "timestamp": ts.isoformat(),
            "forecast_mw": round(pred, 1),
            "lower_bound_mw": round(pred - 1.96 * std_residual, 1),
            "upper_bound_mw": round(pred + 1.96 * std_residual, 1),
        })

    return forecast_data, len(train_df), df["ds"].min(), df["ds"].max()

# ── API Routes ─────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "service": "Verdant Innovations — Energy Demand Forecast API",
        "version": "1.1.0",
        "model": "Gradient Boosting Regressor (scikit-learn)",
        "endpoints": ["/forecast", "/forecast/latest", "/data/summary", "/health"]
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.utcnow().isoformat()})

@app.route("/data/summary", methods=["GET"])
def data_summary():
    try:
        demand_df = fetch_demand_data()
        weather_df = fetch_weather_data()
        return jsonify({
            "status": "success",
            "demand": {
                "rows": len(demand_df),
                "from": demand_df["ds"].min().isoformat() if len(demand_df) > 0 else None,
                "to": demand_df["ds"].max().isoformat() if len(demand_df) > 0 else None,
                "days_covered": demand_df["ds"].dt.date.nunique() if len(demand_df) > 0 else 0,
                "ready_for_training": len(demand_df) >= 96
            },
            "weather": {
                "rows": len(weather_df),
                "from": weather_df["timestamp"].min().isoformat() if len(weather_df) > 0 else None,
                "to": weather_df["timestamp"].max().isoformat() if len(weather_df) > 0 else None,
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/forecast", methods=["GET"])
def forecast():
    try:
        hours = min(int(request.args.get("hours", 48)), 168)
        periods = hours * 2

        demand_df = fetch_demand_data()
        weather_df = fetch_weather_data()

        if len(demand_df) < 96:
            return jsonify({
                "error": "Insufficient data",
                "message": f"Need at least 96 rows, have {len(demand_df)}",
                "status": "error"
            }), 400

        forecast_data, n_train, train_from, train_to = train_and_forecast(
            demand_df, weather_df, horizon_periods=periods
        )

        forecasts = [f["forecast_mw"] for f in forecast_data]
        peak_idx = forecasts.index(max(forecasts))

        return jsonify({
            "status": "success",
            "generated_at": datetime.utcnow().isoformat(),
            "model": "Gradient Boosting Regressor",
            "training_rows": n_train,
            "training_from": train_from.isoformat(),
            "training_to": train_to.isoformat(),
            "forecast_hours": hours,
            "forecast_points": len(forecast_data),
            "summary": {
                "peak_demand_mw": max(forecasts),
                "min_demand_mw": min(forecasts),
                "avg_demand_mw": round(sum(forecasts) / len(forecasts), 1),
                "peak_timestamp": forecast_data[peak_idx]["timestamp"],
            },
            "forecast": forecast_data
        })

    except Exception as e:
        logger.error(f"Forecast error: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/forecast/latest", methods=["GET"])
def forecast_latest():
    try:
        demand_df = fetch_demand_data()
        weather_df = fetch_weather_data()

        if len(demand_df) < 96:
            return jsonify({"error": "Insufficient data"}), 400

        forecast_data, _, _, _ = train_and_forecast(
            demand_df, weather_df, horizon_periods=12
        )

        return jsonify({
            "status": "success",
            "generated_at": datetime.utcnow().isoformat(),
            "next_6_hours": forecast_data
        })

    except Exception as e:
        logger.error(f"Latest forecast error: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

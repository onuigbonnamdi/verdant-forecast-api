import os
import json
import logging
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from prophet import Prophet
from supabase import create_client, Client

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Supabase connection ────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Data fetching ──────────────────────────────────────────────────────────
def fetch_demand_data():
    """Fetch elexon demand data from Supabase."""
    supabase = get_supabase()
    response = supabase.table("elexon_demand") \
        .select("start_time, transmission_system_demand, national_demand") \
        .order("start_time") \
        .execute()
    df = pd.DataFrame(response.data)
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df = df.rename(columns={"start_time": "ds",
                             "transmission_system_demand": "y"})
    df["ds"] = df["ds"].dt.tz_localize(None)
    df = df.dropna(subset=["y"])
    df = df.sort_values("ds").reset_index(drop=True)
    logger.info(f"Fetched {len(df)} demand rows from {df['ds'].min()} to {df['ds'].max()}")
    return df

def fetch_weather_data():
    """Fetch weather data from Supabase."""
    supabase = get_supabase()
    response = supabase.table("weather_forecast") \
        .select("timestamp, temperature_2m, windspeed_10m, shortwave_radiation") \
        .order("timestamp") \
        .execute()
    df = pd.DataFrame(response.data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Fetched {len(df)} weather rows")
    return df

def merge_demand_weather(demand_df, weather_df):
    """Merge demand and weather on nearest hourly timestamp."""
    demand_df["ds_hour"] = demand_df["ds"].dt.floor("H")
    weather_df = weather_df.rename(columns={"timestamp": "ds_hour"})
    merged = pd.merge_asof(
        demand_df.sort_values("ds_hour"),
        weather_df.sort_values("ds_hour"),
        on="ds_hour",
        direction="nearest"
    )
    merged = merged.drop(columns=["ds_hour"])
    merged = merged.fillna(method="ffill").fillna(method="bfill")
    return merged

# ── Model training ─────────────────────────────────────────────────────────
def train_model(df):
    """Train Prophet model with weather regressors."""
    model = Prophet(
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        holidays_prior_scale=10,
        seasonality_mode="multiplicative",
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        interval_width=0.95
    )
    # Add weather regressors if available
    if "temperature_2m" in df.columns:
        model.add_regressor("temperature_2m")
    if "windspeed_10m" in df.columns:
        model.add_regressor("windspeed_10m")
    if "shortwave_radiation" in df.columns:
        model.add_regressor("shortwave_radiation")

    model.fit(df)
    logger.info("Prophet model trained successfully")
    return model

def build_future_df(model, df, periods=48, freq="30min"):
    """Build future dataframe for forecast."""
    future = model.make_future_dataframe(periods=periods, freq=freq)
    # Fill regressors with last known values (forward fill)
    for col in ["temperature_2m", "windspeed_10m", "shortwave_radiation"]:
        if col in df.columns:
            last_val = df[col].iloc[-1]
            future[col] = df.set_index("ds")[col].reindex(
                future["ds"], method="nearest", tolerance=pd.Timedelta("2H")
            ).fillna(last_val).values
    return future

# ── API Routes ─────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "Verdant Innovations — Energy Demand Forecast API",
        "version": "1.0.0",
        "endpoints": ["/forecast", "/forecast/latest", "/health"]
    })

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.utcnow().isoformat()})

@app.route("/forecast", methods=["GET"])
def forecast():
    """
    Generate 24-48hr demand forecast.
    Query params:
      - hours: number of hours ahead (default 48, max 168)
      - include_history: include historical fitted values (default false)
    """
    try:
        hours = min(int(request.args.get("hours", 48)), 168)
        include_history = request.args.get("include_history", "false").lower() == "true"
        periods = hours * 2  # 30-min intervals

        # Fetch data
        demand_df = fetch_demand_data()
        weather_df = fetch_weather_data()

        if len(demand_df) < 48:
            return jsonify({
                "error": "Insufficient data",
                "message": f"Need at least 48 rows, have {len(demand_df)}",
                "status": "error"
            }), 400

        # Merge and prepare
        if len(weather_df) > 0:
            merged_df = merge_demand_weather(demand_df, weather_df)
        else:
            merged_df = demand_df

        # Train model
        model = train_model(merged_df)

        # Generate forecast
        future = build_future_df(model, merged_df, periods=periods)
        forecast_df = model.predict(future)

        # Filter to forecast only (future periods)
        last_actual = merged_df["ds"].max()
        if include_history:
            result_df = forecast_df
        else:
            result_df = forecast_df[forecast_df["ds"] > last_actual]

        # Build response
        forecast_data = []
        for _, row in result_df.iterrows():
            forecast_data.append({
                "timestamp": row["ds"].isoformat(),
                "forecast_mw": round(float(row["yhat"]), 1),
                "lower_bound_mw": round(float(row["yhat_lower"]), 1),
                "upper_bound_mw": round(float(row["yhat_upper"]), 1),
                "trend": round(float(row["trend"]), 1),
            })

        # Summary stats
        forecasts = [f["forecast_mw"] for f in forecast_data]

        return jsonify({
            "status": "success",
            "generated_at": datetime.utcnow().isoformat(),
            "training_rows": len(merged_df),
            "training_from": merged_df["ds"].min().isoformat(),
            "training_to": merged_df["ds"].max().isoformat(),
            "forecast_hours": hours,
            "forecast_points": len(forecast_data),
            "summary": {
                "peak_demand_mw": max(forecasts) if forecasts else None,
                "min_demand_mw": min(forecasts) if forecasts else None,
                "avg_demand_mw": round(sum(forecasts)/len(forecasts), 1) if forecasts else None,
                "peak_timestamp": forecast_data[forecasts.index(max(forecasts))]["timestamp"] if forecasts else None,
            },
            "forecast": forecast_data
        })

    except Exception as e:
        logger.error(f"Forecast error: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/forecast/latest", methods=["GET"])
def forecast_latest():
    """Return just the next 6 hours — lightweight endpoint for dashboard."""
    try:
        demand_df = fetch_demand_data()
        weather_df = fetch_weather_data()

        if len(demand_df) < 48:
            return jsonify({"error": "Insufficient data"}), 400

        if len(weather_df) > 0:
            merged_df = merge_demand_weather(demand_df, weather_df)
        else:
            merged_df = demand_df

        model = train_model(merged_df)
        future = build_future_df(model, merged_df, periods=12)
        forecast_df = model.predict(future)

        last_actual = merged_df["ds"].max()
        result_df = forecast_df[forecast_df["ds"] > last_actual].head(12)

        forecast_data = []
        for _, row in result_df.iterrows():
            forecast_data.append({
                "timestamp": row["ds"].isoformat(),
                "forecast_mw": round(float(row["yhat"]), 1),
                "lower_bound_mw": round(float(row["yhat_lower"]), 1),
                "upper_bound_mw": round(float(row["yhat_upper"]), 1),
            })

        return jsonify({
            "status": "success",
            "generated_at": datetime.utcnow().isoformat(),
            "next_6_hours": forecast_data
        })

    except Exception as e:
        logger.error(f"Latest forecast error: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/data/summary", methods=["GET"])
def data_summary():
    """Return summary of data currently in Supabase."""
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
                "ready_for_training": len(demand_df) >= 48
            },
            "weather": {
                "rows": len(weather_df),
                "from": weather_df["timestamp"].min().isoformat() if len(weather_df) > 0 else None,
                "to": weather_df["timestamp"].max().isoformat() if len(weather_df) > 0 else None,
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

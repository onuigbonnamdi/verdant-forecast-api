# Evervia Innovations — Energy Demand Forecast API

Prophet-based time-series forecasting API for UK grid demand.
Pulls live data from Supabase, trains on the fly, returns 24-48hr forecasts.

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Service info and endpoint list |
| `/health` | GET | Health check |
| `/data/summary` | GET | Summary of data in Supabase |
| `/forecast` | GET | Full 48hr forecast (default) |
| `/forecast?hours=24` | GET | Custom forecast window (max 168hrs) |
| `/forecast?include_history=true` | GET | Include historical fitted values |
| `/forecast/latest` | GET | Next 6 hours only — lightweight |

## Example Response (/forecast/latest)

```json
{
  "status": "success",
  "generated_at": "2026-04-16T10:00:00",
  "next_6_hours": [
    {
      "timestamp": "2026-04-16T10:30:00",
      "forecast_mw": 28450.5,
      "lower_bound_mw": 27100.2,
      "upper_bound_mw": 29800.8
    }
  ]
}
```

## Deploy to Render

1. Push this folder to a GitHub repo
2. Go to render.com → New Web Service
3. Connect your GitHub repo
4. Set environment variables:
   - SUPABASE_URL = your Supabase project URL
   - SUPABASE_KEY = your Supabase anon key
5. Build command: `pip install -r requirements.txt`
6. Start command: `gunicorn app:app --workers 1 --timeout 120 --bind 0.0.0.0:$PORT`
7. Deploy

## Environment Variables Required

- `SUPABASE_URL` — from Supabase project settings
- `SUPABASE_KEY` — Supabase anon public key

## Model Details

- **Algorithm:** Facebook Prophet with additional weather regressors
- **Target:** transmission_system_demand (MW)
- **Regressors:** temperature_2m, windspeed_10m, shortwave_radiation
- **Seasonality:** daily + weekly (multiplicative)
- **Confidence interval:** 95%
- **Forecast resolution:** 30-minute intervals
- **Training data:** All available rows in elexon_demand table

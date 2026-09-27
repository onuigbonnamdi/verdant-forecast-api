# Verdant Forecast API

A Flask API that forecasts UK national electricity demand up to 7 days ahead at half-hourly resolution, using Elexon BMRS demand data and Open-Meteo weather stored in Supabase.

This is the first version of the forecasting service behind Evervia Innovations' products. It proved the approach; the production service has since moved on (see [Evolution](#evolution)).

## How it works

```
Supabase
  |-- elexon_demand     (transmission system demand, half-hourly)
  |-- weather_forecast  (temperature, wind speed, solar radiation)
        |
        v
Merge weather onto demand (nearest hour)
        |
        v
Feature engineering  -->  Gradient Boosting Regressor  -->  Recursive forecast
        |                                                        |
        v                                                        v
 time, lag, rolling,                               forecast + 95% interval
 and weather features                              per half-hour period
```

On each request the API pulls the latest data, trains the model, and forecasts forward one half-hour at a time, feeding each prediction back in as a lag value for the next step.

## Model

- **Algorithm:** scikit-learn `GradientBoostingRegressor` (200 trees, learning rate 0.05, max depth 4, subsample 0.8)
- **Target:** `transmission_system_demand` (MW)
- **Time features:** hour, minute, day of week, day of year, month, weekend flag, settlement period (1 to 48)
- **Demand features:** demand 24 and 48 hours earlier, 24-hour rolling mean and standard deviation
- **Weather features:** temperature, wind speed, shortwave (solar) radiation
- **Uncertainty:** 95% band from the standard deviation of training residuals
- **Resolution:** 30 minutes, up to 168 hours ahead

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Service info and endpoint list |
| `/health` | GET | Health check |
| `/data/summary` | GET | Rows and date coverage for demand and weather data, and training readiness |
| `/forecast` | GET | 48 hour forecast with peak, minimum, and average summary |
| `/forecast?hours=24` | GET | Custom horizon, up to 168 hours |
| `/forecast/latest` | GET | Next 6 hours only, lightweight |

### Example: `/forecast/latest`

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

## Running locally

```bash
pip install -r requirements.txt
export SUPABASE_URL=your_project_url
export SUPABASE_KEY=your_anon_key
python app.py
```

At least 96 rows (48 hours) of demand data are needed before the API will forecast.

## Deployment

Configured for Render (`render.yaml`, `Procfile`):

```
gunicorn app:app --workers 1 --timeout 120 --bind 0.0.0.0:$PORT
```

## Evolution

Running this version in practice exposed three limits worth knowing:

**Training on every request** made responses slow and costly.

**Recursive forecasting** lets errors compound over long horizons, so the 168 hour option is far less reliable than the 48 hour default.

**Residual-based intervals** from training data tend to be too narrow, since the model has already seen that data.

The production forecaster moved to a **Random Forest** model on Evervia's own server, with:

- **Scheduled training:** retrained weekly (Sundays 02:00 UTC) and served from a saved model, not trained per request
- **More history:** 18 months of half-hourly data (about 25,000 rows)
- **Richer features:** wholesale market price and volume, and cyclical (sine/cosine) encodings of time
- **Honest validation:** R² 0.9838 ± 0.0103 and MAE 399 MW, measured with 5-fold time-series cross-validation rather than on training data
- **Own database:** data moved from Supabase to a self-hosted Postgres instance

It powers live grid forecasts, peak and off-peak alerts, and savings insights in [GridSense](https://gridsense.evervia.co.uk). That service is private; this repo is kept as the public record of the first version.

## Tech stack

Python 3.11, Flask, pandas, NumPy, scikit-learn, Supabase, Gunicorn, Render

## Author

**Nnamdi Onuigbo**, Founder and AI Systems Engineer, Evervia Innovations Ltd

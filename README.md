# seguro-cafe-api

REST API for parametric coffee crop insurance risk scoring (Colombia — Risaralda & Cundinamarca).

## Requirements

- Python 3.11+
- Dependencies: `pip install -r requirements.txt`

## Quickstart

```bash
cp .env.example .env
pip install -r requirements.txt
uvicorn src.api.main:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

## Environment variables (`.env`)

| Variable | Default | Description |
|---|---|---|
| `MODELS_DIR` | `./insumos/models` | Path to trained model artifacts |
| `DATA_ANNUAL` | `./insumos/data/dataset_modelado_anual_limpio.csv` | Annual dataset (for training only) |
| `DATA_MONTHLY` | `./insumos/data/dataset_operativo_mensual_limpio.csv` | Monthly dataset (for training only) |
| `API_KEY` | — | Required header value for prediction endpoints |

## Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | No | API status and model metadata |
| `POST` | `/predict/annual` | `X-API-Key` | Annual loss magnitude + event detection + trigger |
| `POST` | `/predict/monthly` | `X-API-Key` | Monthly harvest scores + annualized alert |
| `GET` | `/data/history/{departamento}` | No | Historical series 2007–2024 |
| `GET` | `/calibrate/trigger` | No | Basis risk table across trigger thresholds |

`departamento` accepts `Cundinamarca` or `Risaralda`.

## Models

| Name | Algorithm | Features | Test metric |
|---|---|---|---|
| `magnitude_xgb` | XGBoostRegressor | `baseline_parsimonioso` — 10 vars | MAE 9.63 pp |
| `detector_trigger_hgb` | HistGradientBoostingRegressor | `set_A_interacc` — 18 vars | detector threshold −2.8% / trigger −14.0% |
| `monthly_hgb` | HistGradientBoostingRegressor | `mensual_core_lags` — ~50 vars | MAE annualized 11.08 pp |

Pre-trained artifacts are in `insumos/models/`. To retrain:

```bash
# Datasets must be present in insumos/data/
python scripts/run_pipeline.py
```

## Project layout

```
src/
  features/definitions.py   # feature constants and transformation functions
  models/train.py            # training pipeline
  models/predictor.py        # CafeteroPredictor class
  api/schemas.py             # Pydantic request/response models
  api/main.py                # FastAPI application
scripts/
  run_pipeline.py            # train + validate + save artifacts
insumos/
  models/                    # trained .pkl files (committed)
  data/                      # CSV datasets (.gitignore — training only)
```

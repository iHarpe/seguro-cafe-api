# seguro-cafe-api

API REST para estimación de riesgo climático en seguros paramétricos de café (Risaralda y Cundinamarca, Colombia).

## Inicio rápido

```bash
cp .env.example .env
pip install -r requirements.txt
uvicorn src.api.main:app --port 8000
```

Docs interactivos: http://localhost:8000/docs

## Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `MODELS_DIR` | `./insumos/models` | Artifacts `.pkl` entrenados |
| `DATA_ANNUAL` | `./insumos/data/dataset_modelado_anual_limpio.csv` | Dataset anual (solo entrenamiento) |
| `DATA_MONTHLY` | `./insumos/data/dataset_operativo_mensual_limpio.csv` | Dataset mensual (solo entrenamiento) |
| `API_KEY` | — | Header `X-API-Key` para endpoints de predicción |

## Endpoints

| Método | Ruta | Auth | Descripción |
|---|---|---|---|
| `GET` | `/health` | No | Estado de la API, métricas y freshness de modelos |
| `POST` | `/predict/annual` | Sí | Pérdida estimada (M1) + detección de evento + trigger (M2/M3) |
| `POST` | `/predict/monthly` | Sí | Score mensual de cosecha + alerta anualizada (M4) |
| `GET` | `/data/history/{dept}` | No | Serie histórica 2007-2024 con backtest |
| `GET` | `/data/backtest` | No | Resultados de backtest con filtros por depto y umbral |
| `GET` | `/data/monthly-history/{dept}` | No | Score M4 histórico completo |
| `GET` | `/data/oof` | No | Predicciones out-of-fold (y_true, y_pred_m1, y_pred_m3) |
| `GET` | `/data/correlations` | No | Correlaciones Pearson del Set A (18 vars) |
| `GET` | `/calibrate/trigger` | No | Métricas por umbral: recall, precision, F1, basis risk |

`dept` acepta `Risaralda` o `Cundinamarca`.

## Modelos

| Modelo | Algoritmo | Features | Métrica test |
|---|---|---|---|
| Magnitud (M1) | XGBoost | `baseline_parsimonioso` (10 vars) | MAE 9.63 pp |
| Detector/Trigger (M2/M3) | HGB | `set_A_interacc` (18 vars) | Detector -2.8% / Trigger -14.0% |
| Mensual (M4) | HGB | `mensual_core_lags` (~50 vars) | MAE 11.08 pp |

Artifacts pre-entrenados en `insumos/models/`. Para re-entrenar:

```bash
python scripts/run_pipeline.py
```

## Estructura

```
src/
  api/main.py                  # FastAPI (9 endpoints)
  api/schemas.py               # Pydantic request/response
  models/predictor.py          # CafeteroPredictor
  models/train.py              # Pipeline de entrenamiento
  features/definitions.py      # Constantes y transformaciones
scripts/run_pipeline.py        # Entrena y guarda artifacts
insumos/
  models/                      # .pkl commiteados
  data/                        # CSVs (gitignored)
```

## Requisitos

- Python 3.11+
- Dependencias: `pip install -r requirements.txt`

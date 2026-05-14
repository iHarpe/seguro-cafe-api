# seguro-cafe-api

API REST para la estimación de riesgo en seguros paramétricos de café (Colombia — Risaralda y Cundinamarca).

## Requisitos

- Python 3.11+
- `pip install -r requirements.txt`

## Inicio rápido

```bash
cp .env.example .env
pip install -r requirements.txt
uvicorn src.api.main:app --reload --port 8000
```

Documentación interactiva: http://localhost:8000/docs

## Variables de entorno (`.env`)

| Variable | Por defecto | Descripción |
|---|---|---|
| `MODELS_DIR` | `./insumos/models` | Ruta a los artifacts entrenados |
| `DATA_ANNUAL` | `./insumos/data/dataset_modelado_anual_limpio.csv` | Dataset anual (solo para entrenamiento) |
| `DATA_MONTHLY` | `./insumos/data/dataset_operativo_mensual_limpio.csv` | Dataset mensual (solo para entrenamiento) |
| `API_KEY` | — | Valor requerido en el header de los endpoints de predicción |

## Endpoints

| Método | Ruta | Auth | Descripción |
|---|---|---|---|
| `GET` | `/health` | No | Estado de la API y metadatos de los modelos |
| `POST` | `/predict/annual` | `X-API-Key` | Magnitud de pérdida anual + detección de evento + trigger |
| `POST` | `/predict/monthly` | `X-API-Key` | Scores mensuales de cosecha + alerta anualizada |
| `GET` | `/data/history/{departamento}` | No | Serie histórica 2007–2024 |
| `GET` | `/calibrate/trigger` | No | Tabla de basis risk por umbral de trigger |

`departamento` acepta `Cundinamarca` o `Risaralda`.

## Modelos

| Nombre | Algoritmo | Features | Métrica test |
|---|---|---|---|
| `magnitude_xgb` | XGBoostRegressor | `baseline_parsimonioso` — 10 vars | MAE 9.63 pp |
| `detector_trigger_hgb` | HistGradientBoostingRegressor | `set_A_interacc` — 18 vars | umbral detector −2.8% / trigger −14.0% |
| `monthly_hgb` | HistGradientBoostingRegressor | `mensual_core_lags` — ~50 vars | MAE anualizado 11.08 pp |

Los artifacts pre-entrenados están en `insumos/models/`. Para re-entrenar:

```bash
# Los datasets deben estar en insumos/data/
python scripts/run_pipeline.py
```

## Estructura del proyecto

```
src/
  features/definitions.py   # constantes de features y funciones de transformación
  models/train.py            # pipeline de entrenamiento
  models/predictor.py        # clase CafeteroPredictor
  api/schemas.py             # modelos Pydantic de request/response
  api/main.py                # aplicación FastAPI
scripts/
  run_pipeline.py            # entrena, valida y guarda artifacts
insumos/
  models/                    # archivos .pkl entrenados (commiteados)
  data/                      # datasets CSV (.gitignore — solo para entrenamiento)
```

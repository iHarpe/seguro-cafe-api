# seguro-cafe-api

API REST para el modelo de seguro agrícola indexado en el sector cafetero colombiano.

## Estructura

```
seguro-cafe-api/
  insumos/
    models/    ← artifacts .pkl (commiteados, requeridos para correr la API)
    data/      ← datasets CSV (en .gitignore; solo necesarios para re-entrenar)
  src/
  scripts/
```

## Setup

```bash
cp .env.example .env   # las rutas por defecto ya apuntan a insumos/
pip install -r requirements.txt
```

## Entrenar / re-entrenar modelos (opcional)

Solo necesario si se quieren actualizar los artifacts. Los datos deben estar en `insumos/data/`.

```bash
python scripts/run_pipeline.py
```

Guarda los artifacts en `insumos/models/`.

## Correr la API

```bash
uvicorn src.api.main:app --reload --port 8000
```

Docs interactivas: http://localhost:8000/docs

## Endpoints

| Método | Ruta | Auth | Descripción |
|---|---|---|---|
| GET | `/health` | No | Estado de la API |
| POST | `/predict/annual` | API Key | Magnitud + detector + trigger |
| POST | `/predict/monthly` | API Key | Score mensual + alerta |
| GET | `/data/history/{dept}` | No | Serie histórica del departamento |
| GET | `/calibrate/trigger` | No | Tabla basis risk por umbral |

Autenticación: header `X-API-Key: <valor de API_KEY en .env>`

## Modelos

| Modelo | Algoritmo | Feature set | Métrica test |
|---|---|---|---|
| Magnitud | XGBoostRegressor | baseline_parsimonioso (10 vars) | MAE 9.63 pp |
| Detector+Trigger | HGB | set_A_interacc (18 vars) | umbral -2.8% / -14.0% |
| Mensual | HGB | mensual_core_lags | MAE anualizado ~11 pp |

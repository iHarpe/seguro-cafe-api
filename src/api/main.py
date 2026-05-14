"""FastAPI application for the Seguro Agrícola Indexado model API."""
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.models.predictor import CafeteroPredictor
from src.api.schemas import (
    AnnualPredictionInput,
    AnnualPredictionOutput,
    MonthlyPredictionInput,
    MonthlyPredictionOutput,
    HealthResponse,
)
from src.features.definitions import TRIGGER_THRESHOLD

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]  # src/api/main.py → repo root
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(_REPO_ROOT / "insumos/models")))
API_KEY = os.getenv("API_KEY", "")
DATA_ANNUAL = Path(
    os.getenv("DATA_ANNUAL",
              str(_REPO_ROOT / "insumos/data/dataset_modelado_anual_limpio.csv"))
)

predictor = CafeteroPredictor(MODELS_DIR)

app = FastAPI(
    title="Seguro Agrícola Indexado — API",
    description="API para el modelo de seguro paramétrico de café en Colombia.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_api_key(x_api_key: Optional[str]) -> None:
    if not API_KEY:
        return  # no key configured → open in dev mode
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key inválida")


def _require_models_ready() -> None:
    if not predictor.is_ready():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Modelos no cargados. Verifique logs de startup.")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup() -> None:
    try:
        predictor.load()
        print(f"[startup] Modelos cargados desde {MODELS_DIR.resolve()}")
    except Exception as exc:
        print(f"[startup] ERROR al cargar modelos: {exc}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["sistema"])
async def health() -> HealthResponse:
    meta = predictor.get_metadata()
    return HealthResponse(
        status="ok" if predictor.is_ready() else "modelos_no_cargados",
        models_loaded=predictor.is_ready(),
        trained_at=meta.get("trained_at"),
        train_end_year=meta.get("train_end_year"),
    )


@app.post("/predict/annual", response_model=AnnualPredictionOutput, tags=["predicción"])
async def predict_annual(
    body: AnnualPredictionInput,
    x_api_key: Optional[str] = Header(default=None),
) -> AnnualPredictionOutput:
    _require_api_key(x_api_key)
    _require_models_ready()
    result = predictor.predict_annual(body.model_dump())
    return AnnualPredictionOutput(**result)


@app.post("/predict/monthly", response_model=MonthlyPredictionOutput, tags=["predicción"])
async def predict_monthly(
    body: MonthlyPredictionInput,
    x_api_key: Optional[str] = Header(default=None),
) -> MonthlyPredictionOutput:
    _require_api_key(x_api_key)
    _require_models_ready()
    records = [
        r.model_dump(by_alias=False) for r in body.records
    ]
    # Rename 'def_' → 'def' after Pydantic serialization
    for rec in records:
        if "def_" in rec:
            rec["def"] = rec.pop("def_")
    result = predictor.predict_monthly(records)
    return MonthlyPredictionOutput(**result)


@app.get("/data/history/{departamento}", tags=["datos"])
async def data_history(departamento: str) -> JSONResponse:
    if departamento not in ("Cundinamarca", "Risaralda"):
        raise HTTPException(status_code=400, detail="Departamento debe ser Cundinamarca o Risaralda")
    if not DATA_ANNUAL.exists():
        raise HTTPException(status_code=503, detail="Dataset histórico no disponible")
    df = pd.read_csv(DATA_ANNUAL, sep=";")
    cols = ["anio", "perdida_rendimiento_anual_pct", "rendimiento_t_ha", "evento_perdida_anual"]
    present = [c for c in cols if c in df.columns]
    hist = df[df["departamento"] == departamento][present].dropna(subset=["anio"])
    return JSONResponse(content=hist.to_dict(orient="records"))


@app.get("/calibrate/trigger", tags=["actuarial"])
async def calibrate_trigger() -> JSONResponse:
    """Returns basis-risk table across trigger thresholds from -25% to +5%."""
    _require_models_ready()
    if not DATA_ANNUAL.exists():
        raise HTTPException(status_code=503, detail="Dataset histórico no disponible")

    from src.features.definitions import (
        FEATURES_SET_A, add_es_risaralda, build_set_a_interactions,
        TEST_START_YEAR,
    )
    import numpy as np

    df = pd.read_csv(DATA_ANNUAL, sep=";")
    df = add_es_risaralda(df)
    df = build_set_a_interactions(df)
    df = df[df["anio"] >= TEST_START_YEAR].dropna(subset=["perdida_rendimiento_anual_pct"])

    y_true = df["perdida_rendimiento_anual_pct"].values
    y_pred = predictor._det_model.predict(df[FEATURES_SET_A])

    thresholds = [t / 10 for t in range(-250, 51, 10)]  # -25.0 to +5.0
    rows = []
    for thr in thresholds:
        mask = (y_true <= thr) | (y_pred <= thr)
        if mask.sum() == 0:
            br = None
        else:
            from sklearn.metrics import mean_absolute_error
            br = round(float(mean_absolute_error(y_true[mask], y_pred[mask])), 4)
        actual_pos = int((y_true <= thr).sum())
        pred_pos = int((y_pred <= thr).sum())
        recall = round(float(np.sum((y_true <= thr) & (y_pred <= thr)) / actual_pos), 4) if actual_pos > 0 else None
        rows.append({
            "threshold_pct": thr,
            "n_actual_events": actual_pos,
            "n_predicted_events": pred_pos,
            "recall": recall,
            "basis_risk_pp": br,
        })
    return JSONResponse(content=rows)

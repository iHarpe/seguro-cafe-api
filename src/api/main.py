"""FastAPI application for the Seguro Agrícola Indexado model API."""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query, status
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
DATA_DIR = _REPO_ROOT / "insumos" / "data"

predictor = CafeteroPredictor(MODELS_DIR)

app = FastAPI(
    title="Seguro Agrícola Indexado — API",
    description="API para el modelo de seguro paramétrico de café en Colombia.",
    version="2.0.0",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_api_key(x_api_key: Optional[str]) -> None:
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="API key inválida")


def _require_models_ready() -> None:
    if not predictor.is_ready():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Modelos no cargados. Verifique logs de startup.")


def _load_csv(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=503,
                            detail=f"{filename} no disponible. Ejecutar pipeline.")
    return pd.read_csv(path)


def _df_to_json(df: pd.DataFrame) -> list:
    """Convert DataFrame to JSON-safe list of dicts (NaN -> None)."""
    return json.loads(df.to_json(orient="records"))


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
# System
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["sistema"])
async def health() -> HealthResponse:
    meta = predictor.get_metadata()
    trained_at_str = meta.get("trained_at")
    freshness_days = None
    if trained_at_str:
        try:
            trained_dt = datetime.fromisoformat(trained_at_str)
            freshness_days = (datetime.now() - trained_dt).days
        except (ValueError, TypeError):
            pass
    return HealthResponse(
        status="ok" if predictor.is_ready() else "modelos_no_cargados",
        models_loaded=predictor.is_ready(),
        trained_at=trained_at_str,
        train_end_year=meta.get("train_end_year"),
        metrics=meta.get("models"),
        pipeline_last_run=trained_at_str,
        data_freshness_days=freshness_days,
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@app.post("/predict/annual", response_model=AnnualPredictionOutput,
          tags=["predicción"])
async def predict_annual(
    body: AnnualPredictionInput,
    x_api_key: Optional[str] = Header(default=None),
) -> AnnualPredictionOutput:
    _require_api_key(x_api_key)
    _require_models_ready()
    result = predictor.predict_annual(body.model_dump())
    return AnnualPredictionOutput(**result)


@app.post("/predict/monthly", response_model=MonthlyPredictionOutput,
          tags=["predicción"])
async def predict_monthly(
    body: MonthlyPredictionInput,
    x_api_key: Optional[str] = Header(default=None),
) -> MonthlyPredictionOutput:
    _require_api_key(x_api_key)
    _require_models_ready()
    records = [r.model_dump(by_alias=False) for r in body.records]
    for rec in records:
        if "def_" in rec:
            rec["def"] = rec.pop("def_")
    result = predictor.predict_monthly(records)
    return MonthlyPredictionOutput(**result)


# ---------------------------------------------------------------------------
# Historical data
# ---------------------------------------------------------------------------

@app.get("/data/history/{departamento}", tags=["datos"])
async def data_history(departamento: str) -> JSONResponse:
    """Historical annual data with model predictions (if backtest available)."""
    if departamento not in ("Cundinamarca", "Risaralda"):
        raise HTTPException(400, "Departamento debe ser Cundinamarca o Risaralda")
    if not DATA_ANNUAL.exists():
        raise HTTPException(503, "Dataset histórico no disponible")

    df = pd.read_csv(DATA_ANNUAL, sep=";")
    cols = ["anio", "perdida_rendimiento_anual_pct", "rendimiento_t_ha",
            "evento_perdida_anual"]
    present = [c for c in cols if c in df.columns]
    hist = df[df["departamento"] == departamento][present].dropna(subset=["anio"])

    backtest_path = DATA_DIR / "backtest_resultados.csv"
    if backtest_path.exists():
        bt = pd.read_csv(backtest_path)
        bt_default = bt[
            (bt["departamento"] == departamento) &
            (bt["umbral_pct"] == int(TRIGGER_THRESHOLD))
        ][["anio", "prediccion_m1_pct", "prediccion_m3_pct"]].drop_duplicates()
        hist = hist.merge(bt_default, on="anio", how="left")

    return JSONResponse(content=hist.pipe(_df_to_json))


@app.get("/data/backtest", tags=["datos"])
async def data_backtest(
    departamento: Optional[str] = Query(default=None),
    umbral_pct: Optional[float] = Query(default=None),
) -> JSONResponse:
    """Backtest results 2007-2024 by depto × year × threshold config."""
    df = _load_csv("backtest_resultados.csv")
    if departamento:
        df = df[df["departamento"] == departamento]
    if umbral_pct is not None:
        df = df[df["umbral_pct"] == umbral_pct]
    return JSONResponse(content=df.pipe(_df_to_json))


@app.get("/data/monthly-history/{departamento}", tags=["datos"])
async def monthly_score_history(departamento: str) -> JSONResponse:
    """Full historical M4 monthly score trajectory."""
    if departamento not in ("Cundinamarca", "Risaralda"):
        raise HTTPException(400, "Departamento debe ser Cundinamarca o Risaralda")
    df = _load_csv("score_operacional_mensual_historico.csv")
    df = df[df["departamento"] == departamento]
    return JSONResponse(content=df.pipe(_df_to_json))


@app.get("/data/oof", tags=["datos"])
async def oof_predictions() -> JSONResponse:
    """Out-of-fold predictions for the actuarial simulator (unbiased)."""
    df = _load_csv("predicciones_oof.csv")
    return JSONResponse(content=df.pipe(_df_to_json))


@app.get("/data/correlations", tags=["datos"])
async def data_correlations() -> JSONResponse:
    """Pearson correlations of Set A features vs annual loss."""
    corr_path = DATA_DIR / "correlations.json"
    if not corr_path.exists():
        raise HTTPException(503, "Correlaciones no disponibles. Ejecutar pipeline.")
    with open(corr_path, encoding="utf-8") as f:
        return JSONResponse(content=json.load(f))


# ---------------------------------------------------------------------------
# Actuarial
# ---------------------------------------------------------------------------

@app.get("/calibrate/trigger", tags=["actuarial"])
async def calibrate_trigger() -> JSONResponse:
    """Aggregated metrics per trigger threshold from precomputed backtest."""
    df = _load_csv("backtest_resultados.csv")

    rows = []
    for thr, grp in df.groupby("umbral_pct"):
        n_events = int(grp["evento_real"].sum())
        n_triggered = int(grp["trigger_activado"].sum())
        tp = int((grp["evento_real"] & grp["trigger_activado"]).sum())

        recall = tp / n_events if n_events > 0 else None
        precision = tp / n_triggered if n_triggered > 0 else None
        f1 = None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)

        active = grp["evento_real"] | grp["trigger_activado"]
        br_mean = (float(grp.loc[active, "basis_risk_pp"].abs().mean())
                   if active.sum() > 0 else None)

        rows.append({
            "threshold_pct": float(thr),
            "recall": round(recall, 4) if recall is not None else None,
            "precision": round(precision, 4) if precision is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "basis_risk_medio_pp": round(br_mean, 4) if br_mean is not None else None,
            "n_actual_events": n_events,
            "n_predicted_events": n_triggered,
        })

    return JSONResponse(content=rows)

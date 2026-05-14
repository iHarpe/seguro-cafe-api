"""Pydantic schemas for all API endpoints."""
from typing import Literal, Optional
from pydantic import BaseModel, Field


class AnnualPredictionInput(BaseModel):
    departamento: Literal["Cundinamarca", "Risaralda"]
    anio: int = Field(..., ge=2000, le=2030)
    precio_ico_usd_ton: float
    # Annual climate
    precipitation_annual_sum: float
    temp_aire_C_annual_mean: float
    def_annual_mean: float
    GDD_cafe_annual_mean: float
    NDVI_anomalia_pct_annual_mean: float
    # Harvest season
    precipitation_cosecha_sum: float
    temp_aire_C_cosecha_mean: float
    def_cosecha_mean: float
    NDVI_anomalia_pct_cosecha_mean: float
    # Required extras for Set A interactions
    Gpp_anomalia_pct_annual_mean: float
    Gpp_cosecha_mean: float
    NDVI_cosecha_mean: float
    aet_cosecha_mean: float
    pet_cosecha_mean: float
    LST_Day_1km_cosecha_mean: float
    LST_Night_1km_cosecha_mean: float


class AnnualPredictionOutput(BaseModel):
    departamento: str
    anio: Optional[int]
    perdida_estimada_pct: float
    evento_detectado: bool
    trigger_activado: bool
    nivel_alerta: str
    basis_risk_estimado_pp: float
    modelo_magnitud: str
    modelo_detector_trigger: str
    umbral_detector_pct: float
    umbral_trigger_pct: float


class MonthlyRecord(BaseModel):
    departamento: str
    anio: int
    mes: int = Field(..., ge=1, le=12)
    es_mes_cosecha: int = Field(..., ge=0, le=1)
    factor_mensual: float
    precio_ico_usd_ton: float
    precio_productor_usd_ton: float
    elevacion_media_m: float
    pendiente_media: float
    precipitation: float
    temp_aire_C: float
    def_: float = Field(..., alias="def")
    GDD_cafe: float
    NDVI: float
    EVI: float
    Gpp: float
    NDVI_anomalia_pct: float
    EVI_anomalia_pct: float
    Gpp_anomalia_pct: float

    model_config = {"populate_by_name": True}


class MonthlyPredictionInput(BaseModel):
    records: list[MonthlyRecord] = Field(..., min_length=3)


class MonthlyScoreEntry(BaseModel):
    mes: int
    anio: int
    score: Optional[float]
    es_cosecha: int


class MonthlyPredictionOutput(BaseModel):
    departamento: str
    score_anualizado: Optional[float]
    alerta_activa: bool
    nivel_alerta: str
    scores_por_mes: list[dict]
    modelo_mensual: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    trained_at: Optional[str] = None
    train_end_year: Optional[int] = None

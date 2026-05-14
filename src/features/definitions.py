"""Feature constants and pure transformation functions for the coffee insurance models."""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

FEATURES_MAGNITUDE = [
    "es_risaralda",
    "precio_ico_usd_ton",
    "precipitation_annual_sum",
    "temp_aire_C_annual_mean",
    "def_annual_mean",
    "GDD_cafe_annual_mean",
    "NDVI_anomalia_pct_annual_mean",
    "precipitation_cosecha_sum",
    "temp_aire_C_cosecha_mean",
    "NDVI_anomalia_pct_cosecha_mean",
]

FEATURES_SET_A_BASE = [
    "es_risaralda",
    "precio_ico_usd_ton",
    "precipitation_annual_sum",
    "temp_aire_C_annual_mean",
    "def_annual_mean",
    "GDD_cafe_annual_mean",
    "NDVI_anomalia_pct_annual_mean",
    "precipitation_cosecha_sum",
    "temp_aire_C_cosecha_mean",
    "NDVI_anomalia_pct_cosecha_mean",
]

FEATURES_SET_A = [
    # Baseline (10)
    "es_risaralda",
    "precio_ico_usd_ton",
    "precipitation_annual_sum",
    "temp_aire_C_annual_mean",
    "def_annual_mean",
    "GDD_cafe_annual_mean",
    "NDVI_anomalia_pct_annual_mean",
    "precipitation_cosecha_sum",
    "temp_aire_C_cosecha_mean",
    "NDVI_anomalia_pct_cosecha_mean",
    # Interacciones clima×clima (3)
    "temp_x_def_annual",
    "precip_x_ndvi_anom",
    "GDD_x_gpp_anom",
    # Índices compuestos satelitales (4)
    "gpp_ndvi_ratio_cosecha",
    "stress_hidrico",
    "balance_hidrico",
    "amplitud_termica_cosecha",
    # Variable auxiliar para interacciones (1)
    "Gpp_anomalia_pct_annual_mean",
]

CORE_STATIC_MONTHLY = [
    "es_risaralda",
    "mes",
    "mes_sin",
    "mes_cos",
    "es_mes_cosecha",
    "factor_mensual",
    "precio_ico_usd_ton",
    "precio_productor_usd_ton",
    "elevacion_media_m",
    "pendiente_media",
]

CORE_DYNAMIC_MONTHLY = [
    "precipitation",
    "temp_aire_C",
    "def",
    "GDD_cafe",
    "NDVI",
    "EVI",
    "Gpp",
    "NDVI_anomalia_pct",
    "EVI_anomalia_pct",
    "Gpp_anomalia_pct",
]

# ---------------------------------------------------------------------------
# Operational thresholds
# ---------------------------------------------------------------------------

LOSS_THRESHOLD = -15.0       # event classification threshold (%)
TRIGGER_THRESHOLD = -14.0    # insurance payout threshold (%)
DETECTOR_THRESHOLD = -2.8    # early-warning threshold (%)

# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

TRAIN_END_YEAR = 2020
TEST_START_YEAR = 2021


# ---------------------------------------------------------------------------
# Transformation functions
# ---------------------------------------------------------------------------

def add_es_risaralda(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["es_risaralda"] = (df["departamento"] == "Risaralda").astype(int)
    return df


def build_set_a_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["temp_x_def_annual"] = df["temp_aire_C_annual_mean"] * df["def_annual_mean"]
    df["precip_x_ndvi_anom"] = (
        df["precipitation_annual_sum"] * df["NDVI_anomalia_pct_annual_mean"]
    )
    df["GDD_x_gpp_anom"] = (
        df["GDD_cafe_annual_mean"] * df["Gpp_anomalia_pct_annual_mean"]
    )
    # Guard against division by zero in ratios
    ndvi_cosecha = df["NDVI_cosecha_mean"].replace(0, np.nan)
    precip_annual = df["precipitation_annual_sum"].replace(0, np.nan)
    df["gpp_ndvi_ratio_cosecha"] = df["Gpp_cosecha_mean"] / ndvi_cosecha
    df["stress_hidrico"] = df["def_annual_mean"] / precip_annual
    df["balance_hidrico"] = df["aet_cosecha_mean"] - df["pet_cosecha_mean"]
    df["amplitud_termica_cosecha"] = (
        df["LST_Day_1km_cosecha_mean"] - df["LST_Night_1km_cosecha_mean"]
    )
    return df


def build_monthly_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag1, lag2, roll3 for each CORE_DYNAMIC variable, grouped by departamento."""
    df = df.copy()
    df = df.sort_values(["departamento", "anio", "mes"]).reset_index(drop=True)

    # Cyclic month encoding
    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)

    for col in CORE_DYNAMIC_MONTHLY:
        if col not in df.columns:
            continue
        grp = df.groupby("departamento")[col]
        df[f"{col}_lag1"] = grp.shift(1)
        df[f"{col}_lag2"] = grp.shift(2)
        df[f"{col}_roll3"] = grp.transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
    return df


def get_monthly_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the monthly feature columns that are actually present in df."""
    lag_cols = [
        f"{col}_{suffix}"
        for col in CORE_DYNAMIC_MONTHLY
        for suffix in ("lag1", "lag2", "roll3")
        if f"{col}_{suffix}" in df.columns
    ]
    present_static = [c for c in CORE_STATIC_MONTHLY if c in df.columns]
    present_dynamic = [c for c in CORE_DYNAMIC_MONTHLY if c in df.columns]
    return present_static + present_dynamic + lag_cols

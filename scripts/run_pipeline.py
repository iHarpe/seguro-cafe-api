"""Pipeline: train models + generate all data artifacts for the API."""
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.train import run_training
from src.features.definitions import (
    FEATURES_MAGNITUDE,
    FEATURES_SET_A,
    LOSS_THRESHOLD,
    TRIGGER_THRESHOLD,
    TRAIN_END_YEAR,
    add_es_risaralda,
    build_set_a_interactions,
    build_monthly_lags,
)

FEATURE_LABELS = {
    "es_risaralda": "Departamento (Risaralda=1)",
    "precio_ico_usd_ton": "Precio ICO (USD/ton)",
    "precipitation_annual_sum": "Precipitación anual total (mm)",
    "temp_aire_C_annual_mean": "Temperatura media anual (°C)",
    "def_annual_mean": "Déficit hídrico anual medio (mm)",
    "GDD_cafe_annual_mean": "Grados día café (anual)",
    "NDVI_anomalia_pct_annual_mean": "Anomalía NDVI anual (%)",
    "precipitation_cosecha_sum": "Precipitación cosecha total (mm)",
    "temp_aire_C_cosecha_mean": "Temperatura media cosecha (°C)",
    "NDVI_anomalia_pct_cosecha_mean": "Anomalía NDVI cosecha (%)",
    "temp_x_def_annual": "Interacción: Temp × Déficit hídrico",
    "precip_x_ndvi_anom": "Interacción: Precip × Anomalía NDVI",
    "GDD_x_gpp_anom": "Interacción: GDD × Anomalía GPP",
    "gpp_ndvi_ratio_cosecha": "Ratio GPP/NDVI cosecha",
    "stress_hidrico": "Índice estrés hídrico",
    "balance_hidrico": "Balance hídrico cosecha (AET−PET)",
    "amplitud_termica_cosecha": "Amplitud térmica cosecha (°C)",
    "Gpp_anomalia_pct_annual_mean": "Anomalía GPP anual (%)",
}


def _load_paths() -> dict:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    repo_root = Path(__file__).resolve().parents[1]
    return {
        "data_annual": Path(os.getenv(
            "DATA_ANNUAL",
            str(repo_root / "insumos/data/dataset_modelado_anual_limpio.csv"),
        )),
        "data_monthly": Path(os.getenv(
            "DATA_MONTHLY",
            str(repo_root / "insumos/data/dataset_operativo_mensual_limpio.csv"),
        )),
        "models_dir": Path(os.getenv(
            "MODELS_DIR", str(repo_root / "insumos/models"),
        )),
        "data_dir": repo_root / "insumos" / "data",
    }


def _prepare_annual(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df = add_es_risaralda(df)
    df = build_set_a_interactions(df)
    return df[df["perdida_rendimiento_anual_pct"].notna()].copy()


def _prepare_monthly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df = add_es_risaralda(df)
    return build_monthly_lags(df)


# ── Artifact 1: Backtest ─────────────────────────────────────────────────

def generate_backtest(df_annual: pd.DataFrame, models_dir: Path,
                      data_dir: Path) -> pd.DataFrame:
    """Backtest on all 36 obs × 31 thresholds = 1116 rows (long format)."""
    mag_model = joblib.load(models_dir / "magnitude_xgb.pkl")
    det_model = joblib.load(models_dir / "detector_trigger_hgb.pkl")

    pred_m1 = mag_model.predict(df_annual[FEATURES_MAGNITUDE])
    pred_m3 = det_model.predict(df_annual[FEATURES_SET_A])

    base = df_annual[[
        "anio", "departamento", "perdida_rendimiento_anual_pct",
        "rendimiento_t_ha", "precio_ico_usd_ton",
        "area_cosechada_ha", "rendimiento_medio_t_ha",
    ]].copy()
    base["prediccion_m1_pct"] = pred_m1
    base["prediccion_m3_pct"] = pred_m3
    base["split"] = np.where(base["anio"] <= TRAIN_END_YEAR, "train", "test")
    base["produccion_t_ref"] = base["rendimiento_medio_t_ha"] * base["area_cosechada_ha"]

    thresholds = list(range(-25, 6))  # -25% to +5%
    rows = []
    for _, r in base.iterrows():
        for thr in thresholds:
            triggered = bool(r["prediccion_m3_pct"] <= thr)
            pago_pp = max(0.0, -r["prediccion_m3_pct"]) if triggered else 0.0
            perdida = r["perdida_rendimiento_anual_pct"]
            br_pp = pago_pp + perdida  # positive = overpaid, negative = underpaid
            usd_factor = r["produccion_t_ref"] * r["precio_ico_usd_ton"] / 100_000

            rows.append({
                "anio": int(r["anio"]),
                "departamento": r["departamento"],
                "umbral_pct": thr,
                "perdida_real_pct": round(perdida, 4),
                "prediccion_m1_pct": round(float(r["prediccion_m1_pct"]), 4),
                "prediccion_m3_pct": round(float(r["prediccion_m3_pct"]), 4),
                "evento_real": bool(perdida <= LOSS_THRESHOLD),
                "trigger_activado": triggered,
                "pago_pp": round(pago_pp, 4),
                "basis_risk_pp": round(br_pp, 4),
                "rendimiento_t_ha": round(float(r["rendimiento_t_ha"]), 4),
                "produccion_t_ref": round(float(r["produccion_t_ref"]), 2),
                "precio_ico_usd_ton": float(r["precio_ico_usd_ton"]),
                "pago_usd_k": round(pago_pp * usd_factor, 4),
                "basis_risk_usd_k": round(br_pp * usd_factor, 4),
                "split": r["split"],
            })

    result = pd.DataFrame(rows)
    out = data_dir / "backtest_resultados.csv"
    result.to_csv(out, index=False)

    bt14 = result[result["umbral_pct"] == -14]
    n_events = int(bt14["evento_real"].sum())
    n_hit = int((bt14["evento_real"] & bt14["trigger_activado"]).sum())
    recall = n_hit / n_events if n_events else 0
    active = bt14["evento_real"] | bt14["trigger_activado"]
    br_mean = bt14.loc[active, "basis_risk_pp"].abs().mean() if active.sum() else 0
    print(f"[backtest] {len(result)} rows -> {out}")
    print(f"  @-14%: recall={recall:.2f} ({n_hit}/{n_events}), |BR| medio={br_mean:.2f} pp")
    return result


# ── Artifact 2: OOF Predictions ──────────────────────────────────────────

def generate_oof(df_annual: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """OOF predictions via TimeSeriesSplit(5) for M1 (XGBoost) and M3 (HGB)."""
    years = sorted(df_annual["anio"].unique())
    tscv = TimeSeriesSplit(n_splits=5)

    oof_rows = []
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(years)):
        tr_years = {years[i] for i in tr_idx}
        va_years = {years[i] for i in va_idx}
        tr = df_annual[df_annual["anio"].isin(tr_years)]
        va = df_annual[df_annual["anio"].isin(va_years)]
        y_tr = tr["perdida_rendimiento_anual_pct"]

        m1 = XGBRegressor(
            max_depth=3, learning_rate=0.03, n_estimators=400,
            reg_lambda=1.0, subsample=1.0, random_state=42, verbosity=0,
        )
        m1.fit(tr[FEATURES_MAGNITUDE], y_tr)
        p1 = m1.predict(va[FEATURES_MAGNITUDE])

        m3 = HistGradientBoostingRegressor(
            max_depth=2, learning_rate=0.05, max_iter=200,
            min_samples_leaf=1, l2_regularization=1.0, random_state=42,
        )
        m3.fit(tr[FEATURES_SET_A], y_tr)
        p3 = m3.predict(va[FEATURES_SET_A])

        for i, (_, row) in enumerate(va.iterrows()):
            oof_rows.append({
                "anio": int(row["anio"]),
                "departamento": row["departamento"],
                "fold": fold,
                "y_true": round(float(row["perdida_rendimiento_anual_pct"]), 4),
                "y_pred_m1": round(float(p1[i]), 4),
                "y_pred_m3": round(float(p3[i]), 4),
            })

    result = pd.DataFrame(oof_rows)
    out = data_dir / "predicciones_oof.csv"
    result.to_csv(out, index=False)

    n_ev = int((result["y_true"] <= TRIGGER_THRESHOLD).sum())
    n_hit = int(
        ((result["y_true"] <= TRIGGER_THRESHOLD) &
         (result["y_pred_m3"] <= TRIGGER_THRESHOLD)).sum()
    )
    print(f"[oof] {len(result)} rows -> {out}")
    if n_ev:
        print(f"  OOF recall@{TRIGGER_THRESHOLD}%: {n_hit / n_ev:.2f} ({n_hit}/{n_ev})")
    return result


# ── Artifact 3: Monthly Score History ────────────────────────────────────

def generate_monthly_history(df_monthly: pd.DataFrame,
                             models_dir: Path,
                             data_dir: Path) -> pd.DataFrame:
    """Historical monthly M4 scores for all available rows."""
    mon_model = joblib.load(models_dir / "monthly_hgb.pkl")
    mon_features = joblib.load(models_dir / "monthly_features.pkl")
    feature_cols = [c for c in mon_features if c in df_monthly.columns]

    df = df_monthly.copy()
    valid_mask = df[feature_cols].notna().all(axis=1)
    df["score_m4"] = np.nan
    if valid_mask.any():
        df.loc[valid_mask, "score_m4"] = mon_model.predict(df.loc[valid_mask, feature_cols])

    df["percentil_historico"] = (
        df.groupby(["mes", "departamento"])["score_m4"]
        .rank(pct=True).mul(100)
    )

    result = df[["anio", "mes", "departamento", "score_m4",
                 "es_mes_cosecha", "percentil_historico"]].copy()
    result["score_m4"] = result["score_m4"].round(4)
    result["percentil_historico"] = result["percentil_historico"].round(1)

    out = data_dir / "score_operacional_mensual_historico.csv"
    result.to_csv(out, index=False)
    n_valid = int(result["score_m4"].notna().sum())
    print(f"[monthly] {len(result)} rows ({n_valid} with score) -> {out}")
    return result


# ── Artifact 4: Correlations ─────────────────────────────────────────────

def generate_correlations(df_annual: pd.DataFrame,
                          data_dir: Path) -> list[dict]:
    """Pearson correlations of Set A features vs target."""
    target = df_annual["perdida_rendimiento_anual_pct"].values
    corrs: list[dict] = []
    for feat in FEATURES_SET_A:
        if feat not in df_annual.columns:
            continue
        x = df_annual[feat].values
        mask = np.isfinite(x) & np.isfinite(target)
        if mask.sum() < 5:
            continue
        r, p = stats.pearsonr(x[mask], target[mask])
        corrs.append({
            "variable": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "pearson_r": round(float(r), 4),
            "p_value": round(float(p), 4),
            "significant": bool(p < 0.05),
        })
    corrs.sort(key=lambda c: abs(c["pearson_r"]), reverse=True)

    out = data_dir / "correlations.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(corrs, f, indent=2, ensure_ascii=False)
    print(f"[correlations] {len(corrs)} features -> {out}")
    return corrs


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("PIPELINE: Entrenamiento + Generación de artefactos")
    print("=" * 60)

    metadata = run_training(verbose=True)
    print()

    paths = _load_paths()
    df_annual = _prepare_annual(paths["data_annual"])
    df_monthly = _prepare_monthly(paths["data_monthly"])
    paths["data_dir"].mkdir(parents=True, exist_ok=True)

    print("-" * 60)
    print("Generando artefactos de datos...")
    print("-" * 60)

    generate_backtest(df_annual, paths["models_dir"], paths["data_dir"])
    generate_oof(df_annual, paths["data_dir"])
    generate_monthly_history(df_monthly, paths["models_dir"], paths["data_dir"])
    generate_correlations(df_annual, paths["data_dir"])

    elapsed = time.time() - t0
    print(f"\nPipeline completo en {elapsed:.1f}s")

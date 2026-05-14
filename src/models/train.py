"""Train the 3 production models and save artifacts to models/ directory."""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor

# Allow running as script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.definitions import (
    FEATURES_MAGNITUDE,
    FEATURES_SET_A,
    TRAIN_END_YEAR,
    TEST_START_YEAR,
    TRIGGER_THRESHOLD,
    DETECTOR_THRESHOLD,
    add_es_risaralda,
    build_set_a_interactions,
    build_monthly_lags,
    get_monthly_feature_cols,
)


def _load_env_paths() -> tuple[Path, Path, Path]:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    import os
    repo_root = Path(__file__).resolve().parents[2]
    data_annual = Path(
        os.getenv("DATA_ANNUAL",
                  str(repo_root / "insumos/data/dataset_modelado_anual_limpio.csv"))
    )
    data_monthly = Path(
        os.getenv("DATA_MONTHLY",
                  str(repo_root / "insumos/data/dataset_operativo_mensual_limpio.csv"))
    )
    models_dir = Path(os.getenv("MODELS_DIR", str(repo_root / "insumos/models")))
    return data_annual, data_monthly, models_dir


def _recall_at_threshold(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    actual_pos = y_true <= threshold
    if actual_pos.sum() == 0:
        return float("nan")
    predicted_pos = y_pred <= threshold
    return float((actual_pos & predicted_pos).sum() / actual_pos.sum())


def train_magnitude_model(df_annual: pd.DataFrame) -> tuple:
    """Train XGBoost magnitude model. Returns (model, test_mae, test_r2)."""
    train = df_annual[df_annual["anio"] <= TRAIN_END_YEAR].copy()
    test = df_annual[df_annual["anio"] >= TEST_START_YEAR].copy()

    X_train = train[FEATURES_MAGNITUDE]
    y_train = train["perdida_rendimiento_anual_pct"]
    X_test = test[FEATURES_MAGNITUDE]
    y_test = test["perdida_rendimiento_anual_pct"]

    model = XGBRegressor(
        max_depth=3,
        learning_rate=0.03,
        n_estimators=400,
        reg_lambda=1.0,
        subsample=1.0,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, float(mae), float(r2)


def train_detector_trigger_model(df_annual: pd.DataFrame) -> tuple:
    """Train HGB Set A model. Returns (model, test_mae, recall_detector, recall_trigger, basis_risk)."""
    train = df_annual[df_annual["anio"] <= TRAIN_END_YEAR].copy()
    test = df_annual[df_annual["anio"] >= TEST_START_YEAR].copy()

    X_train = train[FEATURES_SET_A]
    y_train = train["perdida_rendimiento_anual_pct"]
    X_test = test[FEATURES_SET_A]
    y_test = test["perdida_rendimiento_anual_pct"].values

    model = HistGradientBoostingRegressor(
        max_depth=2,
        learning_rate=0.05,
        max_iter=200,
        min_samples_leaf=1,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    recall_det = _recall_at_threshold(y_test, y_pred, DETECTOR_THRESHOLD)
    recall_trig = _recall_at_threshold(y_test, y_pred, TRIGGER_THRESHOLD)

    # Basis risk: MAE only on rows where trigger fired or actual event occurred
    mask = (y_test <= TRIGGER_THRESHOLD) | (y_pred <= TRIGGER_THRESHOLD)
    basis_risk = float(mean_absolute_error(y_test[mask], y_pred[mask])) if mask.sum() > 0 else float("nan")

    return model, float(mae), float(recall_det), float(recall_trig), float(basis_risk)


def train_monthly_model(df_monthly: pd.DataFrame, df_annual: pd.DataFrame) -> tuple:
    """Train HGB monthly model. Returns (model, feature_cols, test_mae_annualized, test_r2)."""
    # Merge annual target into monthly data
    annual_target = df_annual[["departamento", "anio", "perdida_rendimiento_anual_pct"]].drop_duplicates()
    df = df_monthly.merge(annual_target, on=["departamento", "anio"], how="left")

    df = df[df["anio"].between(2007, 2024)].copy()
    df = df[df["es_mes_cosecha"] == 1].copy()
    df = df[df["perdida_rendimiento_anual_pct"].notna()].copy()

    feature_cols = get_monthly_feature_cols(df)

    train = df[df["anio"] <= TRAIN_END_YEAR].copy()
    test = df[df["anio"] >= TEST_START_YEAR].copy()

    X_train = train[feature_cols]
    y_train = train["perdida_rendimiento_anual_pct"]
    X_test = test[feature_cols]
    y_test = test["perdida_rendimiento_anual_pct"]

    model = HistGradientBoostingRegressor(
        max_depth=2,
        learning_rate=0.03,
        max_iter=200,
        min_samples_leaf=2,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Annualized MAE via weighted mean by factor_mensual
    test_pred = test.copy()
    test_pred["score"] = model.predict(X_test)
    grouped = test_pred.groupby(["departamento", "anio"])
    ann_pred, ann_true = [], []
    for _, grp in grouped:
        w = grp["factor_mensual"].values
        if w.sum() == 0:
            continue
        ann_pred.append(np.average(grp["score"].values, weights=w))
        ann_true.append(np.average(grp["perdida_rendimiento_anual_pct"].values, weights=w))

    if len(ann_pred) == 0:
        mae_ann, r2_ann = float("nan"), float("nan")
    else:
        mae_ann = float(mean_absolute_error(ann_true, ann_pred))
        r2_ann = float(r2_score(ann_true, ann_pred))

    return model, feature_cols, mae_ann, r2_ann


def run_training(verbose: bool = True) -> dict:
    t0 = time.time()
    data_annual, data_monthly, models_dir = _load_env_paths()
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── Annual dataset ────────────────────────────────────────────────────
    df_annual = pd.read_csv(data_annual, sep=";")
    df_annual = add_es_risaralda(df_annual)
    df_annual = build_set_a_interactions(df_annual)
    df_annual = df_annual[df_annual["perdida_rendimiento_anual_pct"].notna()].copy()

    # ── Model 1: XGBoost magnitude ────────────────────────────────────────
    if verbose:
        print("[1/3] Modelo magnitud (XGBoost / baseline_parsimonioso)")
    mag_model, mag_mae, mag_r2 = train_magnitude_model(df_annual)
    joblib.dump(mag_model, models_dir / "magnitude_xgb.pkl")
    if verbose:
        print(f"      Test MAE: {mag_mae:.2f} pp | R²: {mag_r2:.3f}")

    # ── Model 2: HGB detector+trigger ────────────────────────────────────
    if verbose:
        print("[2/3] Modelo detector+trigger (HGB / Set A)")
    det_model, det_mae, det_recall_det, det_recall_trig, det_br = train_detector_trigger_model(df_annual)
    joblib.dump(det_model, models_dir / "detector_trigger_hgb.pkl")
    if verbose:
        print(f"      Test MAE: {det_mae:.2f} pp | "
              f"Recall@{DETECTOR_THRESHOLD}%: {det_recall_det:.2f} | "
              f"Recall@{TRIGGER_THRESHOLD}%: {det_recall_trig:.2f} | "
              f"BR: {det_br:.2f} pp")

    # ── Monthly dataset ───────────────────────────────────────────────────
    df_monthly = pd.read_csv(data_monthly, sep=";")
    df_monthly = add_es_risaralda(df_monthly)
    df_monthly = build_monthly_lags(df_monthly)

    # ── Model 3: HGB monthly ──────────────────────────────────────────────
    if verbose:
        print("[3/3] Modelo mensual (HGB / mensual_core_lags, solo cosecha)")
    mon_model, mon_features, mon_mae, mon_r2 = train_monthly_model(df_monthly, df_annual)
    joblib.dump(mon_model, models_dir / "monthly_hgb.pkl")
    joblib.dump(mon_features, models_dir / "monthly_features.pkl")
    if verbose:
        print(f"      MAE anualizado: {mon_mae:.2f} pp | R²: {mon_r2:.3f}")

    elapsed = time.time() - t0

    metadata = {
        "trained_at": pd.Timestamp.now().isoformat(),
        "train_end_year": TRAIN_END_YEAR,
        "test_start_year": TEST_START_YEAR,
        "models": {
            "magnitude_xgb": {
                "algorithm": "XGBoostRegressor",
                "feature_set": "baseline_parsimonioso",
                "n_features": len(FEATURES_MAGNITUDE),
                "test_mae_pp": mag_mae,
                "test_r2": mag_r2,
            },
            "detector_trigger_hgb": {
                "algorithm": "HistGradientBoostingRegressor",
                "feature_set": "set_A_interacc",
                "n_features": len(FEATURES_SET_A),
                "test_mae_pp": det_mae,
                "recall_detector": det_recall_det,
                "recall_trigger": det_recall_trig,
                "basis_risk_pp": det_br,
                "detector_threshold_pct": DETECTOR_THRESHOLD,
                "trigger_threshold_pct": TRIGGER_THRESHOLD,
            },
            "monthly_hgb": {
                "algorithm": "HistGradientBoostingRegressor",
                "feature_set": "mensual_core_lags",
                "n_features": len(mon_features),
                "test_mae_annualized_pp": mon_mae,
                "test_r2": mon_r2,
            },
        },
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(models_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"Artifacts guardados en: {models_dir.resolve()}")
        print(f"Tiempo total: {elapsed:.1f} s")

    # ── Validation checks ─────────────────────────────────────────────────
    assert mag_mae <= 12, f"MAE magnitud {mag_mae:.2f} pp supera límite de 12 pp"
    assert mon_mae <= 14 or np.isnan(mon_mae), \
        f"MAE mensual anualizado {mon_mae:.2f} pp supera límite de 14 pp"
    for fname in ("magnitude_xgb.pkl", "detector_trigger_hgb.pkl",
                  "monthly_hgb.pkl", "monthly_features.pkl"):
        assert (models_dir / fname).exists(), f"Artifact faltante: {fname}"
        joblib.load(models_dir / fname)  # verify loadable

    return metadata


if __name__ == "__main__":
    run_training()

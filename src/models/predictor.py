"""CafeteroPredictor: load trained artifacts and serve predictions."""
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.features.definitions import (
    FEATURES_MAGNITUDE,
    FEATURES_SET_A,
    DETECTOR_THRESHOLD,
    TRIGGER_THRESHOLD,
    CORE_DYNAMIC_MONTHLY,
    add_es_risaralda,
    build_set_a_interactions,
    build_monthly_lags,
    get_monthly_feature_cols,
)


class CafeteroPredictor:
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self._mag_model = None
        self._det_model = None
        self._mon_model = None
        self._mon_features: Optional[list] = None
        self._metadata: dict = {}

    def load(self) -> None:
        self._mag_model = joblib.load(self.models_dir / "magnitude_xgb.pkl")
        self._det_model = joblib.load(self.models_dir / "detector_trigger_hgb.pkl")
        self._mon_model = joblib.load(self.models_dir / "monthly_hgb.pkl")
        self._mon_features = joblib.load(self.models_dir / "monthly_features.pkl")
        meta_path = self.models_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                self._metadata = json.load(f)

    def is_ready(self) -> bool:
        return all(m is not None for m in (
            self._mag_model, self._det_model, self._mon_model, self._mon_features
        ))

    def get_metadata(self) -> dict:
        return self._metadata

    def predict_annual(self, data: dict) -> dict:
        row = {**data}
        row["es_risaralda"] = int(row["departamento"] == "Risaralda")

        df = pd.DataFrame([row])
        df = build_set_a_interactions(df)

        # Model 1 — magnitude
        x_mag = df[FEATURES_MAGNITUDE]
        perdida_estimada = float(self._mag_model.predict(x_mag)[0])

        # Model 2 — detector + trigger
        x_det = df[FEATURES_SET_A]
        score_det = float(self._det_model.predict(x_det)[0])
        evento_detectado = score_det <= DETECTOR_THRESHOLD
        trigger_activado = score_det <= TRIGGER_THRESHOLD

        # Alert level based on magnitude estimate
        if perdida_estimada >= -5.0:
            nivel_alerta = "NORMAL"
        elif perdida_estimada >= TRIGGER_THRESHOLD:
            nivel_alerta = "PRECAUCIÓN"
        else:
            nivel_alerta = "ALERTA"

        basis_risk = abs(perdida_estimada - TRIGGER_THRESHOLD)

        meta_models = self._metadata.get("models", {})
        return {
            "departamento": data["departamento"],
            "anio": data.get("anio"),
            "perdida_estimada_pct": round(perdida_estimada, 4),
            "evento_detectado": evento_detectado,
            "trigger_activado": trigger_activado,
            "nivel_alerta": nivel_alerta,
            "basis_risk_estimado_pp": round(basis_risk, 4),
            "modelo_magnitud": meta_models.get("magnitude_xgb", {}).get("algorithm", "XGBoostRegressor"),
            "modelo_detector_trigger": meta_models.get("detector_trigger_hgb", {}).get("algorithm", "HistGradientBoostingRegressor"),
            "umbral_detector_pct": DETECTOR_THRESHOLD,
            "umbral_trigger_pct": TRIGGER_THRESHOLD,
        }

    def predict_monthly(self, records: list[dict]) -> dict:
        df = pd.DataFrame(records)

        # Rename 'def_' alias back to 'def' if present (Pydantic alias)
        if "def_" in df.columns and "def" not in df.columns:
            df = df.rename(columns={"def_": "def"})

        df = add_es_risaralda(df)
        df = build_monthly_lags(df)

        feature_cols = [c for c in self._mon_features if c in df.columns]
        harvest_df = df[df["es_mes_cosecha"] == 1].copy()

        scores_por_mes = []
        for _, row in df.iterrows():
            entry = {"mes": int(row["mes"]), "anio": int(row["anio"]), "es_cosecha": int(row["es_mes_cosecha"])}
            if row["es_mes_cosecha"] == 1 and all(c in row.index for c in feature_cols):
                x = pd.DataFrame([row[feature_cols]])
                entry["score"] = round(float(self._mon_model.predict(x)[0]), 4)
            else:
                entry["score"] = None
            scores_por_mes.append(entry)

        # Annualized score via weighted mean
        harvest_scores = [e for e in scores_por_mes if e["score"] is not None]
        if harvest_scores:
            harvest_rows = harvest_df.copy()
            harvest_rows["score"] = self._mon_model.predict(harvest_rows[feature_cols])
            weights = harvest_rows["factor_mensual"].values
            score_ann = float(np.average(harvest_rows["score"].values, weights=weights)) if weights.sum() > 0 else float("nan")
        else:
            score_ann = float("nan")

        alerta_activa = (not np.isnan(score_ann)) and (score_ann <= DETECTOR_THRESHOLD)
        if np.isnan(score_ann):
            nivel_alerta = "SIN_DATOS"
        elif score_ann >= -5.0:
            nivel_alerta = "NORMAL"
        elif score_ann >= TRIGGER_THRESHOLD:
            nivel_alerta = "PRECAUCIÓN"
        else:
            nivel_alerta = "ALERTA"

        departamento = records[0].get("departamento", "") if records else ""
        meta_models = self._metadata.get("models", {})
        return {
            "departamento": departamento,
            "score_anualizado": round(score_ann, 4) if not np.isnan(score_ann) else None,
            "alerta_activa": alerta_activa,
            "nivel_alerta": nivel_alerta,
            "scores_por_mes": scores_por_mes,
            "modelo_mensual": meta_models.get("monthly_hgb", {}).get("algorithm", "HistGradientBoostingRegressor"),
        }

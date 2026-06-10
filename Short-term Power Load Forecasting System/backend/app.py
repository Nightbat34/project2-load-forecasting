#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Backend API for the Short-term Power Load Forecasting System.

This service is intentionally small and portable. It exposes the final static
forecast and a model prediction endpoint for already-engineered feature rows.
Raw weather-only forecasting requires the lag/rolling features described in
feature_engineering_reference.py.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

MODEL_FILES = {
    "load_mean": "load_mean_Stacking.joblib",
    "load_max": "load_max_Stacking.joblib",
    "load_min": "load_min_Stacking.joblib",
}

TARGET_NAMES = {
    "load_mean": "日平均负荷",
    "load_max": "日最高负荷",
    "load_min": "日最低负荷",
}


def create_app() -> Flask:
    app = Flask(__name__)
    bundles = {
        target: joblib.load(MODELS_DIR / filename)
        for target, filename in MODEL_FILES.items()
    }

    @app.get("/api/health")
    def health():
        return jsonify({
            "status": "ok",
            "system": "电力系统短期负荷预测系统",
            "targets": TARGET_NAMES,
        })

    @app.get("/api/final-forecast")
    def final_forecast():
        csv_path = MODELS_DIR / "project2_final_prediction_2015_01_11_17.csv"
        df = pd.read_csv(csv_path)
        return jsonify(df.to_dict(orient="records"))

    @app.get("/api/model-manifest")
    def model_manifest():
        csv_path = MODELS_DIR / "project2_model_manifest.csv"
        df = pd.read_csv(csv_path)
        return jsonify(df.to_dict(orient="records"))

    @app.post("/api/predict")
    def predict():
        """Predict one target from pre-engineered feature rows.

        Request JSON:
        {
          "target": "load_mean",
          "rows": [{"temp_max": ..., "load_mean_lag_1": ...}]
        }

        The row must contain the exact feature names listed in the corresponding
        model bundle. This avoids silently producing incorrect predictions from
        incomplete raw weather inputs.
        """
        payload = request.get_json(force=True, silent=False)
        target = payload.get("target")
        rows = payload.get("rows")
        if target not in bundles:
            return jsonify({"error": f"target must be one of {list(bundles)}"}), 400
        if not isinstance(rows, list) or not rows:
            return jsonify({"error": "rows must be a non-empty list of feature objects"}), 400

        bundle = bundles[target]
        features = bundle["features"]
        missing = sorted(set(features) - set(rows[0].keys()))
        if missing:
            return jsonify({
                "error": "missing required engineered features",
                "missing": missing,
                "required_features": features,
            }), 400

        X = pd.DataFrame(rows)[features]
        pred = bundle["model"].predict(X)
        return jsonify({
            "target": target,
            "target_name": TARGET_NAMES[target],
            "unit": "MW",
            "prediction": [round(float(value), 2) for value in pred],
        })

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

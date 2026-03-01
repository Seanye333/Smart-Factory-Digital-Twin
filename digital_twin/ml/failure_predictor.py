"""
Predictive Maintenance — Failure Probability Predictor
=========================================================
Uses an XGBoost classifier trained on synthetic sensor history to predict
the probability that a machine will fail within the next N cycles.

Training data is generated automatically from the MockPLCGenerator so the
model works out-of-the-box without any real labelled dataset.

Features used (8 total):
  temperature, vibration, current, pressure, speed, cycle_time,
  temp_delta (rate of change), vib_delta

Output:
  failure_probability  ∈ [0, 1]
  risk_level           → "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
"""

import logging
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Optional XGBoost — fall back to logistic regression if unavailable
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("[FailurePredictor] xgboost not installed — using sklearn LogisticRegression.")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

MODEL_PATH     = Path(__file__).parent / "models" / "failure_predictor.pkl"
RISK_THRESHOLDS = {
    "LOW":      (0.00, 0.35),
    "MEDIUM":   (0.35, 0.60),
    "HIGH":     (0.60, 0.80),
    "CRITICAL": (0.80, 1.01),
}

FEATURE_NAMES = [
    "temperature", "vibration", "current", "pressure",
    "speed", "cycle_time", "temp_delta", "vib_delta",
]


# --------------------------------------------------------------------------
# Synthetic training data generator
# --------------------------------------------------------------------------

def _generate_training_data(n_samples: int = 8_000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create labelled synthetic sensor data:
      - Healthy machines: low sensor values, label=0
      - Pre-failure / degraded: elevated values, label=1 (~25 % of data)
    """
    rng = np.random.default_rng(42)

    # Healthy samples (~75 %)
    n_healthy = int(n_samples * 0.75)
    X_healthy = np.column_stack([
        rng.normal(45,  3,   n_healthy),   # temperature
        rng.normal(2.1, 0.3, n_healthy),   # vibration
        rng.normal(18,  2,   n_healthy),   # current
        rng.normal(6.0, 0.4, n_healthy),   # pressure
        rng.normal(3000, 100, n_healthy),  # speed
        rng.normal(45,  4,   n_healthy),   # cycle_time
        rng.normal(0,   0.5, n_healthy),   # temp_delta
        rng.normal(0,   0.1, n_healthy),   # vib_delta
    ])

    # Pre-failure samples (~25 %)
    n_fail = n_samples - n_healthy
    X_fail = np.column_stack([
        rng.normal(72,  8,   n_fail),      # elevated temperature
        rng.normal(6.5, 1.5, n_fail),      # high vibration
        rng.normal(32,  5,   n_fail),      # overcurrent
        rng.normal(3.5, 1.0, n_fail),      # low pressure
        rng.normal(2400, 300, n_fail),     # speed deviation
        rng.normal(78,  12,  n_fail),      # long cycle
        rng.normal(3.5, 1.0, n_fail),      # rising temp
        rng.normal(1.2, 0.5, n_fail),      # rising vibration
    ])

    X = np.vstack([X_healthy, X_fail])
    y = np.hstack([
        np.zeros(n_healthy, dtype=int),
        np.ones(n_fail,     dtype=int),
    ])

    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# --------------------------------------------------------------------------
# Predictor
# --------------------------------------------------------------------------

@dataclass
class PredictionResult:
    machine_id:          str
    failure_probability: float
    risk_level:          str
    feature_importances: Dict[str, float]
    timestamp:           float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> Dict:
        return {
            "machine_id":          self.machine_id,
            "failure_probability": round(self.failure_probability, 4),
            "failure_pct":         round(self.failure_probability * 100, 1),
            "risk_level":          self.risk_level,
            "timestamp":           self.timestamp,
        }


class FailurePredictor:
    """
    Machine failure probability predictor.

    Automatically trains a model on synthetic data if no saved model exists.
    Supports per-machine recent-history buffers for delta feature calculation.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path    = model_path or MODEL_PATH
        self._model        = None
        self._history:     Dict[str, List[Dict]] = {}   # machine_id → last N readings
        self._history_len: int = 5                       # readings kept per machine
        self._load_or_train()

    # ------------------------------------------------------------------
    # Model initialisation
    # ------------------------------------------------------------------

    def _load_or_train(self) -> None:
        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    self._model = pickle.load(f)
                logger.info(f"[FailurePredictor] Loaded model from {self.model_path}")
                return
            except Exception as exc:
                logger.warning(f"[FailurePredictor] Could not load model: {exc} — retraining")

        self._train()

    def _train(self) -> None:
        logger.info("[FailurePredictor] Generating training data & fitting model …")
        X, y = _generate_training_data()

        if XGB_AVAILABLE:
            estimator = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="logloss",
                random_state=42,
                use_label_encoder=False,
                verbosity=0,
            )
        else:
            estimator = LogisticRegression(max_iter=1000, C=1.0)

        self._model = Pipeline([
            ("scaler",    StandardScaler()),
            ("estimator", estimator),
        ])
        self._model.fit(X, y)

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self._model, f)
        logger.info(f"[FailurePredictor] Model saved → {self.model_path}")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _extract_features(self, reading: Dict, machine_id: str) -> np.ndarray:
        history = self._history.get(machine_id, [])

        temp_delta = 0.0
        vib_delta  = 0.0
        if history:
            prev = history[-1]
            temp_delta = reading.get("temperature", 0) - prev.get("temperature", 0)
            vib_delta  = reading.get("vibration",   0) - prev.get("vibration",   0)

        return np.array([[
            reading.get("temperature", 45),
            reading.get("vibration",   2.1),
            reading.get("current",     18),
            reading.get("pressure",    6.0),
            reading.get("speed",       3000),
            reading.get("cycle_time",  45),
            temp_delta,
            vib_delta,
        ]])

    def predict(self, reading: Dict, machine_id: str = "UNKNOWN") -> PredictionResult:
        """Return PredictionResult for a single sensor reading dict."""
        X = self._extract_features(reading, machine_id)

        prob = float(self._model.predict_proba(X)[0][1])

        # Clamp to sensible range — faulted machines should show ~1.0
        if reading.get("fault_code", 0) > 0:
            prob = max(prob, 0.92)
        if reading.get("profile") == "faulted":
            prob = max(prob, 0.95)
        elif reading.get("profile") == "critical":
            prob = max(prob, 0.70)

        prob = min(1.0, prob)

        risk = next(
            k for k, (lo, hi) in RISK_THRESHOLDS.items() if lo <= prob < hi
        )

        # Update history
        buf = self._history.setdefault(machine_id, [])
        buf.append(dict(reading))
        if len(buf) > self._history_len:
            buf.pop(0)

        # Feature importance (only available for XGBoost pipeline)
        importances: Dict[str, float] = {}
        try:
            est = self._model.named_steps["estimator"]
            if hasattr(est, "feature_importances_"):
                importances = dict(zip(FEATURE_NAMES, est.feature_importances_.tolist()))
        except Exception:
            pass

        return PredictionResult(
            machine_id=machine_id,
            failure_probability=round(prob, 4),
            risk_level=risk,
            feature_importances=importances,
        )

    def predict_failure_probability(self, reading: Dict, machine_id: str = "") -> float:
        """Convenience — returns just the probability float."""
        mid = machine_id or reading.get("machine_id", "UNKNOWN")
        return self.predict(reading, mid).failure_probability

    def predict_all(self, readings: Dict[str, Dict]) -> Dict[str, PredictionResult]:
        """Predict for every machine in a readings snapshot."""
        return {mid: self.predict(r, mid) for mid, r in readings.items()}

    def retrain(self) -> None:
        """Force a full retrain (e.g. after new labelled data arrives)."""
        self._train()

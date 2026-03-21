"""
models/hmm_model.py
Hidden Markov Model for market-regime detection.
States are automatically labelled: 0=Bear, 1=Sideways, 2=Bull
by sorting on mean log-return of each Gaussian component.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Regime labels
BEAR     = 0
SIDEWAYS = 1
BULL     = 2
REGIME_NAMES = {BEAR: "BEAR", SIDEWAYS: "SIDEWAYS", BULL: "BULL"}


class RegimeHMM:
    """
    Gaussian HMM with full covariance for market-regime detection.

    Features fed to the HMM
    ------------------------
    - Log return of close price
    - 20-bar rolling volatility of log returns
    - Log-volume change (optional)
    """

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 300,
        covariance_type: str = "full",
        random_state: int = 42,
    ) -> None:
        self.n_states = n_states
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state

        self._model: hmm.GaussianHMM | None = None
        self._scaler = StandardScaler()
        # Maps raw HMM component index → semantic label (BEAR/SIDEWAYS/BULL)
        self._state_map: dict[int, int] = {}
        self.is_fitted = False

    # ── Feature engineering ──────────────────────────────────────────────────

    def _build_features(
        self,
        prices: np.ndarray,
        volumes: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Build feature matrix from price (and optionally volume) array.
        Returns array of shape (N-1, n_features).
        """
        log_ret = np.diff(np.log(prices + 1e-12)).astype(np.float64)
        s = pd.Series(log_ret)
        rolling_vol = s.rolling(20, min_periods=3).std().fillna(s.std()).values

        cols = [log_ret, rolling_vol]

        if volumes is not None and len(volumes) == len(prices):
            log_vol_chg = np.diff(np.log(volumes + 1e-12))
            cols.append(log_vol_chg)

        return np.column_stack(cols)

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        prices: np.ndarray,
        volumes: np.ndarray | None = None,
    ) -> "RegimeHMM":
        """
        Fit the HMM to price history.

        Parameters
        ----------
        prices  : 1-D close-price array, length N
        volumes : optional volume array, length N
        """
        features = self._build_features(prices, volumes)
        features_sc = self._scaler.fit_transform(features)

        self._model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False,
        )
        self._model.fit(features_sc)

        # Label states by ascending mean log-return
        means = self._model.means_[:, 0]          # first feature = log return
        order = np.argsort(means)                  # lowest → highest
        for rank, raw in enumerate(order):
            if rank == 0:
                self._state_map[raw] = BEAR
            elif rank == self.n_states - 1:
                self._state_map[raw] = BULL
            else:
                self._state_map[raw] = SIDEWAYS

        self.is_fitted = True

        logger.info(
            "HMM fitted | score=%.2f | state means (log-ret): %s",
            self._model.score(features_sc),
            {REGIME_NAMES[self._state_map[r]]: f"{means[r]:.6f}" for r in range(self.n_states)},
        )
        logger.info("Transition matrix:\n%s", np.round(self._model.transmat_, 4))
        return self

    # ── Predict ──────────────────────────────────────────────────────────────

    def _raw_predict(self, prices: np.ndarray, volumes: np.ndarray | None = None):
        if not self.is_fitted:
            raise RuntimeError("RegimeHMM: call fit() first.")
        features = self._build_features(prices, volumes)
        features_sc = self._scaler.transform(features)
        return features_sc, self._model.predict(features_sc)

    def predict(
        self,
        prices: np.ndarray,
        volumes: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return labelled regime sequence, length N-1.
        0=Bear, 1=Sideways, 2=Bull.
        """
        _, raw = self._raw_predict(prices, volumes)
        return np.array([self._state_map[s] for s in raw], dtype=np.int32)

    def predict_proba(
        self,
        prices: np.ndarray,
        volumes: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return state probabilities, shape (N-1, n_states).
        Columns ordered [Bear, Sideways, Bull].
        """
        features_sc, _ = self._raw_predict(prices, volumes)
        raw_proba = self._model.predict_proba(features_sc)
        ordered = np.zeros_like(raw_proba)
        for raw_idx, label in self._state_map.items():
            ordered[:, label] += raw_proba[:, raw_idx]
        return ordered

    def predict_current_regime(
        self,
        recent_prices: np.ndarray,
        recent_volumes: np.ndarray | None = None,
    ) -> tuple[int, np.ndarray]:
        """
        Predict regime of the *most recent* bar.

        Returns
        -------
        regime : int (BEAR=0, SIDEWAYS=1, BULL=2)
        proba  : np.ndarray shape (n_states,)
        """
        proba = self.predict_proba(recent_prices, recent_volumes)
        last_proba = proba[-1]
        regime = int(np.argmax(last_proba))
        return regime, last_proba

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def get_regime_name(self, regime: int) -> str:
        return REGIME_NAMES.get(regime, "UNKNOWN")

    def get_transition_matrix(self) -> np.ndarray:
        """Transition matrix reordered as [Bear, Sideways, Bull]."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        n = self.n_states
        T = np.zeros((n, n))
        raw_T = self._model.transmat_
        for r_from, l_from in self._state_map.items():
            for r_to, l_to in self._state_map.items():
                T[l_from, l_to] = raw_T[r_from, r_to]
        return T

    def log_likelihood(self, prices: np.ndarray, volumes: np.ndarray | None = None) -> float:
        features = self._build_features(prices, volumes)
        features_sc = self._scaler.transform(features)
        return self._model.score(features_sc)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self._model,
                    "scaler": self._scaler,
                    "state_map": self._state_map,
                    "n_states": self.n_states,
                    "n_iter": self.n_iter,
                    "covariance_type": self.covariance_type,
                },
                f,
            )
        logger.info("HMM saved → %s", path)

    @classmethod
    def load(cls, path: Path) -> "RegimeHMM":
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(
            n_states=data["n_states"],
            n_iter=data["n_iter"],
            covariance_type=data["covariance_type"],
        )
        obj._model    = data["model"]
        obj._scaler   = data["scaler"]
        obj._state_map = data["state_map"]
        obj.is_fitted = True
        logger.info("HMM loaded ← %s", path)
        return obj
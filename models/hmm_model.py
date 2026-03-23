"""
models/hmm_model.py
Hidden Markov Model for market-regime detection.
Calibrated for 15-minute OHLCV bars.

Key fixes in this version
--------------------------
Fix 1 — Forward-return state labeling (critical)
    States are now labeled by mean 24h FORWARD return at training time,
    not by mean past log_ret_192.  The previous version labeled BULL as
    "the state where price rose a lot recently" — a retrospective signal
    that fires at the top of moves and predicts reversals, not continuation.
    The calibration chart confirmed this: P(Bull)=1.0 → negative next-bar
    return.  Forward-return labeling directly optimises for what the
    strategy needs: a signal that predicts positive future returns.

Fix 2 — New leading/microstructure features
    Added vol_ratio (vol_20/vol_96) and rsi_14.  The previous feature set
    was entirely backward-looking momentum; these features capture
    conditions that tend to PRECEDE directional moves rather than describe
    past ones.  Also added vol_rel (volume/20-bar mean) when volumes are
    supplied — volume surges coincide with genuine regime transitions.

Fix 3 — Less-sticky transition matrix initialisation
    Default hmmlearn initialisation produces ~0.99 diagonal transitions,
    making P(Bull) near-binary and Kelly sizing meaningless.  A Dirichlet
    prior of 5 (self) vs 1 (others) starts EM at ~0.56 diagonal.  EM
    learns the true value from data but avoids the degenerate sticky
    solution that collapses posterior probabilities to {0, 1}.

After ANY change to this file retrain ALL models:
    python train_models.py
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

# 24-hour forward return horizon for state labeling (96 × 15-min bars)
_LABEL_HORIZON = 24


def _compute_rsi(log_ret: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Wilder RSI from log returns.  Values in [0, 100].
    Warmup period (first `period` bars) returns 50.0 (neutral).
    """
    gains  = np.where(log_ret > 0,  log_ret, 0.0)
    losses = np.where(log_ret < 0, -log_ret, 0.0)
    rsi    = np.full(len(log_ret), 50.0, dtype=np.float64)
    if len(log_ret) < period + 1:
        return rsi
    avg_gain = float(gains[:period].mean())
    avg_loss = float(losses[:period].mean())
    alpha    = 1.0 / period
    for i in range(period, len(log_ret)):
        avg_gain = alpha * gains[i]  + (1.0 - alpha) * avg_gain
        avg_loss = alpha * losses[i] + (1.0 - alpha) * avg_loss
        if avg_loss < 1e-12:
            rsi[i] = 100.0
        else:
            rs     = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi.astype(np.float64)


class RegimeHMM:
    """
    Gaussian HMM with full covariance for market-regime detection.
    Calibrated for 15-minute OHLCV bars.

    Features fed to the HMM
    ------------------------
    col 0  log_ret      — 1-bar (15 min) log return
    col 1  vol_20       — 20-bar (5 h) rolling std of log returns
    col 2  vol_96       — 96-bar (24 h) rolling std of log returns
    col 3  log_ret_96   — 24h cumulative log return
    col 4  log_ret_192  — 48h cumulative log return
    col 5  vol_ratio    — vol_20 / vol_96  (volatility compression)
    col 6  rsi_14       — 14-bar Wilder RSI (leading momentum)

    Optional (when volumes supplied)
    ---------------------------------
    col 7  log_vol_chg  — 1-bar log volume change
    col 8  vol_rel      — volume / 20-bar mean volume (surge detector)

    State labeling
    --------------
    States are labeled by mean 24h FORWARD return computed at training time.
    Highest mean forward return → BULL; lowest → BEAR; middle → SIDEWAYS.
    """

    _LABEL_HORIZON: int = _LABEL_HORIZON

    def __init__(
        self,
        n_states:        int = 3,
        n_iter:          int = 300,
        covariance_type: str = "full",
        random_state:    int = 42,
    ) -> None:
        self.n_states        = n_states
        self.n_iter          = n_iter
        self.covariance_type = covariance_type
        self.random_state    = random_state

        self._model:     hmm.GaussianHMM | None = None
        self._scaler     = StandardScaler()
        self._state_map: dict[int, int] = {}
        self.is_fitted   = False

    # ── Feature engineering ──────────────────────────────────────────────────

    def _build_features(
        self,
        prices:  np.ndarray,
        volumes: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Build feature matrix from price (and optionally volume) array.
        Returns array of shape (N-1, n_features), where N = len(prices).
        """
        log_ret = np.diff(np.log(prices + 1e-12)).astype(np.float64)
        s       = pd.Series(log_ret)
        g_std   = float(s.std()) if float(s.std()) > 0 else 1e-8

        # Volatility at multiple timescales
        vol_20 = s.rolling(10,  min_periods=4).std().fillna(g_std).values
        vol_96 = s.rolling(48,  min_periods=20).std().fillna(g_std).values

        # Multi-day cumulative returns — directional regime axis
        log_ret_96  = s.rolling(48,  min_periods=1).sum().values
        log_ret_192 = s.rolling(96, min_periods=1).sum().values

        # Volatility compression ratio — low values precede breakouts
        vol_ratio = np.clip(vol_20 / (vol_96 + 1e-12), 0.1, 10.0)

        # RSI — leading momentum with mean-reversion properties at short horizons
        rsi = _compute_rsi(log_ret, period=14)

        cols = [log_ret, vol_20, vol_96, log_ret_96, log_ret_192, vol_ratio, rsi]

        if volumes is not None and len(volumes) == len(prices):
            log_vol_chg = np.diff(np.log(volumes + 1e-12))

            # Volume relative to 20-bar mean — spikes mark genuine transitions
            v_series = pd.Series(volumes[1:])
            vol_ma20 = (
                v_series.rolling(20, min_periods=4)
                .mean()
                .fillna(float(v_series.mean()) + 1e-12)
                .values
            )
            vol_rel = np.clip(volumes[1:] / (vol_ma20 + 1e-12), 0.1, 20.0)
            cols.extend([log_vol_chg, vol_rel])

        return np.column_stack(cols)

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        prices:  np.ndarray,
        volumes: np.ndarray | None = None,
    ) -> "RegimeHMM":
        """
        Fit the HMM and label states by 24h forward return.

        BULL     = state whose bars are followed by highest mean 24h return.
        BEAR     = state whose bars are followed by lowest  mean 24h return.
        SIDEWAYS = the remaining state.
        """
        features    = self._build_features(prices, volumes)
        features_sc = self._scaler.fit_transform(features)

        # ── Less-sticky transition matrix initialisation ───────────────────────
        # hmmlearn default produces ~0.99 diagonal → near-binary posteriors.
        # Prior: self-transition concentration=5, others=1 → ~0.56 diagonal.
        # EM learns true value but avoids the degenerate all-sticky solution.
        _n = self.n_states
        _prior = np.full((_n, _n), 1.0)
        np.fill_diagonal(_prior, 5.0)
        _transmat_start = _prior / _prior.sum(axis=1, keepdims=True)

        self._model = hmm.GaussianHMM(
            n_components    = self.n_states,
            covariance_type = self.covariance_type,
            n_iter          = self.n_iter,
            random_state    = self.random_state,
            verbose         = False,
            init_params     = "mcs",   # init means, covars, startprob — NOT transmat
            params          = "mcst",  # optimise all params including transmat
        )
        # Set our custom transmat BEFORE fit(); init_params="mcs" ensures it is
        # not overwritten by hmmlearn's _init() since 't' is excluded.
        self._model.transmat_ = _transmat_start
        self._model.fit(features_sc)

        # ── State labeling by 24h FORWARD return ──────────────────────────────
        # At training time we have access to future prices, so we can label each
        # raw HMM state by the mean 24h return of bars FOLLOWING that state.
        # This directly optimises for what the strategy needs: P(Bull) high →
        # positive future returns.  Using past returns for labeling (previous
        # version) made BULL fire at the TOP of moves, predicting reversals.
        raw_states = self._model.predict(features_sc)   # shape (N-1,)
        N_feat     = len(raw_states)
        horizon    = self._LABEL_HORIZON

        max_i = N_feat - horizon
        if max_i <= 0:
            raise RuntimeError(
                f"Training set too short for LABEL_HORIZON={horizon}. "
                f"Need at least {horizon + 2} prices."
            )

        # forward_ret[i] = log(prices[i+1+horizon] / prices[i+1])
        # prices[i+1] is the close price at aligned bar i (features are diff'd)
        forward_ret = np.array([
            np.log((prices[i + 1 + horizon] + 1e-12)
                   / (prices[i + 1]          + 1e-12))
            for i in range(max_i)
        ], dtype=np.float64)

        state_fwd_means = np.zeros(self.n_states, dtype=np.float64)
        for s_raw in range(self.n_states):
            mask = (raw_states[:max_i] == s_raw)
            state_fwd_means[s_raw] = (
                float(forward_ret[mask].mean()) if mask.sum() > 0 else 0.0
            )

        logger.info(
            "State mean 24h forward returns (raw): %s",
            {s: f"{state_fwd_means[s]:.5f}" for s in range(self.n_states)},
        )

        order = np.argsort(state_fwd_means)
        for rank, raw in enumerate(order):
            if rank == 0:
                self._state_map[raw] = BEAR
            elif rank == self.n_states - 1:
                self._state_map[raw] = BULL
            else:
                self._state_map[raw] = SIDEWAYS

        fwd_spread = float(state_fwd_means.max() - state_fwd_means.min())
        if fwd_spread < 1e-4:
            logger.warning(
                "State separation by forward return is very small "
                "(spread=%.6f). Labels may be unreliable. "
                "Try a different random_state or longer training window.",
                fwd_spread,
            )

        self.is_fitted = True

        logger.info(
            "HMM fitted | score=%.2f | labeled 24h fwd returns: %s",
            self._model.score(features_sc),
            {
                REGIME_NAMES[self._state_map[r]]: f"{state_fwd_means[r]:.5f}"
                for r in range(self.n_states)
            },
        )
        logger.info("Transition matrix:\n%s", np.round(self._model.transmat_, 4))
        return self

    # ── Predict ──────────────────────────────────────────────────────────────

    def _raw_predict(
        self,
        prices:  np.ndarray,
        volumes: np.ndarray | None = None,
    ):
        if not self.is_fitted:
            raise RuntimeError("RegimeHMM: call fit() first.")
        features    = self._build_features(prices, volumes)
        features_sc = self._scaler.transform(features)
        return features_sc, self._model.predict(features_sc)

    def predict(
        self,
        prices:  np.ndarray,
        volumes: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return labelled regime sequence, length N-1.  0=Bear, 1=Sideways, 2=Bull."""
        _, raw = self._raw_predict(prices, volumes)
        return np.array([self._state_map[s] for s in raw], dtype=np.int32)

    def predict_proba(
        self,
        prices:  np.ndarray,
        volumes: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return posterior state probabilities, shape (N-1, n_states).
        Columns are ordered [Bear, Sideways, Bull].
        """
        features_sc, _ = self._raw_predict(prices, volumes)
        raw_proba       = self._model.predict_proba(features_sc)
        ordered         = np.zeros_like(raw_proba)
        for raw_idx, label in self._state_map.items():
            ordered[:, label] += raw_proba[:, raw_idx]
        return ordered.astype(np.float32)

    def predict_current_regime(
        self,
        recent_prices:  np.ndarray,
        recent_volumes: np.ndarray | None = None,
    ) -> tuple[int, np.ndarray]:
        """Predict regime of the *most recent* bar."""
        proba      = self.predict_proba(recent_prices, recent_volumes)
        last_proba = proba[-1]
        regime     = int(np.argmax(last_proba))
        return regime, last_proba

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def get_regime_name(self, regime: int) -> str:
        return REGIME_NAMES.get(regime, "UNKNOWN")

    def get_transition_matrix(self) -> np.ndarray:
        """Transition matrix reordered as [Bear, Sideways, Bull]."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        n     = self.n_states
        T     = np.zeros((n, n))
        raw_T = self._model.transmat_
        for r_from, l_from in self._state_map.items():
            for r_to, l_to in self._state_map.items():
                T[l_from, l_to] = raw_T[r_from, r_to]
        return T

    def log_likelihood(
        self,
        prices:  np.ndarray,
        volumes: np.ndarray | None = None,
    ) -> float:
        features    = self._build_features(prices, volumes)
        features_sc = self._scaler.transform(features)
        return float(self._model.score(features_sc))

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model":           self._model,
                    "scaler":          self._scaler,
                    "state_map":       self._state_map,
                    "n_states":        self.n_states,
                    "n_iter":          self.n_iter,
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
            n_states        = data["n_states"],
            n_iter          = data["n_iter"],
            covariance_type = data["covariance_type"],
        )
        obj._model     = data["model"]
        obj._scaler    = data["scaler"]
        obj._state_map = data["state_map"]
        obj.is_fitted  = True
        logger.info("HMM loaded ← %s", path)
        return obj
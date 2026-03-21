"""
models/lstm_model.py
Attention-LSTM for predicting P(price goes UP in next N bars).
Technical features are calibrated for **15-minute OHLCV bars** and are
augmented with HMM regime one-hot encodings and an optional daily
Fear & Greed Index value (normalised to [0, 1]).

15-minute bar calibration summary
----------------------------------
Indicator               Bars   Wall-clock
--------------------   ------  ----------
RSI                       14   3.5 h
MACD EMA (short/long)  12/26   3 h / 6.5 h
Bollinger Bands           20   5 h
ATR                       14   3.5 h
Stochastic %%K             14   3.5 h
Volatility  (short)        4   1 h
Volatility  (medium)      16   4 h
Volatility  (long)        96   24 h  (1 day)
EMA trend   (medium)      50   12.5 h
EMA trend   (long)       200   ~2 days
Rate-of-change             4   1 h
Rate-of-change            16   4 h
Rate-of-change            96   24 h  (1 day)
Volume MA                 20   5 h
OBV MA (short / long)   5/20   1.25 h / 5 h
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# ── weights_only safe-globals allowlist ──────────────────────────────────────
# Every numpy type that can appear in a saved checkpoint must be listed here
# so that weights_only=True keeps working after PyTorch 2.6.
#
# Layer 1 – array reconstruction (all numpy versions)
_NUMPY_SAFE_GLOBALS: list = [
    np._core.multiarray._reconstruct,  # ndarray reconstruction
    np.ndarray,                         # the array class itself
    np.dtype,                           # generic dtype wrapper
    np._core.multiarray.scalar,         # 0-d scalar reconstruction
]

# Layer 2 – np.core shim (older numpy exposes these under np.core instead)
try:
    _NUMPY_SAFE_GLOBALS += [
        np.core.multiarray._reconstruct,
        np.core.multiarray.scalar,
    ]
except AttributeError:
    pass  # np.core shim absent in this build — fine

# Layer 3 – concrete dtype classes (numpy >= 1.24 uses numpy.dtypes.* when
# pickling arrays; feat_mean / feat_std will carry Float32DType at minimum)
try:
    import numpy.dtypes as _npdtypes
    _NUMPY_SAFE_GLOBALS += [
        _npdtypes.Float32DType,
        _npdtypes.Float64DType,
        _npdtypes.Int8DType,
        _npdtypes.Int16DType,
        _npdtypes.Int32DType,
        _npdtypes.Int64DType,
        _npdtypes.UInt8DType,
        _npdtypes.UInt16DType,
        _npdtypes.UInt32DType,
        _npdtypes.UInt64DType,
        _npdtypes.BoolDType,
    ]
except (ImportError, AttributeError):
    pass  # numpy < 1.24 — concrete dtype classes not needed


# ── Neural-network architecture ───────────────────────────────────────────────

class _AttentionLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, features)
        out, _ = self.lstm(x)                                # (batch, seq, hidden)
        w = torch.softmax(self.attn(out), dim=1)             # (batch, seq, 1)
        ctx = (out * w).sum(dim=1)                           # (batch, hidden)
        return self.head(ctx).squeeze(-1)                    # (batch,)


# ── Public wrapper ────────────────────────────────────────────────────────────

class DirectionalLSTM:
    """
    Wraps _AttentionLSTM with feature engineering, training, and inference.
    Feature set is calibrated for **15-minute OHLCV bars**.

    Default lookback/horizon wall-clock interpretation at 15-min bars
    ------------------------------------------------------------------
    lookback=60           →  15 hours of context per sequence
    prediction_horizon=5  →  75 minutes ahead

    Usage
    -----
    lstm = DirectionalLSTM(lookback=60, prediction_horizon=5)
    lstm.fit(closes, highs, lows, volumes, hmm_regimes, fear_greed)
    p_up = lstm.predict_proba(recent_closes, ..., fear_greed=fg_array)
    """

    def __init__(
        self,
        lookback: int = 60,
        prediction_horizon: int = 5,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.25,
        lr: float = 0.001,
        epochs: int = 100,
        batch_size: int = 64,
        val_split: float = 0.15,
        early_stop_patience: int = 12,
        device: str | None = None,
    ) -> None:
        self.lookback             = lookback
        self.prediction_horizon   = prediction_horizon
        self.hidden_size          = hidden_size
        self.num_layers           = num_layers
        self.dropout              = dropout
        self.lr                   = lr
        self.epochs               = epochs
        self.batch_size           = batch_size
        self.val_split            = val_split
        self.early_stop_patience  = early_stop_patience
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self._net: _AttentionLSTM | None = None
        self._feat_mean: np.ndarray | None = None
        self._feat_std:  np.ndarray | None = None
        self._input_size: int | None = None
        self.is_fitted = False
        self.history: dict[str, list[float]] = {"loss": [], "val_loss": [], "val_acc": []}

    # ── Feature engineering ──────────────────────────────────────────────────

    @staticmethod
    def _make_features(
        closes:       np.ndarray,
        highs:        np.ndarray | None,
        lows:         np.ndarray | None,
        volumes:      np.ndarray | None,
        hmm_regimes:  np.ndarray | None,
        n_hmm_states: int = 3,
        fear_greed:   np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return feature matrix of shape (N, F), calibrated for 15-minute bars.

        Always-present columns (12 total)
        ----------------------------------
        log_ret       — 1-bar log return
        rsi           — RSI-14  (3.5 h momentum window)
        macd          — MACD(12,26) normalised by price  (3 h / 6.5 h)
        bb_pos        — Bollinger Band position, BB-20   (5 h bands)
        vol_4         — realised volatility, 4-bar std   (1 h)
        vol_16        — realised volatility, 16-bar std  (4 h)
        vol_96        — realised volatility, 96-bar std  (24 h / 1 day)
        ema50_ratio   — (price − EMA-50)  / EMA-50       (12.5 h trend)
        ema200_ratio  — (price − EMA-200) / EMA-200      (~2-day trend)
        roc_4         — rate of change over  4 bars      (1 h)
        roc_16        — rate of change over 16 bars      (4 h)
        roc_96        — rate of change over 96 bars      (24 h / 1 day)

        Conditional on highs & lows (+2 columns)
        -----------------------------------------
        atr           — ATR-14 normalised by price       (3.5 h)
        stoch_k       — Stochastic %%K-14 in [0, 1]       (3.5 h)

        Conditional on volumes (+2 columns)
        ------------------------------------
        volume_ratio  — log1p(volume / 20-bar vol MA)    (vs 5 h MA)
        obv_momentum  — signed OBV(5) − OBV(20) ratio    (1.25 h vs 5 h)

        Conditional on hmm_regimes (+n_hmm_states columns)
        ---------------------------------------------------
        hmm_onehot_0 … hmm_onehot_{n_hmm_states-1}

        Conditional on fear_greed (+1 column, appended last)
        ------------------------------------------------------
        fear_greed    — daily Fear & Greed Index in [0, 1]

        Notes
        -----
        - All rolling operations use min_periods=1, so no NaN is introduced
          even during the warm-up period.
        - roc_96 / ema200_ratio will carry imprecise values for the first
          ~96 / ~200 bars respectively; this is handled by zero-filling.
        - The conditional blocks must be called with the same combination of
          optional inputs at both fit() and predict_proba() time, because they
          control the total input_size stored in the checkpoint.
        """
        s = pd.Series(closes.astype(np.float64))

        # ── Log return ────────────────────────────────────────────────────────
        log_ret = np.log(s / s.shift(1)).fillna(0.0).values

        # ── RSI-14  (3.5 h at 15-min bars) ────────────────────────────────────
        delta = s.diff()
        gain  = delta.clip(lower=0).rolling(14, min_periods=1).mean()
        loss  = (-delta).clip(lower=0).rolling(14, min_periods=1).mean()
        rsi   = (100 - 100 / (1 + gain / (loss + 1e-9))).fillna(50.0).values / 100.0

        # ── MACD(12, 26) normalised by price  (3 h / 6.5 h) ──────────────────
        ema12 = s.ewm(span=12, adjust=False).mean()
        ema26 = s.ewm(span=26, adjust=False).mean()
        macd  = ((ema12 - ema26) / (s + 1e-9)).fillna(0.0).values

        # ── Bollinger Band position, BB-20  (5 h window) ─────────────────────
        bb_m   = s.rolling(20, min_periods=1).mean()
        bb_s   = s.rolling(20, min_periods=2).std().replace(0, np.nan)
        bb_pos = ((s - (bb_m - 2 * bb_s)) / (4 * bb_s + 1e-9)).clip(0, 1).fillna(0.5).values

        # ── Realised volatility at 3 horizons ─────────────────────────────────
        #    vol_4  = 4  bars =  1 hour
        #    vol_16 = 16 bars =  4 hours
        #    vol_96 = 96 bars = 24 hours  (1 trading day)
        log_ret_s = pd.Series(log_ret)
        vol_4  = log_ret_s.rolling(4,  min_periods=1).std().fillna(0.0).values
        vol_16 = log_ret_s.rolling(16, min_periods=1).std().fillna(0.0).values
        vol_96 = log_ret_s.rolling(96, min_periods=1).std().fillna(0.0).values

        # ── EMA trend context ─────────────────────────────────────────────────
        #    ema50_ratio  : (price − EMA-50)  / EMA-50   — 12.5 h medium trend
        #    ema200_ratio : (price − EMA-200) / EMA-200  — ~2-day long trend
        ema50        = s.ewm(span=50,  adjust=False).mean()
        ema200       = s.ewm(span=200, adjust=False).mean()
        ema50_ratio  = ((s - ema50)  / (ema50  + 1e-9)).fillna(0.0).values
        ema200_ratio = ((s - ema200) / (ema200 + 1e-9)).fillna(0.0).values

        # ── Rate of change at 3 horizons ──────────────────────────────────────
        #    roc_4  = 1 h,  roc_16 = 4 h,  roc_96 = 24 h
        roc_4  = (s / s.shift(4).replace(0, np.nan)  - 1).fillna(0.0).values
        roc_16 = (s / s.shift(16).replace(0, np.nan) - 1).fillna(0.0).values
        roc_96 = (s / s.shift(96).replace(0, np.nan) - 1).fillna(0.0).values

        cols = [
            log_ret,
            rsi, macd, bb_pos,
            vol_4, vol_16, vol_96,
            ema50_ratio, ema200_ratio,
            roc_4, roc_16, roc_96,
        ]

        # ── ATR-14 + Stochastic %%K-14  (conditional on highs & lows) ──────────
        #    ATR-14   = 3.5 h average true range, normalised by price
        #    Stoch %%K = 3.5 h stochastic oscillator in [0, 1]
        if highs is not None and lows is not None:
            h = pd.Series(highs.astype(np.float64))
            l = pd.Series(lows.astype(np.float64))

            # ATR
            tr  = pd.concat(
                [h - l, (h - s.shift()).abs(), (l - s.shift()).abs()], axis=1
            ).max(axis=1)
            atr = (tr.rolling(14, min_periods=1).mean() / (s + 1e-9)).fillna(0.0).values

            # Stochastic %K
            h14     = h.rolling(14, min_periods=1).max()
            l14     = l.rolling(14, min_periods=1).min()
            stoch_k = ((s - l14) / (h14 - l14 + 1e-9)).clip(0, 1).fillna(0.5).values

            cols += [atr, stoch_k]

        # ── Volume ratio + OBV momentum  (conditional on volumes) ────────────
        #    volume_ratio : log1p(v / 20-bar vol MA)      — v vs 5 h MA
        #    obv_momentum : signed (OBV_5 − OBV_20) / |OBV_20|
        #                   short OBV (1.25 h) vs long OBV (5 h);
        #                   positive = accumulation, negative = distribution
        if volumes is not None:
            v    = pd.Series(volumes.astype(np.float64))
            v_ma = v.rolling(20, min_periods=1).mean().replace(0, np.nan)
            v_rat = np.log1p((v / v_ma).fillna(1.0).values)

            direction = np.sign(s.diff().fillna(0))
            obv       = (direction * v).cumsum()
            obv_5     = obv.rolling(5,  min_periods=1).mean()
            obv_20    = obv.rolling(20, min_periods=1).mean()
            obv_mom   = (
                (obv_5 - obv_20) / (obv_20.abs() + 1e-9)
            ).clip(-3, 3).fillna(0.0).values

            cols += [v_rat, obv_mom]

        mat = np.column_stack(cols)

        # ── HMM one-hot encoding ──────────────────────────────────────────────
        if hmm_regimes is not None:
            ohe = np.zeros((len(hmm_regimes), n_hmm_states), dtype=np.float32)
            for i, r in enumerate(hmm_regimes):
                if 0 <= int(r) < n_hmm_states:
                    ohe[i, int(r)] = 1.0
            mat = np.hstack([mat, ohe])

        # ── Fear & Greed Index ────────────────────────────────────────────────
        # Appended last so it does not shift indices of existing features,
        # which matters when loading older models that pre-date this column.
        if fear_greed is not None:
            fg = np.asarray(fear_greed, dtype=np.float32).reshape(-1, 1)
            if len(fg) != len(mat):
                raise ValueError(
                    f"fear_greed length {len(fg)} does not match "
                    f"feature matrix length {len(mat)}."
                )
            mat = np.hstack([mat, fg])

        return mat.astype(np.float32)

    def _make_labels(self, closes: np.ndarray) -> np.ndarray:
        labels = np.zeros(len(closes), dtype=np.float32)
        h = self.prediction_horizon
        for i in range(len(closes) - h):
            labels[i] = 1.0 if closes[i + h] > closes[i] else 0.0
        return labels

    def _make_sequences(
        self, feat: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(self.lookback, len(feat)):
            X.append(feat[i - self.lookback : i])
            y.append(labels[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(
        self,
        closes:      np.ndarray,
        highs:       np.ndarray | None = None,
        lows:        np.ndarray | None = None,
        volumes:     np.ndarray | None = None,
        hmm_regimes: np.ndarray | None = None,
        fear_greed:  np.ndarray | None = None,
    ) -> "DirectionalLSTM":
        """
        Train the model on 15-minute bar data.

        Parameters
        ----------
        closes      : 1-D float32 close-price array, length N
        highs       : optional high-price array, length N
                      (enables ATR and Stochastic %%K features)
        lows        : optional low-price array, length N
                      (enables ATR and Stochastic %%K features)
        volumes     : optional volume array, length N
                      (enables volume-ratio and OBV-momentum features)
        hmm_regimes : optional integer regime array, length N or N-1
                      (enables HMM one-hot features)
        fear_greed  : optional float32 array of shape (N,) with values in
                      [0, 1], aligned bar-by-bar with closes.  Produced by
                      train_models.align_fear_greed_to_bars().  When None
                      the feature column is simply omitted (input_size will
                      be one smaller).
        """
        feat   = self._make_features(
            closes, highs, lows, volumes, hmm_regimes, fear_greed=fear_greed
        )
        labels = self._make_labels(closes)

        # Normalise (fit on training portion only)
        split = int(len(feat) * (1 - self.val_split))
        self._feat_mean = feat[:split].mean(axis=0)
        self._feat_std  = feat[:split].std(axis=0) + 1e-8
        feat_n = np.nan_to_num((feat - self._feat_mean) / self._feat_std)

        X, y = self._make_sequences(feat_n, labels)

        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        self._input_size = X.shape[2]
        self._net = _AttentionLSTM(
            self._input_size, self.hidden_size, self.num_layers, self.dropout
        ).to(self.device)

        opt   = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        crit  = nn.BCELoss()
        dl    = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
            batch_size=self.batch_size,
            shuffle=True,
        )

        best_val, no_improve, best_state = float("inf"), 0, None
        X_val_t = torch.from_numpy(X_val).to(self.device)
        y_val_t = torch.from_numpy(y_val).to(self.device)

        logger.info(
            "LSTM training | device=%s | train=%d val=%d features=%d  "
            "(fear_greed=%s)",
            self.device, len(X_tr), len(X_val), self._input_size,
            "yes" if fear_greed is not None else "no",
        )

        for ep in range(self.epochs):
            self._net.train()
            tr_losses = []
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = crit(self._net(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                opt.step()
                tr_losses.append(loss.item())

            self._net.eval()
            with torch.no_grad():
                val_pred = self._net(X_val_t)
                val_loss = crit(val_pred, y_val_t).item()
                val_acc  = ((val_pred > 0.5) == (y_val_t > 0.5)).float().mean().item()

            sched.step(val_loss)
            self.history["loss"].append(float(np.mean(tr_losses)))
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
            else:
                no_improve += 1

            if (ep + 1) % 10 == 0:
                logger.info(
                    "Epoch %d/%d | loss=%.4f | val_loss=%.4f | val_acc=%.3f",
                    ep + 1, self.epochs, np.mean(tr_losses), val_loss, val_acc,
                )

            if no_improve >= self.early_stop_patience:
                logger.info("Early stopping at epoch %d (best val_loss=%.4f)", ep + 1, best_val)
                break

        if best_state:
            self._net.load_state_dict(best_state)
        self._net.eval()
        self.is_fitted = True
        logger.info("LSTM training complete | best val_loss=%.4f", best_val)
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(
        self,
        closes:      np.ndarray,
        highs:       np.ndarray | None = None,
        lows:        np.ndarray | None = None,
        volumes:     np.ndarray | None = None,
        hmm_regimes: np.ndarray | None = None,
        fear_greed:  np.ndarray | None = None,
    ) -> float:
        """
        Return P(price up in next N bars), using the most recent `lookback`
        rows.  At 15-minute resolution, lookback=60 covers 15 hours of
        market history.

        Parameters
        ----------
        fear_greed :
            Must be supplied at inference time if the model was trained with
            it, otherwise the input_size will not match and PyTorch will raise
            a shape error.  Pass None only if the model was trained without it.
        """
        if not self.is_fitted or self._net is None:
            raise RuntimeError("DirectionalLSTM: call fit() first.")

        feat = self._make_features(
            closes, highs, lows, volumes, hmm_regimes, fear_greed=fear_greed
        )
        feat_n = np.nan_to_num((feat - self._feat_mean) / self._feat_std)

        if len(feat_n) < self.lookback:
            raise ValueError(f"Need ≥{self.lookback} rows, got {len(feat_n)}.")

        seq = feat_n[-self.lookback:]                                # (L, F)
        x   = torch.from_numpy(seq).unsqueeze(0).to(self.device)    # (1, L, F)

        with torch.no_grad():
            p = self._net(x).item()

        del x, seq, feat, feat_n
        return float(p)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict":   self._net.state_dict() if self._net else None,
                "feat_mean":    self._feat_mean,
                "feat_std":     self._feat_std,
                "input_size":   self._input_size,
                "lookback":     self.lookback,
                "horizon":      self.prediction_horizon,
                "hidden_size":  self.hidden_size,
                "num_layers":   self.num_layers,
                "dropout":      self.dropout,
                "history":      self.history,
            },
            path,
        )
        logger.info("LSTM saved → %s", path)

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> "DirectionalLSTM":
        # feat_mean / feat_std are numpy arrays whose pickle stream references
        # concrete dtype classes (e.g. numpy.dtypes.Float32DType on numpy>=1.24)
        # in addition to the reconstruction helpers.  All of them are listed in
        # _NUMPY_SAFE_GLOBALS so weights_only=True can remain enabled.
        with torch.serialization.safe_globals(_NUMPY_SAFE_GLOBALS):
            data = torch.load(path, map_location="cpu", weights_only=True)

        obj  = cls(
            lookback=data["lookback"],
            prediction_horizon=data["horizon"],
            hidden_size=data["hidden_size"],
            num_layers=data["num_layers"],
            dropout=data["dropout"],
            device=device,
        )
        obj._feat_mean  = data["feat_mean"]
        obj._feat_std   = data["feat_std"]
        obj._input_size = data["input_size"]
        obj.history     = data.get("history", {})

        if data["state_dict"] and data["input_size"]:
            obj._net = _AttentionLSTM(
                data["input_size"], data["hidden_size"],
                data["num_layers"],  data["dropout"],
            ).to(obj.device)
            obj._net.load_state_dict(data["state_dict"])
            obj._net.eval()

        obj.is_fitted = True
        logger.info("LSTM loaded ← %s", path)
        return obj
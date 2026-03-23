"""
train_models.py
Train one HMM model per instrument defined in config.TRADING_INSTRUMENTS.

Usage
-----
    python train_models.py
"""

import logging

import numpy as np
import pandas as pd

from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.data import BarType
from nautilus_trader.persistence.catalog import ParquetDataCatalog

import config as cfg
from models.hmm_model import BEAR, BULL, RegimeHMM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("train_models")

# Minimum BULL-vs-BEAR forward return separation for the model to be accepted.
# Below this threshold the model cannot distinguish bullish from bearish regimes
# and will not generalise to unseen data.
_MIN_SEPARATION = 0.001   # ~0.1% per 24h

# Forward return horizon used for the signal quality gate (must match
# RegimeHMM._LABEL_HORIZON so the check is consistent with training labeling).
_QUALITY_HORIZON = 96     # 24h at 15-min resolution


# ── Data loading ──────────────────────────────────────────────────────────────

def load_bars_as_arrays(
    catalog:      ParquetDataCatalog,
    bar_type_str: str,
    start:        str,
    end:          str,
) -> dict[str, np.ndarray]:
    """
    Load Bar objects from the ParquetDataCatalog for the given bar_type and
    date range.  Returns numpy arrays of OHLCV and nanosecond timestamps.
    """
    start_ns = dt_to_unix_nanos(pd.Timestamp(start, tz="UTC"))
    end_ns   = dt_to_unix_nanos(pd.Timestamp(end,   tz="UTC"))
    bar_type = BarType.from_str(bar_type_str)

    bars = catalog.bars(
        bar_types=[str(bar_type)],
        start=start_ns,
        end=end_ns,
    )
    if not bars:
        raise RuntimeError(
            f"No bars found for {bar_type} between {start} and {end}. "
            "Run fetch_data.py first."
        )

    bars.sort(key=lambda b: b.ts_init)

    data = {
        "closes":     np.array([float(b.close)  for b in bars], dtype=np.float32),
        "highs":      np.array([float(b.high)   for b in bars], dtype=np.float32),
        "lows":       np.array([float(b.low)    for b in bars], dtype=np.float32),
        "volumes":    np.array([float(b.volume) for b in bars], dtype=np.float32),
        "timestamps": np.array([b.ts_init       for b in bars], dtype=np.int64),
    }
    logger.info(
        "Loaded %d bars | %s → %s",
        len(bars),
        pd.Timestamp(data["timestamps"][0],  unit="ns", tz="UTC").strftime("%Y-%m-%d %H:%M"),
        pd.Timestamp(data["timestamps"][-1], unit="ns", tz="UTC").strftime("%Y-%m-%d %H:%M"),
    )
    return data


# ── Signal quality gate ────────────────────────────────────────────────────────

def _check_signal_quality(
    symbol:  str,
    hmm_obj: RegimeHMM,
    closes:  np.ndarray,
    volumes: np.ndarray,
) -> bool:
    """
    Verify that the fitted model's BULL state has materially higher 24h forward
    return than its BEAR state on the training data.

    Returns True if the model passes; False (with a warning) if it does not.

    A model that fails this check was fitted to a training window in which the
    BULL/BEAR states do not have different predictive properties.  Deploying it
    will produce a flat or inverted calibration curve (P(Bull) → next-bar
    return) and consistent losses.

    Actions when this fails
    -----------------------
    1. Try a different random_state in RegimeHMM (see config.py comments).
    2. Extend the training window — a longer window covers more market cycles.
    3. Review the feature set in hmm_model._build_features().
    """
    regimes_train = hmm_obj.predict(closes, volumes)   # labeled, length N-1
    n_train       = len(regimes_train)

    if n_train <= _QUALITY_HORIZON + 10:
        logger.warning(
            "[%s] Signal quality check skipped — training window too short "
            "(%d bars, need > %d).",
            symbol, n_train, _QUALITY_HORIZON + 10,
        )
        return True   # can't check, don't block

    # forward_ret[i] = log(closes[i+1+H] / closes[i+1])
    # closes[i+1] is the close price at aligned bar i.
    forward_ret = np.array([
        np.log((closes[i + 1 + _QUALITY_HORIZON] + 1e-12)
               / (closes[i + 1]                  + 1e-12))
        for i in range(n_train - _QUALITY_HORIZON)
    ], dtype=np.float64)

    labeled_clipped = regimes_train[:n_train - _QUALITY_HORIZON]

    bull_mask = labeled_clipped == BULL
    bear_mask = labeled_clipped == BEAR

    bull_fwd = float(forward_ret[bull_mask].mean()) if bull_mask.sum() > 0 else 0.0
    bear_fwd = float(forward_ret[bear_mask].mean()) if bear_mask.sum() > 0 else 0.0
    sep      = bull_fwd - bear_fwd

    logger.info(
        "[%s] Signal quality | BULL 24h fwd=%+.5f  BEAR 24h fwd=%+.5f  "
        "separation=%+.5f  (threshold=%.4f)",
        symbol, bull_fwd, bear_fwd, sep, _MIN_SEPARATION,
    )

    if sep < _MIN_SEPARATION:
        logger.warning(
            "[%s] !! SIGNAL QUALITY FAILURE !!  separation=%.5f < %.4f.  "
            "The BULL state does not reliably predict positive 24h returns.  "
            "This model will likely produce a flat or inverted calibration "
            "chart.  Remedies: (1) change random_state in RegimeHMM, "
            "(2) extend training window, (3) review features.",
            symbol, sep, _MIN_SEPARATION,
        )
        return False

    logger.info("[%s] Signal quality check PASSED.", symbol)
    return True


# ── Per-instrument training ───────────────────────────────────────────────────

def train_instrument(
    catalog:  ParquetDataCatalog,
    inst_cfg: dict,
    start:    str,
    end:      str,
) -> None:
    """Fit and save one HMM for a single instrument."""
    symbol   = inst_cfg["binance_symbol"]
    hmm_path = inst_cfg["hmm_model_path"]

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║  Instrument : %-42s ║", symbol)
    logger.info("║  HMM model  → %-42s ║", hmm_path.name)
    logger.info("╚══════════════════════════════════════════════════════════╝")

    data    = load_bars_as_arrays(catalog, inst_cfg["bar_type_str"], start, end)
    closes  = data["closes"]
    volumes = data["volumes"]

    logger.info("[%s] Fitting HMM on %d bars …", symbol, len(closes))
    hmm_obj = RegimeHMM(
        n_states        = cfg.HMM_N_STATES,
        n_iter          = cfg.HMM_N_ITER,
        covariance_type = cfg.HMM_COVARIANCE,
    )
    hmm_obj.fit(closes, volumes)
    hmm_obj.save(hmm_path)

    # ── Post-fit diagnostics ──────────────────────────────────────────────────
    T = hmm_obj.get_transition_matrix()
    logger.info("[%s] Transition matrix:\n%s", symbol, np.round(T, 4))

    regimes = hmm_obj.predict(closes, volumes)
    dist    = {
        hmm_obj.get_regime_name(r): int((regimes == r).sum())
        for r in range(cfg.HMM_N_STATES)
    }
    total = len(regimes)
    logger.info(
        "[%s] Training regime distribution: %s (total=%d)",
        symbol,
        {k: f"{v} ({100*v/total:.1f}%)" for k, v in dist.items()},
        total,
    )

    ll = hmm_obj.log_likelihood(closes, volumes)
    logger.info("[%s] Training log-likelihood: %.2f", symbol, ll)

    try:
        proba = hmm_obj.predict_proba(closes, volumes)
        logger.info(
            "[%s] predict_proba() shape: %s  (columns: [BEAR, SIDEWAYS, BULL])",
            symbol, proba.shape,
        )
        logger.info(
            "[%s] Mean posteriors | BEAR=%.3f  SIDEWAYS=%.3f  BULL=%.3f",
            symbol,
            float(proba[:, 0].mean()),
            float(proba[:, 1].mean()),
            float(proba[:, 2].mean()),
        )
    except Exception as exc:
        logger.warning("[%s] predict_proba() check failed: %s", symbol, exc)

    # ── Signal quality gate ───────────────────────────────────────────────────
    # Verifies that the BULL state predicts higher 24h forward returns than BEAR.
    # A model that fails this check will not generalise — do not deploy it.
    _check_signal_quality(symbol, hmm_obj, closes, volumes)

    logger.info("[%s] HMM saved → %s", symbol, hmm_path)


# ── Entry point ───────────────────────────────────────────────────────────────

def train(start: str = cfg.FETCH_START, end: str = cfg.BACKTEST_START) -> None:
    """
    Train HMM models for every instrument in cfg.INSTRUMENTS.
    Training uses data BEFORE the backtest window to avoid look-ahead bias.
    """
    catalog = ParquetDataCatalog(cfg.CATALOG_PATH)
    symbols = [i["binance_symbol"] for i in cfg.INSTRUMENTS]

    logger.info(
        "Training HMM for %d instrument(s): %s  |  period: %s → %s",
        len(cfg.INSTRUMENTS), symbols, start, end,
    )

    failed = []
    for inst_cfg in cfg.INSTRUMENTS:
        try:
            train_instrument(catalog, inst_cfg, start, end)
        except Exception as exc:
            logger.error(
                "Training failed for %s: %s",
                inst_cfg["binance_symbol"], exc,
                exc_info=True,
            )
            failed.append(inst_cfg["binance_symbol"])

    logger.info("=" * 60)
    logger.info("Training complete.")
    for inst_cfg in cfg.INSTRUMENTS:
        sym    = inst_cfg["binance_symbol"]
        status = "FAILED" if sym in failed else "OK"
        logger.info("  %-12s  %s → %s", sym, status, inst_cfg["hmm_model_path"].name)
    logger.info("  Models directory: %s", cfg.MODEL_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    train(start=cfg.FETCH_START, end=cfg.BACKTEST_START)
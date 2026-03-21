"""
evaluate_models.py
Comprehensive diagnostics for the trained HMM and LSTM models.

Runs a full evaluation loop for EVERY instrument defined in
config.TRADING_INSTRUMENTS.  Results for each instrument are written to an
isolated sub-directory under cfg.RESULTS_DIR/<ticker>/ so outputs never
overwrite each other.

To evaluate a new asset, add it to TRADING_INSTRUMENTS in config.py and
ensure its models have been trained — no changes are needed in this file.

Usage
-----
    python evaluate_models.py
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.data import BarType
from nautilus_trader.persistence.catalog import ParquetDataCatalog

import config as cfg
from models.hmm_model import RegimeHMM
from models.lstm_model import DirectionalLSTM
from utils.diagnostics import DiagnosticEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("evaluate_models")


def load_arrays(
    catalog:      ParquetDataCatalog,
    start:        str,
    end:          str,
    bar_type_str: str,
) -> dict[str, np.ndarray]:
    """
    Load OHLCV arrays and nanosecond timestamps from the catalog for a given
    bar type and date range.

    Parameters
    ----------
    bar_type_str :
        The full bar-type string for the instrument, e.g. inst_cfg["bar_type_str"].
        Passing this explicitly (rather than reading from cfg) keeps the function
        instrument-agnostic.
    """
    start_ns = dt_to_unix_nanos(pd.Timestamp(start, tz="UTC"))
    end_ns   = dt_to_unix_nanos(pd.Timestamp(end,   tz="UTC"))
    bar_type = BarType.from_str(bar_type_str)
    bars     = catalog.bars(bar_types=[str(bar_type)], start=start_ns, end=end_ns)
    bars.sort(key=lambda b: b.ts_init)
    return {
        "closes":     np.array([float(b.close)  for b in bars], dtype=np.float32),
        "highs":      np.array([float(b.high)   for b in bars], dtype=np.float32),
        "lows":       np.array([float(b.low)    for b in bars], dtype=np.float32),
        "volumes":    np.array([float(b.volume) for b in bars], dtype=np.float32),
        "timestamps": np.array([b.ts_init       for b in bars], dtype=np.int64),
    }


def _load_aligned_fear_greed(timestamps_ns: np.ndarray) -> np.ndarray | None:
    """
    Load the Fear & Greed Index from cfg.FEAR_GREED_PATH and align it to the
    provided bar timestamps, returning a float32 array of values in [0, 1].

    Each bar is mapped to the F&G value for its UTC calendar day.  Gaps in the
    daily series are forward-filled then backward-filled.  Returns None if the
    file is missing or empty so callers can fall back gracefully.

    Parameters
    ----------
    timestamps_ns :
        int64 array of bar open-times in nanoseconds (UTC), as returned by
        load_arrays()["timestamps"].
    """
    if not cfg.FEAR_GREED_PATH.exists():
        logger.warning(
            "Fear & Greed file not found at %s — "
            "evaluation will run without it.",
            cfg.FEAR_GREED_PATH,
        )
        return None

    df = pd.read_parquet(cfg.FEAR_GREED_PATH)
    if df["date"].dt.tz is None:
        df["date"] = df["date"].dt.tz_localize("UTC")

    if df.empty:
        logger.warning("Fear & Greed file is empty — evaluation will run without it.")
        return None

    bar_dates = pd.to_datetime(timestamps_ns, unit="ns", utc=True).normalize()
    fg_series = df.set_index("date")["value"]
    aligned   = fg_series.reindex(bar_dates).ffill().bfill().fillna(50)

    logger.info(
        "Fear & Greed aligned to %d bars (min=%.2f  mean=%.2f  max=%.2f)",
        len(aligned),
        aligned.min() / 100.0,
        aligned.mean() / 100.0,
        aligned.max() / 100.0,
    )
    return aligned.values.astype(np.float32) / 100.0


def evaluate_instrument(
    catalog:  ParquetDataCatalog,
    inst_cfg: dict,
) -> None:
    """Run the full diagnostic suite for a single instrument."""
    symbol = inst_cfg["binance_symbol"]
    ticker = inst_cfg["ticker"]

    # Each instrument writes to its own results sub-directory
    out_dir = cfg.RESULTS_DIR / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    diag = DiagnosticEngine(out_dir)

    logger.info("=" * 60)
    logger.info("Evaluating  %s   →   %s", symbol, out_dir)
    logger.info("=" * 60)

    # ── Guard: models must exist ──────────────────────────────────────────────
    hmm_path  = inst_cfg["hmm_model_path"]
    lstm_path = inst_cfg["lstm_model_path"]

    if not hmm_path.exists():
        logger.error(
            "[%s] HMM model not found at %s — run train_models.py first.",
            symbol, hmm_path,
        )
        return
    if not lstm_path.exists():
        logger.error(
            "[%s] LSTM model not found at %s — run train_models.py first.",
            symbol, lstm_path,
        )
        return

    # ── Load models ───────────────────────────────────────────────────────────
    hmm  = RegimeHMM.load(hmm_path)
    lstm = DirectionalLSTM.load(lstm_path, device="cpu")
    logger.info("[%s] Models loaded.", symbol)

    # ── Load OOS data ─────────────────────────────────────────────────────────
    logger.info(
        "[%s] Loading OOS data (%s → %s) …",
        symbol, cfg.BACKTEST_START, cfg.BACKTEST_END,
    )
    data    = load_arrays(
        catalog, cfg.BACKTEST_START, cfg.BACKTEST_END, inst_cfg["bar_type_str"]
    )
    closes     = data["closes"]
    highs      = data["highs"]
    lows       = data["lows"]
    volumes    = data["volumes"]
    timestamps = data["timestamps"]

    min_bars = cfg.HMM_MIN_HISTORY + cfg.LSTM_LOOKBACK + 100
    if len(closes) < min_bars:
        logger.error(
            "[%s] Only %d OOS bars available; need at least %d. "
            "Fetch more data and re-run.",
            symbol, len(closes), min_bars,
        )
        return

    # ── Load & align Fear & Greed Index ──────────────────────────────────────
    # fear_greed_raw is length N (same as closes).
    # aligned_fg = fear_greed_raw[1:] mirrors the [1:] trim applied to
    # the other arrays so everything stays aligned with hmm regimes.
    fear_greed_raw = _load_aligned_fear_greed(timestamps_ns=timestamps)
    aligned_fg     = fear_greed_raw[1:] if fear_greed_raw is not None else None
    if aligned_fg is not None:
        logger.info("[%s] Fear & Greed ready for OOS evaluation.", symbol)
    else:
        logger.warning(
            "[%s] Fear & Greed unavailable — inference will use 0.5 fallback.",
            symbol,
        )
        # Build neutral fallback so the inference loop never passes None to a
        # model that was trained with the F&G feature column.
        aligned_fg = np.full(len(closes) - 1, 0.5, dtype=np.float32)

    # ── HMM diagnostics ───────────────────────────────────────────────────────
    logger.info("[%s] Running HMM diagnostics …", symbol)

    regimes = hmm.predict(closes, volumes)
    ll      = hmm.log_likelihood(closes, volumes)
    logger.info("[%s] HMM log-likelihood (OOS): %.2f", symbol, ll)

    diag.plot_regime_overlay(closes, regimes, title=f"OOS Regime Detection — {symbol}")
    diag.plot_transition_matrix(hmm.get_transition_matrix())

    regime_counts = {
        hmm.get_regime_name(r): int((regimes == r).sum())
        for r in range(cfg.HMM_N_STATES)
    }
    logger.info("[%s] Regime distribution: %s", symbol, regime_counts)

    # ── LSTM diagnostics ──────────────────────────────────────────────────────
    logger.info("[%s] Running LSTM diagnostics …", symbol)

    diag.plot_training_history(lstm.history)

    aligned_closes  = closes[1:]
    aligned_highs   = highs[1:]
    aligned_lows    = lows[1:]
    aligned_volumes = volumes[1:]

    horizon = cfg.LSTM_PREDICTION_HORIZON
    lb      = cfg.LSTM_LOOKBACK

    pred_probas, true_labels, eval_regimes = [], [], []

    for i in range(lb, len(aligned_closes) - horizon):
        p_up = lstm.predict_proba(
            closes=aligned_closes[:i + 1],
            highs=aligned_highs[:i + 1],
            lows=aligned_lows[:i + 1],
            volumes=aligned_volumes[:i + 1],
            hmm_regimes=regimes[:i + 1],
            fear_greed=aligned_fg[:i + 1],
        )
        label = 1.0 if aligned_closes[i + horizon] > aligned_closes[i] else 0.0
        pred_probas.append(p_up)
        true_labels.append(label)
        eval_regimes.append(int(regimes[i]))

        if i % 500 == 0:
            logger.info(
                "[%s] Inference progress: %d / %d bars",
                symbol, i, len(aligned_closes) - horizon,
            )

    pred_probas  = np.array(pred_probas,  dtype=np.float32)
    true_labels  = np.array(true_labels,  dtype=np.float32)
    eval_regimes = np.array(eval_regimes, dtype=np.int32)

    logger.info("[%s] LSTM classification report (threshold=0.5):", symbol)
    diag.print_classification_report(true_labels.astype(int), pred_probas)

    diag.plot_lstm_accuracy_by_regime(true_labels.astype(int), pred_probas, eval_regimes)

    # Confidence histogram
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(pred_probas, bins=50, edgecolor="k")
    ax.axvline(0.5, color="red",   ls="--", label="50%")
    ax.axvline(
        cfg.MIN_CONFIDENCE_ENTRY, color="green", ls="--",
        label=f"Entry threshold ({cfg.MIN_CONFIDENCE_ENTRY})",
    )
    ax.set_title(f"LSTM P(up) Distribution — {symbol} (OOS)")
    ax.set_xlabel("P(up)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"lstm_prob_distribution_{ticker}.png", dpi=150)
    plt.close(fig)
    logger.info("[%s] Confidence histogram saved.", symbol)

    # ── Walk-forward evaluation ───────────────────────────────────────────────
    # NOTE: DiagnosticEngine.walk_forward_evaluation() calls lstm.predict_proba()
    # internally.  If that method does not yet accept a fear_greed argument the
    # call will produce a shape mismatch because the retrained model has one
    # extra input feature.  Update utils/diagnostics.py to pass fear_greed and
    # then remove this try/except guard.
    logger.info("[%s] Walk-forward evaluation …", symbol)
    try:
        wf_df = diag.walk_forward_evaluation(
            closes=closes,
            highs=highs,
            lows=lows,
            volumes=volumes,
            fear_greed=fear_greed_raw,   # full-length array, not the [1:]-trimmed one
            hmm=hmm,
            lstm=lstm,
            window=cfg.WALK_FORWARD_BARS,
            step=cfg.WALK_FORWARD_STEP,
            min_conf=cfg.MIN_CONFIDENCE_ENTRY,
            )
        if len(wf_df) > 0:
            logger.info("[%s] Walk-forward summary:\n%s", symbol, wf_df.describe())
            wf_df.to_csv(out_dir / f"walk_forward_results_{ticker}.csv", index=False)
    except Exception as exc:
        logger.warning(
            "[%s] Walk-forward evaluation failed: %s — "
            "update DiagnosticEngine.walk_forward_evaluation() to pass "
            "fear_greed and re-run.",
            symbol, exc,
        )

    # ── Signal quality at entry threshold ─────────────────────────────────────
    mask_entry = pred_probas > cfg.MIN_CONFIDENCE_ENTRY
    if mask_entry.sum() > 0:
        acc = (pred_probas[mask_entry] > 0.5) == (true_labels[mask_entry] > 0.5)
        logger.info(
            "[%s] At entry threshold %.2f: n_signals=%d | accuracy=%.3f | mean_p_up=%.3f",
            symbol,
            cfg.MIN_CONFIDENCE_ENTRY,
            int(mask_entry.sum()),
            float(acc.mean()),
            float(pred_probas[mask_entry].mean()),
        )

    logger.info("[%s] All outputs saved to: %s", symbol, out_dir)


def main() -> None:
    catalog = ParquetDataCatalog(cfg.CATALOG_PATH)

    symbols = [i["binance_symbol"] for i in cfg.INSTRUMENTS]
    logger.info(
        "Running evaluation for %d instrument(s): %s",
        len(cfg.INSTRUMENTS), symbols,
    )

    for inst_cfg in cfg.INSTRUMENTS:
        evaluate_instrument(catalog, inst_cfg)

    logger.info("All evaluations complete. Results directory: %s", cfg.RESULTS_DIR)


if __name__ == "__main__":
    main()
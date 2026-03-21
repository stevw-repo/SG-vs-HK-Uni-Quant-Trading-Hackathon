"""
train_models.py
Load bars from the ParquetDataCatalog, engineer features,
train the HMM regime detector and LSTM directional predictor for EVERY
instrument defined in config.TRADING_INSTRUMENTS, then save both models
to disk.

Fear & Greed Index (fetched by fetch_data.py) is loaded from
cfg.FEAR_GREED_PATH and passed to the LSTM as an additional daily sentiment
feature.  It is intentionally excluded from the HMM so that regime labels
remain grounded in price/volume structure only.

To train models for a new asset, add it to TRADING_INSTRUMENTS in config.py
and re-run this script — no changes are needed here.

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
from models.hmm_model import RegimeHMM
from models.lstm_model import DirectionalLSTM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("train_models")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_bars_as_arrays(
    catalog:      ParquetDataCatalog,
    bar_type_str: str,
    start:        str,
    end:          str,
) -> dict[str, np.ndarray]:
    """
    Load Bar objects from the catalog for a given bar_type_str and date range,
    and return numpy arrays: closes, opens, highs, lows, volumes, timestamps.

    Parameters
    ----------
    bar_type_str :
        The full bar-type string for this instrument, e.g.
        inst_cfg["bar_type_str"].  Passed explicitly so this function remains
        instrument-agnostic — it works for any symbol without modification.
    """
    start_ns = dt_to_unix_nanos(pd.Timestamp(start, tz="UTC"))
    end_ns   = dt_to_unix_nanos(pd.Timestamp(end,   tz="UTC"))

    bar_type = BarType.from_str(bar_type_str)
    bars     = catalog.bars(
        bar_types=[str(bar_type)],
        start=start_ns,
        end=end_ns,
    )

    if not bars:
        raise RuntimeError(
            f"No bars found in catalog for {bar_type} between {start} and {end}. "
            "Run fetch_data.py first."
        )

    bars.sort(key=lambda b: b.ts_init)

    data = {
        "closes":     np.array([float(b.close)  for b in bars], dtype=np.float32),
        "opens":      np.array([float(b.open)   for b in bars], dtype=np.float32),
        "highs":      np.array([float(b.high)   for b in bars], dtype=np.float32),
        "lows":       np.array([float(b.low)    for b in bars], dtype=np.float32),
        "volumes":    np.array([float(b.volume) for b in bars], dtype=np.float32),
        "timestamps": np.array([b.ts_init       for b in bars], dtype=np.int64),
    }

    logger.info(
        "Loaded %d bars | range %s → %s",
        len(bars),
        pd.Timestamp(data["timestamps"][0],  unit="ns", tz="UTC").strftime("%Y-%m-%d %H:%M"),
        pd.Timestamp(data["timestamps"][-1], unit="ns", tz="UTC").strftime("%Y-%m-%d %H:%M"),
    )
    return data


# ── Fear & Greed loading ──────────────────────────────────────────────────────

def load_fear_greed(start: str, end: str) -> pd.DataFrame | None:
    """
    Load the Fear & Greed Index from the Parquet file written by fetch_data.py.

    Returns a DataFrame with columns [date (UTC tz-aware), value (int 0–100)],
    or None if the file is missing or empty — callers must handle the None case
    and fall back to a neutral constant so training is never blocked.
    """
    if not cfg.FEAR_GREED_PATH.exists():
        logger.warning(
            "Fear & Greed file not found at %s — LSTM will train without it. "
            "Run fetch_data.py to populate it.",
            cfg.FEAR_GREED_PATH,
        )
        return None

    df = pd.read_parquet(cfg.FEAR_GREED_PATH)

    # Defensive: ensure the date column is tz-aware UTC.
    if df["date"].dt.tz is None:
        df["date"] = df["date"].dt.tz_localize("UTC")

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts   = pd.Timestamp(end,   tz="UTC")
    df = df[(df["date"] >= start_ts) & (df["date"] < end_ts)].reset_index(drop=True)

    if df.empty:
        logger.warning(
            "Fear & Greed: no records in range %s → %s — "
            "LSTM will train without it.",
            start, end,
        )
        return None

    logger.info(
        "Fear & Greed: loaded %d daily records  (%s → %s)",
        len(df),
        df["date"].iloc[0].strftime("%Y-%m-%d"),
        df["date"].iloc[-1].strftime("%Y-%m-%d"),
    )
    return df[["date", "value"]]


def align_fear_greed_to_bars(
    timestamps_ns: np.ndarray,
    fg_df:         pd.DataFrame,
) -> np.ndarray:
    """
    Map each bar's nanosecond timestamp to the Fear & Greed value for that
    UTC calendar day, then normalise the result to [0, 1].

    The F&G index is published once per day, so every 15-minute bar within the
    same UTC day receives the same value.  Gaps in the daily series (e.g. API
    outages) are forward-filled then backward-filled so no NaNs reach the model.
    Any residual NaNs (e.g. if the training window pre-dates the F&G history)
    are replaced by the neutral value 0.5.

    Parameters
    ----------
    timestamps_ns : int64 array — bar open-times in nanoseconds (UTC).
    fg_df         : DataFrame with columns [date (UTC tz-aware), value (0–100)].

    Returns
    -------
    float32 array of shape (len(timestamps_ns),) with values in [0, 1].
    """
    # Snap each bar timestamp to its UTC midnight date.
    bar_dates = pd.to_datetime(timestamps_ns, unit="ns", utc=True).normalize()

    # Index F&G by date for fast reindex lookup.
    fg_series = fg_df.set_index("date")["value"]

    # Reindex to bar dates; fill forward then backward to cover any gaps.
    aligned = fg_series.reindex(bar_dates).ffill().bfill().fillna(50)

    return aligned.values.astype(np.float32) / 100.0


# ── Per-instrument training ───────────────────────────────────────────────────

def train_instrument(
    catalog:  ParquetDataCatalog,
    inst_cfg: dict,
    start:    str,
    end:      str,
) -> None:
    """
    Train one HMM + one LSTM for a single instrument and save both models.

    Fear & Greed is wired into the LSTM only.  The HMM is deliberately kept
    pure (price + volume) so its regime labels stay interpretable as market-
    structure states independent of external sentiment.

    All paths and identifiers are taken from inst_cfg so that this function
    requires no modification when instruments are added or removed.
    """
    symbol    = inst_cfg["binance_symbol"]
    hmm_path  = inst_cfg["hmm_model_path"]
    lstm_path = inst_cfg["lstm_model_path"]

    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  Instrument : %-43s ║", symbol)
    logger.info("║  HMM  →  %-47s ║", hmm_path.name)
    logger.info("║  LSTM →  %-47s ║", lstm_path.name)
    logger.info("╚" + "═" * 58 + "╝")

    # ── Load OHLCV bars ───────────────────────────────────────────────────────
    data = load_bars_as_arrays(
        catalog=catalog,
        bar_type_str=inst_cfg["bar_type_str"],
        start=start,
        end=end,
    )

    closes  = data["closes"]
    highs   = data["highs"]
    lows    = data["lows"]
    volumes = data["volumes"]

    # ── Load & align Fear & Greed Index ──────────────────────────────────────
    # Alignment happens against the full (un-trimmed) timestamp array so that
    # the subsequent [1:] trim keeps everything in sync with hmm_regimes.
    fg_df = load_fear_greed(start=start, end=end)

    if fg_df is not None:
        fear_greed_raw = align_fear_greed_to_bars(
            timestamps_ns=data["timestamps"],
            fg_df=fg_df,
        )
        logger.info(
            "[%s] Fear & Greed aligned to %d bars  "
            "(min=%.2f  mean=%.2f  max=%.2f)",
            symbol, len(fear_greed_raw),
            fear_greed_raw.min(),
            fear_greed_raw.mean(),
            fear_greed_raw.max(),
        )
    else:
        # Neutral fallback: 0.5 keeps the feature column inert rather than
        # omitting it, so the LSTM architecture stays consistent whether or
        # not the F&G file is present.
        fear_greed_raw = np.full(len(closes), 0.5, dtype=np.float32)
        logger.warning(
            "[%s] Fear & Greed unavailable — using neutral fallback (0.5).",
            symbol,
        )

    # ── 1. Train HMM ──────────────────────────────────────────────────────────
    # No sentiment data here — regime detection is grounded in price/volume
    # only so the resulting labels reflect market structure, not mood.
    logger.info("[%s] Training HMM on %d bars …", symbol, len(closes))

    hmm = RegimeHMM(
        n_states=cfg.HMM_N_STATES,
        n_iter=cfg.HMM_N_ITER,
        covariance_type=cfg.HMM_COVARIANCE,
    )
    hmm.fit(closes, volumes)
    hmm.save(hmm_path)
    logger.info("[%s] HMM saved → %s", symbol, hmm_path)

    T = hmm.get_transition_matrix()
    logger.info("[%s] Regime transition matrix:\n%s", symbol, np.round(T, 4))

    # ── 2. Compute HMM regime labels for LSTM features ────────────────────────
    logger.info(
        "[%s] Computing HMM regime labels for LSTM feature augmentation …",
        symbol,
    )
    hmm_regimes = hmm.predict(closes, volumes)
    # hmm_regimes is len(closes)-1 due to internal differencing.
    # Trim all arrays by one bar from the front to maintain alignment.
    aligned_closes     = closes[1:]
    aligned_highs      = highs[1:]
    aligned_lows       = lows[1:]
    aligned_volumes    = volumes[1:]
    aligned_fear_greed = fear_greed_raw[1:]

    # ── 3. Train LSTM ─────────────────────────────────────────────────────────
    # F&G is passed as a slow-moving sentiment context that complements the
    # high-frequency OHLCV and HMM regime features.  Extreme fear / greed
    # readings historically precede mean-reversion and momentum setups in
    # crypto, making this a meaningful addition to the feature set.
    logger.info("[%s] Training LSTM on %d bars …", symbol, len(aligned_closes))

    lstm = DirectionalLSTM(
        lookback=cfg.LSTM_LOOKBACK,
        prediction_horizon=cfg.LSTM_PREDICTION_HORIZON,
        hidden_size=cfg.LSTM_HIDDEN_SIZE,
        num_layers=cfg.LSTM_NUM_LAYERS,
        dropout=cfg.LSTM_DROPOUT,
        lr=cfg.LSTM_LR,
        epochs=cfg.LSTM_EPOCHS,
        batch_size=cfg.LSTM_BATCH_SIZE,
        val_split=cfg.LSTM_VAL_SPLIT,
        early_stop_patience=cfg.LSTM_EARLY_STOP_PATIENCE,
    )
    lstm.fit(
        closes=aligned_closes,
        highs=aligned_highs,
        lows=aligned_lows,
        volumes=aligned_volumes,
        hmm_regimes=hmm_regimes,
        fear_greed=aligned_fear_greed,
    )
    lstm.save(lstm_path)
    logger.info("[%s] LSTM saved → %s", symbol, lstm_path)

    final_val_acc = (
        lstm.history["val_acc"][-1]
        if lstm.history.get("val_acc")
        else float("nan")
    )
    logger.info("[%s] Final val accuracy: %.3f", symbol, final_val_acc)


# ── Entry point ───────────────────────────────────────────────────────────────

def train(start: str = cfg.FETCH_START, end: str = cfg.BACKTEST_START) -> None:
    """
    Train HMM + LSTM for every instrument in cfg.INSTRUMENTS.

    Training uses data BEFORE the backtest window to avoid look-ahead bias.
    Each instrument produces its own model files (named by ticker) so models
    never overwrite each other.
    """
    catalog = ParquetDataCatalog(cfg.CATALOG_PATH)

    symbols = [i["binance_symbol"] for i in cfg.INSTRUMENTS]
    logger.info(
        "Training %d instrument(s): %s  |  period: %s → %s",
        len(cfg.INSTRUMENTS), symbols, start, end,
    )

    failed = []
    for inst_cfg in cfg.INSTRUMENTS:
        try:
            train_instrument(
                catalog=catalog,
                inst_cfg=inst_cfg,
                start=start,
                end=end,
            )
        except Exception as exc:
            # Log the error but continue so one bad instrument does not abort
            # training for the remaining instruments.
            logger.error(
                "Training failed for %s: %s",
                inst_cfg["binance_symbol"], exc,
                exc_info=True,
            )
            failed.append(inst_cfg["binance_symbol"])

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Training complete.")
    for inst_cfg in cfg.INSTRUMENTS:
        sym = inst_cfg["binance_symbol"]
        if sym in failed:
            logger.warning("  %-12s  FAILED — see errors above.", sym)
        else:
            logger.info(
                "  %-12s  HMM → %-30s  LSTM → %s",
                sym,
                inst_cfg["hmm_model_path"].name,
                inst_cfg["lstm_model_path"].name,
            )
    logger.info("  Models saved to: %s", cfg.MODEL_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    train(start=cfg.FETCH_START, end=cfg.BACKTEST_START)
"""
config.py — Central configuration for the HMM-LSTM crypto trading workflow.
Fill in your Roostoo API credentials before running live_trading.py.

Adding / removing instruments
------------------------------
  1. Add or remove a (binance_symbol, roostoo_pair) tuple in TRADING_INSTRUMENTS.
  2. Add the matching TestInstrumentProvider factory in INSTRUMENT_PROVIDERS.
  That is all — fetch_data, backtest, live_trading, and evaluate_models will
  automatically pick up every change with no further edits required.
"""

from decimal import Decimal
from pathlib import Path
from datetime import datetime

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
DATA_DIR     = BASE_DIR / "data"
CATALOG_PATH = DATA_DIR / "catalog"
MODEL_DIR    = BASE_DIR / "saved_models"
LOG_DIR      = BASE_DIR / "logs"
RESULTS_DIR  = BASE_DIR / "results"

for _d in [DATA_DIR, CATALOG_PATH, MODEL_DIR, LOG_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  INSTRUMENTS — the only section you need to edit to add / remove assets
# ══════════════════════════════════════════════════════════════════════════════

# Step 1 ── List every (Binance symbol, Roostoo pair) you want to trade.
TRADING_INSTRUMENTS: list[tuple[str, str]] = [
    ("BTCUSDT", "BTC/USD"),
    ("SOLUSDT", "SOL/USD"),
    # ("SOLUSDT", "SOL/USD"),  ← uncomment (and add provider below) to add SOL
]

# Step 2 ── For each symbol above, provide its NautilusTrader instrument
# factory.  Used by fetch_data.py and backtest.py only (not live trading).
def _build_instrument_providers() -> dict:
    from nautilus_trader.test_kit.providers import TestInstrumentProvider
    return {
        "BTCUSDT": TestInstrumentProvider.btcusdt_binance,
        "SOLUSDT": TestInstrumentProvider.ethusdt_binance,
        # "SOLUSDT": TestInstrumentProvider.solusdt_binance,  ← add here too
    }

INSTRUMENT_PROVIDERS: dict = _build_instrument_providers()


# ══════════════════════════════════════════════════════════════════════════════
#  Bar specification — shared across all instruments
# ══════════════════════════════════════════════════════════════════════════════
VENUE           = "BINANCE"
BAR_STEP        = 15             # ← changed from 1 to 15 (15-minute bars)
BAR_AGGREGATION = "MINUTE"       # must match a BarAggregation enum name
BAR_PRICE_TYPE  = "LAST"         # must match a PriceType enum name


# ── Derived instrument configs (auto-generated — do not edit) ─────────────────
def _build_instrument_configs() -> list[dict]:
    configs = []
    for binance_symbol, roostoo_pair in TRADING_INSTRUMENTS:
        # Derive a short ticker: BTCUSDT → btc, ETHUSDT → eth, SOLUSDT → sol …
        ticker = binance_symbol.lower()
        for suffix in ("usdt", "usd", "busd", "usdc", ):
            if ticker.endswith(suffix):
                ticker = ticker[: -len(suffix)]
                break

        instrument_id_str = f"{binance_symbol}.{VENUE}"
        bar_type_str = (
            f"{instrument_id_str}-{BAR_STEP}-{BAR_AGGREGATION}"
            f"-{BAR_PRICE_TYPE}-EXTERNAL"
        )
        configs.append({
            "binance_symbol":    binance_symbol,
            "roostoo_pair":      roostoo_pair,
            "ticker":            ticker,                            # "btc", "eth" …
            "instrument_id_str": instrument_id_str,                 # "BTCUSDT.BINANCE"
            "bar_type_str":      bar_type_str,
            "hmm_model_path":    MODEL_DIR / f"hmm_{ticker}.pkl",
            "lstm_model_path":   MODEL_DIR / f"lstm_{ticker}.pt",
        })
    return configs


INSTRUMENTS:    list[dict]        = _build_instrument_configs()
INSTRUMENT_MAP: dict[str, dict]   = {i["instrument_id_str"]: i for i in INSTRUMENTS}
SYMBOL_MAP:     dict[str, dict]   = {i["binance_symbol"]:    i for i in INSTRUMENTS}


# ══════════════════════════════════════════════════════════════════════════════
#  Roostoo API
# ══════════════════════════════════════════════════════════════════════════════
ROOSTOO_BASE_URL    = "https://mock-api.roostoo.com"
ROOSTOO_API_KEY     = "YOUR_API_KEY_HERE"       # ← replace
ROOSTOO_SECRET_KEY  = "YOUR_SECRET_KEY_HERE"    # ← replace
ROOSTOO_AMOUNT_PREC = 6                         # decimal places for asset qty

# ── Data Periods ──────────────────────────────────────────────────────────────
FETCH_START      = "2023-10-01"
FETCH_END        = "2025-12-31"
BACKTEST_START   = "2026-03-10"
BACKTEST_END     = datetime.now().strftime("%Y-%m-%d")
STARTING_BALANCE = 1_000_000.0

# ── Fear & Greed Index ────────────────────────────────────────────────────────
# Sourced from the free Alternative.me public API — no API key required.
# The index is published once per day (one integer value 0–100).
# fetch_data.py stores the full filtered history at FEAR_GREED_PATH as Parquet.
# Downstream files (train_models, backtest, live_trading) read it from disk;
# they should join on the UTC date of each bar to get the daily value.
FEAR_GREED_URL  = "https://api.alternative.me/fng/"
FEAR_GREED_PATH = DATA_DIR / "fear_greed.parquet"

# ── HMM ───────────────────────────────────────────────────────────────────────
HMM_N_STATES    = 3
HMM_N_ITER      = 300
HMM_COVARIANCE  = "full"
HMM_MIN_HISTORY = 200

# ── LSTM ──────────────────────────────────────────────────────────────────────
LSTM_LOOKBACK            = 60
LSTM_PREDICTION_HORIZON  = 5
LSTM_HIDDEN_SIZE         = 128
LSTM_NUM_LAYERS          = 2
LSTM_DROPOUT             = 0.25
LSTM_EPOCHS              = 100
LSTM_BATCH_SIZE          = 64
LSTM_LR                  = 0.001
LSTM_VAL_SPLIT           = 0.15
LSTM_EARLY_STOP_PATIENCE = 12

# ── Kelly / Risk ──────────────────────────────────────────────────────────────
KELLY_FRACTION          = 0.40
MAX_POSITION_PCT        = 0.65
MIN_CONFIDENCE_ENTRY    = 0.53
MIN_CONFIDENCE_EXIT     = 0.47
EXPECTED_WIN_LOSS_RATIO = 2.67

STOP_LOSS_PCT      = 0.015
TAKE_PROFIT_PCT    = 0.040
TRAILING_STOP_PCT  = 0.020

# ── Commission ────────────────────────────────────────────────────────────────
TAKER_FEE = Decimal("0.001")
MAKER_FEE = Decimal("0.0005")

# ── Live Trading ──────────────────────────────────────────────────────────────
LOOP_SLEEP_SECS    = 5
BALANCE_SYNC_BARS  = 15
ORDER_TIMEOUT_SECS = 30

# ── Diagnostics ───────────────────────────────────────────────────────────────
WALK_FORWARD_BARS = 3_000
WALK_FORWARD_STEP = 500
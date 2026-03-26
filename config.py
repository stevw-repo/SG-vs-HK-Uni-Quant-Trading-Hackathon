"""
config.py — HMM-only crypto trading workflow.

Adding a new trading pair
--------------------------
Simply append a tuple to TRADING_INSTRUMENTS below:

    ("SOLUSDT", "SOL/USD"),

That is the ONLY file you need to edit.  Instrument definitions are fetched
automatically from Binance at runtime — no code changes in any other file.

Roostoo pair availability
--------------------------
The mock Roostoo platform supports a limited set of pairs.
Run  GET https://mock-api.roostoo.com/v3/exchangeInfo  to see what is live.
For backtesting the Roostoo pair is not used, so you can add any Binance
symbol freely.  For live trading ensure the Roostoo pair exists on the mock
exchange first.

Fill in your Roostoo API credentials in the ROOSTOO section before running
live_trading.py.
"""

from decimal import Decimal
from datetime import datetime, timedelta
from pathlib import Path

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
#  INSTRUMENTS  ← edit this list to add / remove pairs
# ══════════════════════════════════════════════════════════════════════════════
#
# Format: (BINANCE_SPOT_SYMBOL, ROOSTOO_PAIR)
#
# Any Binance Spot USDT pair is valid here — instrument specs are fetched
# automatically from Binance via utils/binance_instruments.py.
# The disk cache (INSTRUMENT_CACHE_PATH) means Binance is only queried once
# per machine; subsequent runs load from the pickle file in < 1 ms.
#
# Examples of valid Binance Spot symbols:
#   BTCUSDT  ETHUSDT  BNBUSDT  SOLUSDT  XRPUSDT  DOGEUSDT  ADAUSDT
#   AVAXUSDT DOTUSDT  LINKUSDT LTCUSDT  MATICUSDT SHIBUSDT  TRXUSDT
#   UNIUSDT  ATOMUSDT FILUSDT  NEARUSDT APTUSDT   ARBUSDT   OPUSDT
#
TRADING_INSTRUMENTS: list[tuple[str, str]] = [
    ("BTCUSDT", "BTC/USD"),
    ("ETHUSDT", "ETH/USD"),
    ("BNBUSDT", "BNB/USD"),
    # ── Add any pair below — no other files need to change ──────────────────
    ("SOLUSDT",   "SOL/USD"),
    ("XRPUSDT",   "XRP/USD"),
    ("DOGEUSDT",  "DOGE/USD"),
    ("ADAUSDT",   "ADA/USD"),
    ("AVAXUSDT",  "AVAX/USD"),
    ("DOTUSDT",   "DOT/USD"),
    ("LINKUSDT",  "LINK/USD"),
    ("LTCUSDT",   "LTC/USD"),
    # ("TAOUSDT",   "TAO/USD"),
    # ("FETUSDT",   "FET/USD"),
]

# ── Bar specification ─────────────────────────────────────────────────────────
VENUE           = "BINANCE"
BAR_STEP        = 5
BAR_AGGREGATION = "MINUTE"
BAR_PRICE_TYPE  = "LAST"

# ── Instrument disk cache ─────────────────────────────────────────────────────
# Binance instrument specs (price_precision, size_precision, lot sizes, etc.)
# are cached here so the exchange-info API is only called ONCE per machine.
# Delete this file to force a refresh (e.g. after Binance updates tick sizes).
INSTRUMENT_CACHE_PATH = DATA_DIR / "instrument_cache.pkl"


def _build_instrument_configs() -> list[dict]:
    """Derive per-instrument metadata from TRADING_INSTRUMENTS."""
    configs = []
    for binance_symbol, roostoo_pair in TRADING_INSTRUMENTS:
        ticker = binance_symbol.lower()
        for suffix in ("usdt", "usd", "busd", "usdc"):
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
            "ticker":            ticker,
            "instrument_id_str": instrument_id_str,
            "bar_type_str":      bar_type_str,
            "hmm_model_path":    MODEL_DIR / f"hmm_{ticker}.pkl",
        })
    return configs


INSTRUMENTS:    list[dict]      = _build_instrument_configs()
INSTRUMENT_MAP: dict[str, dict] = {i["instrument_id_str"]: i for i in INSTRUMENTS}
SYMBOL_MAP:     dict[str, dict] = {i["binance_symbol"]:    i for i in INSTRUMENTS}

# ══════════════════════════════════════════════════════════════════════════════
#  Roostoo API
# ══════════════════════════════════════════════════════════════════════════════
ROOSTOO_BASE_URL   = "https://mock-api.roostoo.com"
ROOSTOO_API_KEY    = "TBshw3KMSyYWso0poqwML2GiXdd5Y0bB7b7mveomD1jvtV0mf0T0G5VeZNNphAwg"    # ← replace before live trading
ROOSTOO_SECRET_KEY = "NIKGWlW8AYGRj0VHjnijU46No6A6ha1HZvY9qgdvuozH9zPFh56oXs70ITlz7KBt" # ← replace before live trading
# ROOSTOO_API_KEY    = "4mBXHQ2ihr5gos7S9dDiXcoxPFNu9RKyoXH91dgXIfqzYI4gjtsUKZGNRAncww91" # ← IGNORE (mock credentials)
# ROOSTOO_SECRET_KEY = "Xyvcu2WL8BtDGfflvePuLYa8P4ZkZ3Pv9THbGfbmP8qAiH4dwJWqz8nVlZOzTA7M" # ← IGNORE (mock credentials)

# ── Data periods ──────────────────────────────────────────────────────────────
FETCH_START = "2021-01-01"
FETCH_END   = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

BACKTEST_START = "2026-03-21"   # ← update after retraining; must be > FETCH_START
BACKTEST_END   = "2026-03-24"   # ← fixed; update manually after re-fetching data

STARTING_BALANCE = 1_000_000.0

# ── HMM ───────────────────────────────────────────────────────────────────────
HMM_N_STATES    = 3
HMM_N_ITER      = 300
HMM_COVARIANCE  = "full"
HMM_MIN_HISTORY = 150

# ── Live warm-up ──────────────────────────────────────────────────────────────
# Number of historical 15-min bars to pre-fetch from Binance REST on startup.
# Must be >= HMM_MIN_HISTORY.  50-bar buffer absorbs any gaps or stale bars.
# Set to 0 to disable (not recommended for live trading).
# Backtest is unaffected — the strategy config defaults warmup_bars=0 there.
WARMUP_BARS = HMM_MIN_HISTORY + 50   # ← NEW

# ── Entry ─────────────────────────────────────────────────────────────────────
MIN_BULL_PROBA          = 0.45
MIN_KELLY_FRACTION      = 0.005
TREND_EMA_BARS          = 2
TREND_LOOKBACK_BARS     = 4
BULL_ENTRY_CONSECUTIVE  = 1
BEAR_EXIT_CONSECUTIVE   = 2

# ── Kelly / position sizing ───────────────────────────────────────────────────
KELLY_FRACTION     = 0.70
MAX_POSITION_PCT   = 0.70
COMMISSION_RATE    = 0.001

# ── Exit ──────────────────────────────────────────────────────────────────────
TAKE_PROFIT_PCT    = 0.0025
TRAIL_BULL_PCT     = 0.012
TRAIL_SIDEWAYS_PCT = 0.008
TRAIL_BEAR_PCT     = 0.004
BEAR_EXIT_PROBA    = 0.40
MAX_HOLDING_BARS   = 500

# ── Re-entry cooldown ─────────────────────────────────────────────────────────
MIN_BARS_BETWEEN_TRADES = 8

# ── Commission ────────────────────────────────────────────────────────────────
TAKER_FEE = Decimal("0.001")

# ── Live trading ──────────────────────────────────────────────────────────────
LOOP_SLEEP_SECS    = 5
BALANCE_SYNC_BARS  = 15
ORDER_TIMEOUT_SECS = 30
SYNC_INTERVAL_SECS = 300   # ← NEW: balance + position reconcile every 5 min
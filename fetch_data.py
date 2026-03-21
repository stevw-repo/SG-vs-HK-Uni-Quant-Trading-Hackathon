"""
fetch_data.py
Download historical Binance OHLCV bars for ALL instruments defined in
config.TRADING_INSTRUMENTS and the daily Fear & Greed Index from
Alternative.me, then persist everything to disk.

Bar data  → local ParquetDataCatalog (cfg.CATALOG_PATH)
F&G data  → Parquet file             (cfg.FEAR_GREED_PATH)

To fetch data for a new asset, add it to TRADING_INSTRUMENTS in config.py and
re-run this script — no other changes are needed.

Usage
-----
    python fetch_data.py
    python fetch_data.py --start 2023-10-01 --end 2024-04-30
    python fetch_data.py --start 2023-10-01 --end 2024-04-30 --reset
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import pandas as pd

from nautilus_trader.adapters.binance.common.enums import BinanceAccountType, BinanceKlineInterval
from nautilus_trader.adapters.binance.http.client import BinanceHttpClient
from nautilus_trader.adapters.binance.spot.http.market import BinanceSpotMarketHttpAPI
from nautilus_trader.common.component import LiveClock
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import AggregationSource, BarAggregation, PriceType
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog

import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("fetch_data")

KLINES_PER_REQUEST = 1000

# ── Map (BAR_STEP, BAR_AGGREGATION) → Binance kline interval enum ─────────────
_KLINE_INTERVAL_MAP: dict[tuple[int, str], BinanceKlineInterval] = {
    (1,  "MINUTE"):  BinanceKlineInterval.MINUTE_1,
    (3,  "MINUTE"):  BinanceKlineInterval.MINUTE_3,
    (5,  "MINUTE"):  BinanceKlineInterval.MINUTE_5,
    (15, "MINUTE"):  BinanceKlineInterval.MINUTE_15,
    (30, "MINUTE"):  BinanceKlineInterval.MINUTE_30,
    (60, "MINUTE"):  BinanceKlineInterval.HOUR_1,
}

# Duration of one bar in milliseconds — derived from config, not hardcoded
_BAR_MS_MAP: dict[tuple[int, str], int] = {
    (1,  "MINUTE"):       60_000,
    (3,  "MINUTE"):      180_000,
    (5,  "MINUTE"):      300_000,
    (15, "MINUTE"):      900_000,
    (30, "MINUTE"):    1_800_000,
    (60, "MINUTE"):    3_600_000,
}

_interval_key = (cfg.BAR_STEP, cfg.BAR_AGGREGATION)
if _interval_key not in _KLINE_INTERVAL_MAP:
    raise ValueError(
        f"Unsupported BAR_STEP / BAR_AGGREGATION in config: {_interval_key}. "
        f"Supported combinations: {list(_KLINE_INTERVAL_MAP.keys())}"
    )

KLINE_INTERVAL: BinanceKlineInterval = _KLINE_INTERVAL_MAP[_interval_key]
BAR_DURATION_MS: int                 = _BAR_MS_MAP[_interval_key]


# ── Helpers ───────────────────────────────────────────────────────────────────

def dt_to_ms(dt_str: str) -> int:
    """Convert 'YYYY-MM-DD' string to millisecond UTC timestamp."""
    return int(
        datetime.strptime(dt_str, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp() * 1000
    )


def ms_to_ns(ms: int) -> int:
    return ms * 1_000_000


# ── Kline fetching ────────────────────────────────────────────────────────────

async def fetch_klines_range(
    market_api: BinanceSpotMarketHttpAPI,
    symbol:     str,
    start_ms:   int,
    end_ms:     int,
) -> list[dict]:
    """
    Paginate the Binance klines endpoint to cover the full [start_ms, end_ms)
    range for a single symbol.  Returns a list of raw kline dicts.
    """
    all_klines: list[dict] = []
    current = start_ms

    while current < end_ms:
        batch_end = min(current + KLINES_PER_REQUEST * BAR_DURATION_MS, end_ms)
        logger.info(
            "  %s | %s – %s",
            symbol,
            pd.Timestamp(current,   unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M"),
            pd.Timestamp(batch_end, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M"),
        )

        klines = await market_api.query_klines(
            symbol=symbol,
            interval=KLINE_INTERVAL,
            start_time=current,
            end_time=batch_end,
            limit=KLINES_PER_REQUEST,
        )

        if not klines:
            logger.warning("Empty response for %s range %d – %d", symbol, current, batch_end)
            break

        for k in klines:
            if hasattr(k, "open_time"):
                all_klines.append({
                    "open_time":  int(k.open_time),
                    "open":       float(k.open),
                    "high":       float(k.high),
                    "low":        float(k.low),
                    "close":      float(k.close),
                    "volume":     float(k.volume),
                    "close_time": int(k.close_time),
                })
            else:
                # Raw list/tuple format: [open_time, open, high, low, close, volume, …]
                all_klines.append({
                    "open_time":  int(k[0]),
                    "open":       float(k[1]),
                    "high":       float(k[2]),
                    "low":        float(k[3]),
                    "close":      float(k[4]),
                    "volume":     float(k[5]),
                    "close_time": int(k[6]),
                })

        last_ts = all_klines[-1]["open_time"]
        if last_ts + BAR_DURATION_MS >= end_ms:
            break
        current = last_ts + BAR_DURATION_MS

        await asyncio.sleep(0.25)   # polite rate limiting

    return all_klines


def klines_to_bars(
    klines:     list[dict],
    bar_type:   BarType,
    price_prec: int,
    size_prec:  int,
) -> list[Bar]:
    bars = []
    for k in klines:
        ts = ms_to_ns(k["open_time"])
        try:
            bars.append(Bar(
                bar_type=bar_type,
                open=Price(k["open"],   price_prec),
                high=Price(k["high"],   price_prec),
                low=Price(k["low"],     price_prec),
                close=Price(k["close"], price_prec),
                volume=Quantity(k["volume"], size_prec),
                ts_event=ts,
                ts_init=ts,
            ))
        except Exception as exc:
            logger.warning("Skipping kline %s: %s", k, exc)
    return bars


# ── Per-instrument fetch ───────────────────────────────────────────────────────

async def fetch_instrument(
    market_api: BinanceSpotMarketHttpAPI,
    catalog:    ParquetDataCatalog,
    inst_cfg:   dict,
    start_ms:   int,
    end_ms:     int,
) -> None:
    """Fetch, convert, and persist all bars for one instrument config dict."""
    symbol = inst_cfg["binance_symbol"]

    provider_fn = cfg.INSTRUMENT_PROVIDERS.get(symbol)
    if provider_fn is None:
        logger.error(
            "No INSTRUMENT_PROVIDERS entry for '%s'. "
            "Add one to config.py and re-run.",
            symbol,
        )
        return

    instrument = provider_fn()
    bar_type = BarType(
        instrument_id=instrument.id,
        bar_spec=BarSpecification(
            step=cfg.BAR_STEP,
            aggregation=BarAggregation[cfg.BAR_AGGREGATION],
            price_type=PriceType[cfg.BAR_PRICE_TYPE],
        ),
        aggregation_source=AggregationSource.EXTERNAL,
    )

    klines = await fetch_klines_range(
        market_api=market_api,
        symbol=symbol,
        start_ms=start_ms,
        end_ms=end_ms,
    )

    if not klines:
        logger.warning("No klines fetched for %s — skipping.", symbol)
        return

    bars = klines_to_bars(
        klines=klines,
        bar_type=bar_type,
        price_prec=instrument.price_precision,
        size_prec=instrument.size_precision,
    )
    bars.sort(key=lambda b: b.ts_init)

    catalog.write_data([instrument])
    catalog.write_data(bars)
    logger.info(
        "%s: wrote %d bars to catalog  (%s → %s)",
        symbol,
        len(bars),
        pd.Timestamp(klines[0]["open_time"],  unit="ms", tz="UTC").strftime("%Y-%m-%d"),
        pd.Timestamp(klines[-1]["open_time"], unit="ms", tz="UTC").strftime("%Y-%m-%d"),
    )


# ── Fear & Greed Index fetch ──────────────────────────────────────────────────

async def fetch_fear_and_greed(start: str, end: str) -> None:
    """
    Download the full Fear & Greed Index history from Alternative.me
    (https://api.alternative.me/fng/), filter to [start, end), and save the
    result as a Parquet file at cfg.FEAR_GREED_PATH.

    The API is free and requires no authentication.  It returns one record per
    calendar day, each containing:

      date            — UTC midnight timestamp (tz-aware pd.Timestamp)
      timestamp_ns    — same moment in nanoseconds (int64, for NautilusTrader)
      value           — Fear & Greed score 0–100 (int)
      classification  — e.g. "Extreme Fear", "Fear", "Neutral",
                        "Greed", "Extreme Greed"

    Downstream consumers (train_models, backtest, live_trading) should load
    this file with pd.read_parquet(cfg.FEAR_GREED_PATH) and merge on the UTC
    date of each bar to obtain the day's F&G value.
    """
    logger.info("Fetching Fear & Greed Index from %s …", cfg.FEAR_GREED_URL)

    # limit=0 requests the maximum available history (since Feb 2018).
    # date_format="" keeps the timestamp field as a plain Unix integer string,
    # which we parse ourselves for full control.
    params = {"limit": 0, "format": "json", "date_format": ""}
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(cfg.FEAR_GREED_URL, params=params) as resp:
            resp.raise_for_status()
            # content_type=None because the API sometimes returns text/html
            # even for valid JSON payloads.
            payload = await resp.json(content_type=None)

    records = payload.get("data", [])
    if not records:
        raise RuntimeError(
            "Fear & Greed API returned an empty 'data' list. "
            "The endpoint may be temporarily unavailable."
        )

    # Build a tidy DataFrame from the raw records.
    df = pd.DataFrame(records)

    # Keep only the columns we care about; rename for clarity.
    df = df[["timestamp", "value", "value_classification"]].copy()
    df["timestamp"]  = df["timestamp"].astype(int)
    df["value"]      = df["value"].astype(int)

    # Derive tz-aware UTC date and nanosecond timestamp for NautilusTrader.
    df["date"]         = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.normalize()
    df["timestamp_ns"] = df["timestamp"] * 1_000_000_000

    df = (
        df[["date", "timestamp_ns", "value", "value_classification"]]
        .rename(columns={"value_classification": "classification"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Filter to the requested date range.
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts   = pd.Timestamp(end,   tz="UTC")
    df = df[(df["date"] >= start_ts) & (df["date"] < end_ts)].reset_index(drop=True)

    if df.empty:
        logger.warning(
            "Fear & Greed: no records found between %s and %s after filtering. "
            "The index only goes back to February 2018.",
            start, end,
        )
    else:
        logger.info(
            "Fear & Greed: %d daily records  (%s → %s)",
            len(df),
            df["date"].iloc[0].strftime("%Y-%m-%d"),
            df["date"].iloc[-1].strftime("%Y-%m-%d"),
        )

    df.to_parquet(cfg.FEAR_GREED_PATH, index=False)
    logger.info("Fear & Greed Index saved → %s", cfg.FEAR_GREED_PATH)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main(start: str, end: str, reset_catalog: bool = False) -> None:

    # ── Optional reset ────────────────────────────────────────────────────────
    if reset_catalog:
        if cfg.CATALOG_PATH.exists():
            shutil.rmtree(cfg.CATALOG_PATH)
            logger.info("Cleared existing catalog at %s", cfg.CATALOG_PATH)
        if cfg.FEAR_GREED_PATH.exists():
            cfg.FEAR_GREED_PATH.unlink()
            logger.info("Cleared existing Fear & Greed data at %s", cfg.FEAR_GREED_PATH)

    cfg.CATALOG_PATH.mkdir(parents=True, exist_ok=True)
    catalog = ParquetDataCatalog(cfg.CATALOG_PATH)

    clock       = LiveClock()
    http_client = BinanceHttpClient(
    clock=clock,
    api_key="",
    api_secret="",
    base_url="https://api.binance.com",
)
    market_api  = BinanceSpotMarketHttpAPI(
        client=http_client,
        account_type=BinanceAccountType.SPOT,
    )

    start_ms = dt_to_ms(start)
    end_ms   = dt_to_ms(end)

    # ── OHLCV bars (one instrument at a time) ─────────────────────────────────
    symbols = [i["binance_symbol"] for i in cfg.INSTRUMENTS]
    logger.info("Fetching %d instrument(s): %s", len(symbols), symbols)

    for inst_cfg in cfg.INSTRUMENTS:
        logger.info("─" * 60)
        logger.info("Starting fetch for %s", inst_cfg["binance_symbol"])
        await fetch_instrument(
            market_api=market_api,
            catalog=catalog,
            inst_cfg=inst_cfg,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        await asyncio.sleep(1.0)   # brief pause between instruments

    await asyncio.sleep(1.0)

    # ── Fear & Greed Index ────────────────────────────────────────────────────
    logger.info("─" * 60)
    await fetch_fear_and_greed(start=start, end=end)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(
        "Done. Catalog at: %s  |  instruments: %s", cfg.CATALOG_PATH, symbols
    )
    logger.info("Fear & Greed Index at: %s", cfg.FEAR_GREED_PATH)
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Binance historical bars and the Fear & Greed Index "
            "for all configured instruments."
        )
    )
    parser.add_argument("--start", default=cfg.FETCH_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default=cfg.FETCH_END,   help="End date YYYY-MM-DD")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing catalog and Fear & Greed file before fetching",
    )
    args = parser.parse_args()

    asyncio.run(main(start=args.start, end=args.end, reset_catalog=args.reset))
"""
fetch_data.py
Download historical Binance OHLCV bars for ALL instruments defined in
config.TRADING_INSTRUMENTS and persist them to the local ParquetDataCatalog.

Usage
-----
    python fetch_data.py
    python fetch_data.py --start 2023-10-01 --end 2024-04-30
    python fetch_data.py --start 2023-10-01 --end 2024-04-30 --reset
    python fetch_data.py --refresh-instruments

Changelog vs. previous version
--------------------------------
- FIXED: is_testnet=False → environment=BinanceEnvironment.LIVE on the
  get_cached_binance_http_client() call for the klines HTTP client.
  Same root cause as binance_instruments.py — the nightly build removed
  is_testnet in favour of the BinanceEnvironment enum.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nautilus_trader.adapters.binance import get_cached_binance_http_client
from nautilus_trader.adapters.binance.common.enums import (
    BinanceAccountType,
    BinanceEnvironment,
    BinanceKlineInterval,
)
from nautilus_trader.adapters.binance.spot.http.market import BinanceSpotMarketHttpAPI
from nautilus_trader.common.component import LiveClock
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import AggregationSource, BarAggregation, PriceType
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog

import config as cfg
from utils.binance_instruments import (
    load_binance_instruments_async,
    save_instrument_cache,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("fetch_data")

KLINES_PER_REQUEST = 1000

# ── Bar interval map ──────────────────────────────────────────────────────────
_KLINE_INTERVAL_MAP: dict[tuple[int, str], BinanceKlineInterval] = {
    (1,  "MINUTE"): BinanceKlineInterval.MINUTE_1,
    (3,  "MINUTE"): BinanceKlineInterval.MINUTE_3,
    (5,  "MINUTE"): BinanceKlineInterval.MINUTE_5,
    (15, "MINUTE"): BinanceKlineInterval.MINUTE_15,
    (30, "MINUTE"): BinanceKlineInterval.MINUTE_30,
    (60, "MINUTE"): BinanceKlineInterval.HOUR_1,
}
_BAR_MS_MAP: dict[tuple[int, str], int] = {
    (1,  "MINUTE"):    60_000,
    (3,  "MINUTE"):   180_000,
    (5,  "MINUTE"):   300_000,
    (15, "MINUTE"):   900_000,
    (30, "MINUTE"): 1_800_000,
    (60, "MINUTE"): 3_600_000,
}

_interval_key = (cfg.BAR_STEP, cfg.BAR_AGGREGATION)
if _interval_key not in _KLINE_INTERVAL_MAP:
    raise ValueError(
        f"Unsupported BAR_STEP / BAR_AGGREGATION in config: {_interval_key}. "
        f"Supported: {list(_KLINE_INTERVAL_MAP.keys())}"
    )

KLINE_INTERVAL: BinanceKlineInterval = _KLINE_INTERVAL_MAP[_interval_key]
BAR_DURATION_MS: int                 = _BAR_MS_MAP[_interval_key]


# ── Helpers ───────────────────────────────────────────────────────────────────

def dt_to_ms(dt_str: str) -> int:
    return int(
        datetime.strptime(dt_str, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp() * 1000
    )


def ms_to_ns(ms: int) -> int:
    return ms * 1_000_000


# ── Kline pagination ──────────────────────────────────────────────────────────

async def fetch_klines_range(
    market_api: BinanceSpotMarketHttpAPI,
    symbol:     str,
    start_ms:   int,
    end_ms:     int,
) -> list[dict]:
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
            logger.warning(
                "Empty response for %s %d – %d", symbol, current, batch_end
            )
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
        await asyncio.sleep(0.25)

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


# ── Per-instrument fetch ──────────────────────────────────────────────────────

async def fetch_instrument(
    market_api:  BinanceSpotMarketHttpAPI,
    catalog:     ParquetDataCatalog,
    inst_cfg:    dict,
    start_ms:    int,
    end_ms:      int,
    symbol_map:  dict[str, Instrument],
) -> None:
    symbol     = inst_cfg["binance_symbol"]
    instrument = symbol_map.get(symbol)

    if instrument is None:
        sample = sorted(symbol_map.keys())[:12]
        logger.error(
            "[%s] Not found in Binance Spot exchange info.  "
            "Verify the symbol spelling in config.TRADING_INSTRUMENTS.  "
            "Sample valid symbols: %s",
            symbol, sample,
        )
        return

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
        "%s: wrote %d bars  |  price_precision=%d  size_precision=%d  (%s → %s)",
        symbol,
        len(bars),
        instrument.price_precision,
        instrument.size_precision,
        pd.Timestamp(klines[0]["open_time"],  unit="ms", tz="UTC").strftime("%Y-%m-%d"),
        pd.Timestamp(klines[-1]["open_time"], unit="ms", tz="UTC").strftime("%Y-%m-%d"),
    )


# ── Entry point ───────────────────────────────────────────────────────────────

async def main(
    start:               str,
    end:                 str,
    reset_catalog:       bool = False,
    refresh_instruments: bool = False,
) -> None:

    if reset_catalog:
        if cfg.CATALOG_PATH.exists():
            shutil.rmtree(cfg.CATALOG_PATH)
            logger.info("Cleared existing catalog at %s", cfg.CATALOG_PATH)

    cfg.CATALOG_PATH.mkdir(parents=True, exist_ok=True)
    catalog = ParquetDataCatalog(cfg.CATALOG_PATH)

    if refresh_instruments and cfg.INSTRUMENT_CACHE_PATH.exists():
        cfg.INSTRUMENT_CACHE_PATH.unlink()
        logger.info("Instrument disk cache cleared — will re-fetch from Binance.")

    # Load ALL Binance Spot instruments ONCE (no API key needed)
    symbol_map = await load_binance_instruments_async()

    if not symbol_map:
        logger.error(
            "No instruments could be loaded from Binance.  "
            "Check network connectivity and try again."
        )
        return

    symbols   = [i["binance_symbol"] for i in cfg.INSTRUMENTS]
    resolved  = {sym: symbol_map[sym] for sym in symbols if sym in symbol_map}
    missing_s = [sym for sym in symbols if sym not in symbol_map]

    if missing_s:
        logger.error(
            "Symbols from config NOT found on Binance Spot (check spelling): %s.  "
            "Sample valid symbols: %s",
            missing_s, sorted(symbol_map.keys())[:12],
        )

    if resolved:
        save_instrument_cache(cfg.INSTRUMENT_CACHE_PATH, resolved)
        logger.info(
            "Instrument disk cache updated → %s  (%d symbols)",
            cfg.INSTRUMENT_CACHE_PATH, len(resolved),
        )

    # ── HTTP client for kline requests ────────────────────────────────────────
    # environment=BinanceEnvironment.LIVE replaces the removed is_testnet=False
    clock       = LiveClock()
    klines_http = get_cached_binance_http_client(
        clock=clock,
        account_type=BinanceAccountType.SPOT,
        api_key=None,
        api_secret=None,
        environment=BinanceEnvironment.LIVE,   # ← FIXED (was is_testnet=False)
    )
    market_api = BinanceSpotMarketHttpAPI(
        client=klines_http,
        account_type=BinanceAccountType.SPOT,
    )

    start_ms = dt_to_ms(start)
    end_ms   = dt_to_ms(end)

    logger.info(
        "Fetching %d instrument(s): %s  |  %s → %s",
        len(resolved), list(resolved.keys()), start, end,
    )

    for inst_cfg in cfg.INSTRUMENTS:
        sym = inst_cfg["binance_symbol"]
        if sym not in resolved:
            continue
        logger.info("─" * 60)
        logger.info("Fetching bars for  %s", sym)
        await fetch_instrument(
            market_api=market_api,
            catalog=catalog,
            inst_cfg=inst_cfg,
            start_ms=start_ms,
            end_ms=end_ms,
            symbol_map=symbol_map,
        )
        await asyncio.sleep(1.0)

    logger.info("=" * 60)
    logger.info(
        "Done.  Catalog: %s  |  instruments fetched: %s",
        cfg.CATALOG_PATH, list(resolved.keys()),
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch Binance historical bars for all configured instruments."
    )
    parser.add_argument("--start",  default=cfg.FETCH_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",    default=cfg.FETCH_END,   help="End date YYYY-MM-DD")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the existing catalog before fetching",
    )
    parser.add_argument(
        "--refresh-instruments",
        action="store_true",
        help="Force re-fetch of instrument specs from Binance, ignoring disk cache",
    )
    args = parser.parse_args()

    asyncio.run(
        main(
            start=args.start,
            end=args.end,
            reset_catalog=args.reset,
            refresh_instruments=args.refresh_instruments,
        )
    )
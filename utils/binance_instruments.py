"""
utils/binance_instruments.py

Fetches live Binance Spot instrument definitions via NautilusTrader's
BinanceSpotInstrumentProvider (public exchange-info endpoint, no API key).

Changelog vs. previous version
--------------------------------
- FIXED: is_testnet=False → environment=BinanceEnvironment.LIVE
  The installed NautilusTrader build (nightly API) replaced the is_testnet
  bool with environment=BinanceEnvironment enum on get_cached_binance_http_client.
"""

from __future__ import annotations

import asyncio
import logging
import pickle
from pathlib import Path
from typing import Optional

from nautilus_trader.adapters.binance import get_cached_binance_http_client
from nautilus_trader.adapters.binance.common.enums import BinanceAccountType, BinanceEnvironment
from nautilus_trader.adapters.binance.spot.providers import BinanceSpotInstrumentProvider
from nautilus_trader.common.component import LiveClock
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.model.instruments import Instrument

logger = logging.getLogger(__name__)

# Module-level in-memory cache — populated on first call to get_instruments_sync().
_MEM_CACHE: dict[str, Instrument] = {}


# ── Core async loader ─────────────────────────────────────────────────────────

async def load_binance_instruments_async() -> dict[str, Instrument]:
    """
    Fetch ALL Binance Spot instrument definitions from the public exchange-info
    endpoint and return them as a plain {symbol_str: Instrument} dict.

    No API key is required — the exchange-info endpoint is fully public.
    """
    clock = LiveClock()

    # ── Step 1: HTTP client via factory ──────────────────────────────────────
    # environment=BinanceEnvironment.LIVE replaces the removed is_testnet=False
    # parameter in the nightly NautilusTrader build.
    # api_key=None activates the public-endpoint (no-auth) path.
    client = get_cached_binance_http_client(
        clock=clock,
        account_type=BinanceAccountType.SPOT,
        api_key=None,
        api_secret=None,
        environment=BinanceEnvironment.LIVE,
    )

    # ── Step 2: Provider with explicit account_type ───────────────────────────
    try:
        provider = BinanceSpotInstrumentProvider(
            client=client,
            clock=clock,
            account_type=BinanceAccountType.SPOT,
            config=InstrumentProviderConfig(load_all=True, log_warnings=False),
        )
    except TypeError:
        # Older builds that don't accept clock / config as keyword args
        provider = BinanceSpotInstrumentProvider(
            client=client,
            account_type=BinanceAccountType.SPOT,
        )

    # ── Step 3: Load ──────────────────────────────────────────────────────────
    logger.info("Fetching Binance Spot exchange info (public endpoint)…")
    await provider.load_all_async()

    # ── Step 4: Build symbol map via list_all() (NOT get_all()) ──────────────
    instruments: list[Instrument] = provider.list_all()
    count = len(instruments)

    if count == 0:
        logger.error(
            "BinanceSpotInstrumentProvider loaded 0 instruments after load_all_async().  "
            "Check: (1) network connectivity to api.binance.com, "
            "(2) NautilusTrader version compatibility."
        )
        return {}

    logger.info("Binance Spot instrument definitions loaded (%d symbols).", count)

    symbol_map: dict[str, Instrument] = {}
    for inst in instruments:
        try:
            sym: str = inst.id.symbol.value       # e.g. "BTCUSDT"
        except AttributeError:
            sym = str(inst.id).split(".")[0]       # fallback: "BTCUSDT.BINANCE" → "BTCUSDT"
        symbol_map[sym] = inst

    logger.debug("build_symbol_map: indexed %d symbols.", len(symbol_map))
    return symbol_map


# ── Backward-compatible provider wrapper ─────────────────────────────────────

async def load_binance_provider_async() -> BinanceSpotInstrumentProvider:
    """Legacy entry point — prefer load_binance_instruments_async() in new code."""
    clock = LiveClock()
    client = get_cached_binance_http_client(
        clock=clock,
        account_type=BinanceAccountType.SPOT,
        api_key=None,
        api_secret=None,
        environment=BinanceEnvironment.LIVE,
    )
    try:
        provider = BinanceSpotInstrumentProvider(
            client=client,
            clock=clock,
            account_type=BinanceAccountType.SPOT,
            config=InstrumentProviderConfig(load_all=True, log_warnings=False),
        )
    except TypeError:
        provider = BinanceSpotInstrumentProvider(
            client=client,
            account_type=BinanceAccountType.SPOT,
        )

    logger.info("Fetching Binance Spot exchange info (public endpoint)…")
    await provider.load_all_async()

    count = len(provider.list_all())
    if count == 0:
        logger.error(
            "BinanceSpotInstrumentProvider loaded 0 instruments.  "
            "Check network connectivity — the Binance exchange-info endpoint "
            "is public and requires no authentication."
        )
    else:
        logger.info("Binance Spot instrument definitions loaded (%d symbols).", count)

    return provider


def build_symbol_map(
    provider: BinanceSpotInstrumentProvider,
) -> dict[str, Instrument]:
    """Build a plain {symbol_str: Instrument} dict from a loaded provider."""
    result: dict[str, Instrument] = {}
    for inst in provider.list_all():
        try:
            sym: str = inst.id.symbol.value
        except AttributeError:
            sym = str(inst.id).split(".")[0]
        result[sym] = inst
    logger.debug("build_symbol_map: indexed %d symbols.", len(result))
    return result


# ── Disk cache helpers ────────────────────────────────────────────────────────

def _load_disk_cache(cache_path: Path) -> dict[str, Instrument]:
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, "rb") as fh:
            data: dict[str, Instrument] = pickle.load(fh)
        logger.debug(
            "Loaded %d instruments from disk cache ← %s", len(data), cache_path
        )
        return data
    except Exception as exc:
        logger.warning(
            "Instrument disk cache unreadable (%s) — will re-fetch.", exc
        )
        return {}


def save_instrument_cache(
    cache_path: Path,
    instruments: dict[str, Instrument],
) -> None:
    """Persist the {symbol: Instrument} map to a pickle file."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump(instruments, fh)
        logger.debug(
            "Saved %d instruments to disk cache → %s",
            len(instruments), cache_path,
        )
    except Exception as exc:
        logger.warning("Could not save instrument disk cache: %s", exc)


_save_disk_cache = save_instrument_cache  # backward-compatible alias


# ── Synchronous public entry point ────────────────────────────────────────────

def get_instruments_sync(
    symbols:       list[str],
    venue:         str = "BINANCE",
    cache_path:    Optional[Path] = None,
    force_refresh: bool = False,
) -> dict[str, Instrument]:
    """
    Return NautilusTrader Instrument objects for the requested Binance Spot
    symbols.  Safe to call from synchronous scripts.

    Lookup order: in-memory cache → disk cache → live Binance API.
    """
    global _MEM_CACHE

    if force_refresh:
        _MEM_CACHE.clear()

    if cache_path:
        for s, inst in _load_disk_cache(cache_path).items():
            if s not in _MEM_CACHE:
                _MEM_CACHE[s] = inst

    missing = [s for s in symbols if s not in _MEM_CACHE]

    if missing:
        logger.info(
            "Fetching instrument definitions from Binance Spot (needed: %s)", missing,
        )
        try:
            all_instruments = asyncio.run(load_binance_instruments_async())
        except Exception as exc:
            logger.error(
                "Failed to fetch instruments from Binance: %s  "
                "(check internet connection or symbol spelling).", exc,
            )
            all_instruments = {}

        _MEM_CACHE.update(all_instruments)
        if cache_path and all_instruments:
            disk = _load_disk_cache(cache_path)
            disk.update(all_instruments)
            save_instrument_cache(cache_path, disk)

    found         = {s: _MEM_CACHE[s] for s in symbols if s in _MEM_CACHE}
    still_missing = [s for s in symbols if s not in found]
    if still_missing:
        logger.error(
            "The following symbols could NOT be resolved and will be skipped: %s",
            still_missing,
        )
    return found
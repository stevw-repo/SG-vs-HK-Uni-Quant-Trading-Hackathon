"""
backtest.py
Run NautilusTrader BacktestEngine for all instruments in config.INSTRUMENTS.

Instruments are resolved via utils/binance_instruments.get_instruments_sync(),
which fetches live Binance Spot specs (public endpoint, no key) and caches them
to cfg.INSTRUMENT_CACHE_PATH.  Any symbol in config.TRADING_INSTRUMENTS is
supported — no TestInstrumentProvider required.

Usage
-----
    python backtest.py
"""

import logging
import time

import numpy as np
import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, BookType, OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Currency, Money
from nautilus_trader.persistence.catalog import ParquetDataCatalog

try:
    from nautilus_trader.analysis import create_tearsheet, TearsheetConfig
    from nautilus_trader.analysis import (
        TearsheetRunInfoChart,
        TearsheetStatsTableChart,
        TearsheetEquityChart,
        TearsheetDrawdownChart,
    )
    from nautilus_trader.analysis.tearsheet import create_bars_with_fills
    _TEARSHEET_AVAILABLE = True
except ImportError as _e:
    _TEARSHEET_AVAILABLE = False
    _TEARSHEET_IMPORT_ERROR = str(_e)

import config as cfg
from strategy.hmm_strategy import HMMStrategy, HMMStrategyConfig
from utils.binance_instruments import get_instruments_sync
from utils.capital_allocator import CapitalAllocator
from utils.diagnostics import DiagnosticEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("backtest")


def _parse_pnl(raw) -> float:
    if raw is None:
        return 0.0
    try:
        return float(str(raw).split()[0].replace(",", ""))
    except (ValueError, IndexError, AttributeError):
        return 0.0


def _build_equity_curve(
    pos_df:      pd.DataFrame,
    balance_usd: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    trades   = [{"pnl": _parse_pnl(row.get("realized_pnl"))} for _, row in pos_df.iterrows()]
    trade_df = pd.DataFrame(trades)

    if trade_df["pnl"].abs().sum() == 0:
        logger.warning(
            "All extracted PnL values are zero.  Available columns: %s.  "
            "Ensure 'realized_pnl' is present and positions were closed.",
            list(pos_df.columns),
        )
    equity = np.concatenate(
        [[balance_usd], balance_usd + trade_df["pnl"].cumsum().values]
    )
    return equity, trade_df


def run_backtest(
    start:       str   = cfg.BACKTEST_START,
    end:         str   = cfg.BACKTEST_END,
    balance_usd: float = cfg.STARTING_BALANCE,
) -> None:

    # ── Resolve instrument definitions ────────────────────────────────────────
    # get_instruments_sync() checks the in-memory cache, then the disk cache
    # at INSTRUMENT_CACHE_PATH, and finally the live Binance API (~5 s) only
    # if an instrument is not already cached.  After the first run per machine
    # the disk cache makes this effectively instant.
    #
    # Any Binance Spot symbol in config.TRADING_INSTRUMENTS is supported here.
    all_symbols = [i["binance_symbol"] for i in cfg.INSTRUMENTS]

    logger.info(
        "Resolving instrument definitions for: %s  "
        "(disk cache: %s)",
        all_symbols,
        cfg.INSTRUMENT_CACHE_PATH,
    )
    binance_instruments = get_instruments_sync(
        symbols=all_symbols,
        venue=cfg.VENUE,
        cache_path=cfg.INSTRUMENT_CACHE_PATH,
    )

    if not binance_instruments:
        logger.error(
            "Could not resolve any instruments.  "
            "Run fetch_data.py first (it populates the instrument cache), "
            "or check your internet connection.",
        )
        return

    # ── Engine setup ──────────────────────────────────────────────────────────
    engine = BacktestEngine(config=BacktestEngineConfig(
        logging=LoggingConfig(log_level="WARNING"),
    ))

    venue = Venue("BINANCE")
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=None,
        starting_balances=[Money(balance_usd, Currency.from_str("USDT"))],
        book_type=BookType.L1_MBP,
        bar_execution=True,
    )

    catalog  = ParquetDataCatalog(cfg.CATALOG_PATH)
    start_ns = dt_to_unix_nanos(pd.Timestamp(start, tz="UTC"))
    end_ns   = dt_to_unix_nanos(pd.Timestamp(end,   tz="UTC"))

    allocator = CapitalAllocator()
    active: list[dict] = []

    for inst_cfg in cfg.INSTRUMENTS:
        symbol   = inst_cfg["binance_symbol"]
        hmm_path = inst_cfg["hmm_model_path"]

        # ── Skip if model missing ─────────────────────────────────────────────
        if not hmm_path.exists():
            logger.warning(
                "[%s] HMM model not found at %s — skipping.  "
                "Run train_models.py first.",
                symbol, hmm_path,
            )
            continue

        # ── Look up instrument from the resolved Binance definitions ──────────
        instrument = binance_instruments.get(symbol)
        if instrument is None:
            logger.warning(
                "[%s] Instrument not resolved (check symbol spelling in "
                "config.TRADING_INSTRUMENTS) — skipping.",
                symbol,
            )
            continue

        logger.info(
            "[%s] Using live Binance instrument spec: "
            "price_precision=%d  size_precision=%d",
            symbol,
            instrument.price_precision,
            instrument.size_precision,
        )
        engine.add_instrument(instrument)

        # ── Load bars ─────────────────────────────────────────────────────────
        bar_type = BarType.from_str(inst_cfg["bar_type_str"])
        bars     = catalog.bars(bar_types=[str(bar_type)], start=start_ns, end=end_ns)
        bars.sort(key=lambda b: b.ts_init)

        if not bars:
            logger.error(
                "[%s] No bars found in catalog for %s → %s.  "
                "Run fetch_data.py first.",
                symbol, start, end,
            )
            continue

        logger.info("Loaded %d bars for %s (%s → %s)", len(bars), symbol, start, end)
        engine.add_data(bars)

        # ── Strategy ──────────────────────────────────────────────────────────
        strategy_cfg = HMMStrategyConfig(
            instrument_id           = inst_cfg["instrument_id_str"],
            bar_type                = inst_cfg["bar_type_str"],
            hmm_model_path          = str(hmm_path),
            min_bull_proba          = cfg.MIN_BULL_PROBA,
            min_kelly_fraction      = cfg.MIN_KELLY_FRACTION,
            trend_ema_bars          = cfg.TREND_EMA_BARS,
            kelly_fraction          = cfg.KELLY_FRACTION,
            max_position_pct        = cfg.MAX_POSITION_PCT,
            commission_rate         = cfg.COMMISSION_RATE,
            take_profit_pct         = cfg.TAKE_PROFIT_PCT,
            trail_bull_pct          = cfg.TRAIL_BULL_PCT,
            trail_sideways_pct      = cfg.TRAIL_SIDEWAYS_PCT,
            trail_bear_pct          = cfg.TRAIL_BEAR_PCT,
            bear_exit_proba         = cfg.BEAR_EXIT_PROBA,
            bear_exit_consecutive   = cfg.BEAR_EXIT_CONSECUTIVE,
            min_bars_between_trades = cfg.MIN_BARS_BETWEEN_TRADES,
            hmm_min_history         = cfg.HMM_MIN_HISTORY,
            starting_balance_usd    = balance_usd,
            trend_lookback_bars     = cfg.TREND_LOOKBACK_BARS,
            bull_entry_consecutive  = cfg.BULL_ENTRY_CONSECUTIVE,
            max_holding_bars        = cfg.MAX_HOLDING_BARS,
        )
        engine.add_strategy(HMMStrategy(config=strategy_cfg, allocator=allocator))
        active.append(inst_cfg)
        logger.info("Strategy registered for %s", symbol)

    if not active:
        logger.error("No instruments loaded.  Aborting.")
        return

    syms = [i["binance_symbol"] for i in active]
    logger.info("Starting backtest | instruments=%s | %s → %s", syms, start, end)
    t0 = time.perf_counter()
    engine.run()
    logger.info("Backtest complete in %.1f s", time.perf_counter() - t0)

    portfolio = engine.portfolio

    # ── Tearsheet ─────────────────────────────────────────────────────────────
    if _TEARSHEET_AVAILABLE:
        tearsheet_path = str(cfg.RESULTS_DIR / "tearsheet.html")
        try:
            create_tearsheet(
                engine=engine,
                output_path=tearsheet_path,
                config=TearsheetConfig(
                    title=f"HMM Regime Strategy — {', '.join(syms)}",
                    theme="nautilus",
                    charts=[
                        TearsheetRunInfoChart(),
                        TearsheetStatsTableChart(),
                        TearsheetEquityChart(),
                        TearsheetDrawdownChart(),
                    ],
                ),
            )
            logger.info("Tearsheet → %s", tearsheet_path)
        except Exception as exc:
            logger.warning("Tearsheet failed: %s", exc)

        for inst_cfg in active:
            try:
                bar_type = BarType.from_str(inst_cfg["bar_type_str"])
                fig = create_bars_with_fills(
                    engine=engine,
                    bar_type=bar_type,
                    title=f"{inst_cfg['binance_symbol']} — Bars with Fills",
                )
                out = cfg.RESULTS_DIR / f"bars_with_fills_{inst_cfg['binance_symbol']}.html"
                fig.write_html(str(out))
            except Exception as exc:
                logger.warning("[%s] bars_with_fills failed: %s", inst_cfg["binance_symbol"], exc)
    else:
        logger.warning(
            "Tearsheet skipped — import failed: %s  "
            "(pip install plotly nautilus_trader[analysis])",
            _TEARSHEET_IMPORT_ERROR,
        )

    # ── Portfolio analyzer ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PORTFOLIO ANALYZER — BUILT-IN STATISTICS")
    print("=" * 70)

    try:
        _ccy   = Currency.from_str("USDT")
        _stats = {
            "PnL Statistics":     portfolio.analyzer.get_performance_stats_pnls(currency=_ccy),
            "Return Statistics":  portfolio.analyzer.get_performance_stats_returns(),
            "General Statistics": portfolio.analyzer.get_performance_stats_general(),
        }
        for section, stats in _stats.items():
            print(f"\n  {section}:")
            for k, v in (stats or {}).items():
                print(f"    {k:<38} {v}")
    except Exception as exc:
        logger.warning("PortfolioAnalyzer unavailable: %s", exc)

    # ── Standard reports ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ACCOUNT / FILLS / POSITIONS REPORTS")
    print("=" * 70)

    fills_df = None
    pos_df   = None

    try:
        with pd.option_context("display.max_columns", None, "display.width", 130):
            print("\n  ACCOUNT")
            print(engine.trader.generate_account_report(venue))

            print("\n  ORDER FILLS")
            fills_df = engine.trader.generate_order_fills_report()
            print(fills_df.to_string(index=False) if fills_df is not None else "No fills.")

            print("\n  POSITIONS")
            pos_df = engine.trader.generate_positions_report()
            print(pos_df.to_string(index=False) if pos_df is not None else "No positions.")
    except Exception as exc:
        logger.warning("Report generation failed: %s", exc)

    # ── DiagnosticEngine metrics ───────────────────────────────────────────────
    if pos_df is not None and len(pos_df) > 0:
        diag           = DiagnosticEngine(cfg.RESULTS_DIR)
        equity, t_df   = _build_equity_curve(pos_df, balance_usd)
        metrics        = diag.compute_strategy_metrics(equity_curve=equity, trades=t_df)

        print(f"\n  DIAGNOSTIC METRICS  ({', '.join(syms)} — combined)")
        for k, v in metrics.items():
            print(f"    {k:<28} {v:+.4f}")
    elif fills_df is not None and len(fills_df) > 0:
        logger.warning(
            "Fills found but no closed positions.  Extend BACKTEST_END "
            "or verify the strategy closes positions before data ends."
        )
    else:
        logger.warning("No trades generated for %s → %s.", start, end)

    engine.reset()
    engine.dispose()


if __name__ == "__main__":
    run_backtest()
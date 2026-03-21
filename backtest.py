"""
backtest.py
Run a NautilusTrader BacktestEngine simulation for ALL instruments defined in
config.TRADING_INSTRUMENTS.

One HMMLSTMStrategy is instantiated per instrument, each backed by its own
pre-trained HMM and LSTM models.  All strategies share the same BINANCE venue
and USD starting balance.

To add or remove an asset, edit only config.TRADING_INSTRUMENTS — no changes
are needed in this file.

Usage
-----
    python backtest.py
"""

import logging
import time

import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, BookType, OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model import Currency

import config as cfg
from strategy.hmm_lstm_strategy import HMMLSTMStrategy, HMMLSTMStrategyConfig
from utils.capital_allocator import CapitalAllocator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("backtest")


def run_backtest(
    start:       str   = cfg.BACKTEST_START,
    end:         str   = cfg.BACKTEST_END,
    balance_usd: float = cfg.STARTING_BALANCE,
) -> None:

    # ── Engine ────────────────────────────────────────────────────────────────
    engine = BacktestEngine(config=BacktestEngineConfig(
        logging=LoggingConfig(log_level="WARNING"),
    ))

    # ── Venue ─────────────────────────────────────────────────────────────────
    venue = Venue("BINANCE")
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=None,                 # multi-currency
        starting_balances=[Money(1_000_000.0, Currency.from_str("USDT"))],
        book_type=BookType.L1_MBP,
        bar_execution=True,
    )

    # ── Catalog ───────────────────────────────────────────────────────────────
    catalog  = ParquetDataCatalog(cfg.CATALOG_PATH)
    start_ns = dt_to_unix_nanos(pd.Timestamp(start, tz="UTC"))
    end_ns   = dt_to_unix_nanos(pd.Timestamp(end,   tz="UTC"))

    active: list[dict] = []

    # ── Shared capital allocator ──────────────────────────────────────────────
    # One allocator is shared across all strategy instances so that when two
    # strategies fire entry signals on the same bar, the second one sees the
    # first one's reservation already deducted from the available balance.
    allocator = CapitalAllocator()

    # ── Loop over every configured instrument ─────────────────────────────────
    for inst_cfg in cfg.INSTRUMENTS:
        symbol = inst_cfg["binance_symbol"]

        # Instrument definition
        provider_fn = cfg.INSTRUMENT_PROVIDERS.get(symbol)
        if provider_fn is None:
            logger.warning(
                "No INSTRUMENT_PROVIDERS entry for '%s' — skipping. "
                "Add one to config.py to include it in the backtest.",
                symbol,
            )
            continue

        instrument = provider_fn()
        engine.add_instrument(instrument)

        # Bar data
        bar_type = BarType.from_str(inst_cfg["bar_type_str"])
        bars = catalog.bars(
            bar_types=[str(bar_type)],
            start=start_ns,
            end=end_ns,
        )
        bars.sort(key=lambda b: b.ts_init)

        if not bars:
            logger.error(
                "No bars found for %s (%s → %s). Run fetch_data.py first.",
                symbol, start, end,
            )
            continue

        logger.info("Loaded %d bars for %s (%s → %s)", len(bars), symbol, start, end)
        engine.add_data(bars)

        # Strategy — fear_greed_path is passed so the strategy loads the daily
        # F&G index and threads it through LSTM inference on every bar.
        strategy_config = HMMLSTMStrategyConfig(
            instrument_id=inst_cfg["instrument_id_str"],
            bar_type=inst_cfg["bar_type_str"],
            hmm_model_path=str(inst_cfg["hmm_model_path"]),
            lstm_model_path=str(inst_cfg["lstm_model_path"]),
            fear_greed_path=str(cfg.FEAR_GREED_PATH),
            min_confidence_buy=cfg.MIN_CONFIDENCE_ENTRY,
            min_confidence_sell=cfg.MIN_CONFIDENCE_EXIT,
            kelly_fraction=cfg.KELLY_FRACTION,
            max_position_pct=cfg.MAX_POSITION_PCT,
            win_loss_ratio=cfg.EXPECTED_WIN_LOSS_RATIO,
            stop_loss_pct=cfg.STOP_LOSS_PCT,
            take_profit_pct=cfg.TAKE_PROFIT_PCT,
            trailing_stop_pct=cfg.TRAILING_STOP_PCT,
            hmm_min_history=cfg.HMM_MIN_HISTORY,
            lstm_lookback=cfg.LSTM_LOOKBACK,
            starting_balance_usd=balance_usd,
        )
        # Pass the shared allocator so all strategies coordinate on the same
        # capital pool rather than each independently claiming the full balance.
        engine.add_strategy(HMMLSTMStrategy(config=strategy_config, allocator=allocator))
        active.append(inst_cfg)
        logger.info("Strategy registered for %s", symbol)

    if not active:
        logger.error("No instruments could be loaded. Aborting backtest.")
        return

    # ── Run ───────────────────────────────────────────────────────────────────
    active_symbols = [i["binance_symbol"] for i in active]
    logger.info(
        "Starting backtest for %d instrument(s): %s",
        len(active), active_symbols,
    )
    t0 = time.perf_counter()
    engine.run()
    logger.info("Backtest completed in %.1f s", time.perf_counter() - t0)

    # ── Reports ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ACCOUNT REPORT")
    print("=" * 70)
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(engine.trader.generate_account_report(venue))

        print("\n  ORDER FILLS")
        fills_df = engine.trader.generate_order_fills_report()
        print(fills_df.to_string(index=False) if fills_df is not None else "No fills.")

        print("\n  POSITIONS")
        pos_df = engine.trader.generate_positions_report()
        print(pos_df.to_string(index=False) if pos_df is not None else "No positions.")

    # ── Aggregate performance metrics ─────────────────────────────────────────
    if fills_df is not None and len(fills_df) > 0:
        import numpy as np
        from utils.diagnostics import DiagnosticEngine

        diag   = DiagnosticEngine(cfg.RESULTS_DIR)
        trades = [
            {"pnl": float(row.get("realized_pnl", 0) or 0)}
            for _, row in fills_df.iterrows()
        ]
        if trades:
            trade_df = pd.DataFrame(trades)
            equity   = balance_usd + trade_df["pnl"].cumsum().values
            metrics  = diag.compute_strategy_metrics(
                equity_curve=np.concatenate([[balance_usd], equity]),
                trades=trade_df,
            )
            print(f"\n  PERFORMANCE METRICS  ({', '.join(active_symbols)} — combined)")
            for k, v in metrics.items():
                print(f"    {k:<28} {v:+.4f}")

    engine.reset()
    engine.dispose()


if __name__ == "__main__":
    run_backtest()
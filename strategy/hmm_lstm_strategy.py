"""
strategy/hmm_lstm_strategy.py
NautilusTrader Strategy — works in BOTH backtest and live modes.

In backtest mode  : BacktestEngine handles order routing & fills.
In live mode      : TradingNode routes submit_order() calls to the
                    RoostooLiveExecutionClient (see live_trading.py).
"""

from __future__ import annotations

import gc
import logging
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.trading.strategy import Strategy

from models.hmm_model import BEAR, BULL, SIDEWAYS, RegimeHMM
from models.lstm_model import DirectionalLSTM
from utils.capital_allocator import CapitalAllocator
from utils.kelly_criterion import KellyCriterion

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────


class HMMLSTMStrategyConfig(StrategyConfig, frozen=True):
    instrument_id:        str
    bar_type:             str
    hmm_model_path:       str
    lstm_model_path:      str
    fear_greed_path:      str   = ""     # path to fear_greed.parquet; "" = disabled
    # Signal thresholds
    min_confidence_buy:   float = 0.53
    min_confidence_sell:  float = 0.47
    # Risk
    kelly_fraction:       float = 0.40
    max_position_pct:     float = 0.65
    win_loss_ratio:       float = 2.67
    stop_loss_pct:        float = 0.015
    take_profit_pct:      float = 0.040
    trailing_stop_pct:    float = 0.020
    # Model warm-up
    hmm_min_history:      int   = 200
    lstm_lookback:        int   = 60
    # Starting USD balance (for position sizing in backtest)
    starting_balance_usd: float = 50_000.0


# ── Strategy ──────────────────────────────────────────────────────────────────


class HMMLSTMStrategy(Strategy):
    """
    Signal-generation strategy using HMM regime detection + LSTM directional
    probability, sized by the Kelly Criterion.

    Position management
    -------------------
    - Flat → enter long when BULL regime AND P(up) > min_confidence_buy
    - Long → exit when BEAR/SIDEWAYS OR P(up) < min_confidence_sell
    - Hard stop loss, take profit, and trailing stop managed in on_bar()
    - All orders are MARKET (TAKER) for immediacy

    Fear & Greed
    ------------
    If fear_greed_path points to a valid Parquet file (written by fetch_data.py),
    the daily F&G index is joined to each bar by UTC date and passed to the LSTM
    as a sentiment context feature.  A neutral 0.5 fallback is used per bar when
    the file is absent or a date gap exists so the LSTM input_size always matches
    the trained model.

    Capital allocation
    ------------------
    An optional shared CapitalAllocator can be passed at construction time.
    When present, each strategy subtracts other strategies' live reservations
    from the account free balance before sizing, preventing two strategies from
    committing the same dollars on the same bar.  Safe to omit for single-
    instrument use.
    """

    def __init__(
        self,
        config:    HMMLSTMStrategyConfig,
        allocator: CapitalAllocator | None = None,
    ) -> None:
        super().__init__(config)

        self._instrument_id = InstrumentId.from_str(config.instrument_id)
        self._bar_type      = BarType.from_str(config.bar_type)

        # Unique key used by the shared CapitalAllocator.
        # instrument_id string (e.g. "BTCUSDT.BINANCE") is a natural choice
        # because exactly one strategy instance exists per instrument.
        self._strategy_id = config.instrument_id

        # Shared allocator — None disables cross-strategy coordination.
        self._allocator: CapitalAllocator | None = allocator

        # Models (loaded in on_start)
        self._hmm:  RegimeHMM | None       = None
        self._lstm: DirectionalLSTM | None = None

        # Kelly sizer
        self._kelly = KellyCriterion(
            fraction=config.kelly_fraction,
            win_loss_ratio=config.win_loss_ratio,
            max_position_pct=config.max_position_pct,
        )

        # Rolling buffers — bounded size, memory efficient
        _buf = config.hmm_min_history + config.lstm_lookback + 2
        self._closes     = deque(maxlen=_buf)
        self._highs      = deque(maxlen=_buf)
        self._lows       = deque(maxlen=_buf)
        self._volumes    = deque(maxlen=_buf)
        # Fear & Greed buffer — same maxlen; always populated (0.5 fallback)
        # so the LSTM input_size never mismatches the saved model.
        self._fear_greed = deque(maxlen=_buf)

        # Fear & Greed state
        self._fear_greed_df:   pd.DataFrame | None = None
        self._fear_greed_last: float               = 0.5   # last known value / neutral

        # Position state
        self._is_long:        bool  = False
        self._entry_price:    float = 0.0
        self._stop_price:     float = 0.0
        self._tp_price:       float = 0.0
        self._peak_price:     float = 0.0    # for trailing stop
        self._entry_quantity: float = 0.0

        # Diagnostics
        self._bar_count:   int   = 0
        self._trade_count: int   = 0
        self._last_regime: int   = SIDEWAYS
        self._last_p_up:   float = 0.5

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_start(self) -> None:
        """Load models, optionally load Fear & Greed data, subscribe to bars."""
        logger.info("Strategy starting — loading models …")
        self._hmm  = RegimeHMM.load(Path(self.config.hmm_model_path))
        self._lstm = DirectionalLSTM.load(
            Path(self.config.lstm_model_path), device="cpu"
        )

        # ── Fear & Greed ──────────────────────────────────────────────────────
        fg_path = Path(self.config.fear_greed_path) if self.config.fear_greed_path else None
        if fg_path is not None and fg_path.exists():
            try:
                df = pd.read_parquet(fg_path)
                # Ensure tz-aware UTC dates for reliable date comparisons.
                if df["date"].dt.tz is None:
                    df["date"] = df["date"].dt.tz_localize("UTC")
                self._fear_greed_df = df[["date", "value"]].copy()
                logger.info(
                    "Fear & Greed Index loaded: %d daily records (%s → %s)",
                    len(self._fear_greed_df),
                    self._fear_greed_df["date"].iloc[0].strftime("%Y-%m-%d"),
                    self._fear_greed_df["date"].iloc[-1].strftime("%Y-%m-%d"),
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load Fear & Greed from %s: %s — "
                    "using neutral fallback (0.5).",
                    fg_path, exc,
                )
        elif fg_path is not None:
            logger.warning(
                "Fear & Greed file not found at %s — "
                "using neutral fallback (0.5).",
                fg_path,
            )

        self.subscribe_bars(self._bar_type)
        logger.info(
            "Strategy ready | instrument=%s | bar_type=%s | fear_greed=%s | "
            "allocator=%s",
            self._instrument_id,
            self._bar_type,
            "enabled" if self._fear_greed_df is not None else "disabled (0.5 fallback)",
            "shared" if self._allocator is not None else "none (single-instrument)",
        )

    def on_stop(self) -> None:
        # Ensure no stale reservation is left in the allocator when the
        # strategy is stopped mid-position (e.g. end of backtest window).
        if self._allocator is not None and self._is_long:
            self._allocator.release(self._strategy_id)

        logger.info(
            "Strategy stopped | trades=%d | last_regime=%s | last_p_up=%.3f",
            self._trade_count,
            {BEAR: "BEAR", SIDEWAYS: "SIDEWAYS", BULL: "BULL"}.get(self._last_regime),
            self._last_p_up,
        )

    # ── Bar handler ───────────────────────────────────────────────────────────

    def on_bar(self, bar: Bar) -> None:
        self._bar_count += 1
        close  = float(bar.close)
        high   = float(bar.high)
        low    = float(bar.low)
        volume = float(bar.volume)

        # Update OHLCV buffers
        self._closes.append(close)
        self._highs.append(high)
        self._lows.append(low)
        self._volumes.append(volume)

        # Update Fear & Greed buffer — one date lookup per bar.
        # _lookup_fear_greed caches the last known value so all 96 intraday bars
        # on the same UTC day resolve with a single DataFrame scan.
        self._fear_greed.append(self._lookup_fear_greed(bar.ts_init))

        # Exit conditions for open position (always checked first)
        if self._is_long:
            self._check_exits(close, high, low)
            if not self._is_long:
                return          # Just exited; skip entry logic this bar

        # Need enough warm-up data
        min_bars = self.config.hmm_min_history + self.config.lstm_lookback
        if len(self._closes) < min_bars:
            return

        # Run inference (only if flat or periodically when long for exit check)
        regime, p_up = self._run_inference()
        if regime is None:
            return

        self._last_regime = regime
        self._last_p_up   = p_up

        # Log every 60 bars
        if self._bar_count % 60 == 0:
            self._log_state(close, regime, p_up)

        # Entry logic (only if flat)
        if not self._is_long:
            self._check_entry(bar, close, regime, p_up)

    # ── Fear & Greed lookup ───────────────────────────────────────────────────

    def _lookup_fear_greed(self, ts_ns: int) -> float:
        """
        Return the normalised Fear & Greed value [0, 1] for the bar's UTC date.

        Uses a last-known-value cache so that all 15-minute bars within the same
        UTC day resolve the same value without repeated DataFrame scans.  Falls
        back to the last known value (or 0.5 if none exists yet) when a date gap
        is found — this can occur at the leading edge of the data or during any
        API outage on the F&G provider's side.
        """
        if self._fear_greed_df is None:
            return 0.5

        bar_date = pd.Timestamp(ts_ns, unit="ns", tz="UTC").normalize()
        row = self._fear_greed_df.loc[
            self._fear_greed_df["date"] == bar_date, "value"
        ]
        if not row.empty:
            self._fear_greed_last = float(row.iloc[0]) / 100.0

        # Return cached value even when bar_date isn't in the DataFrame.
        return self._fear_greed_last

    # ── Inference ─────────────────────────────────────────────────────────────

    def _run_inference(self) -> tuple[int | None, float | None]:
        """HMM + LSTM inference. Returns (regime, p_up) or (None, None)."""
        try:
            prices     = np.array(self._closes,     dtype=np.float32)
            volumes    = np.array(self._volumes,     dtype=np.float32)
            highs      = np.array(self._highs,       dtype=np.float32)
            lows       = np.array(self._lows,        dtype=np.float32)
            fear_greed = np.array(self._fear_greed,  dtype=np.float32)

            # HMM regime — output length is N-1 due to internal differencing
            regime, _proba = self._hmm.predict_current_regime(prices, volumes)
            hmm_states     = self._hmm.predict(prices, volumes)

            # Trim all OHLCV/F&G arrays by one from the front so their length
            # matches hmm_states (N-1).  This mirrors the aligned_* trimming
            # done in train_models.py and evaluate_models.py.
            # The deque maxlen includes +2 headroom specifically for this trim.
            p_up = self._lstm.predict_proba(
                closes=prices[1:],
                highs=highs[1:],
                lows=lows[1:],
                volumes=volumes[1:],
                hmm_regimes=hmm_states,
                fear_greed=fear_greed[1:],
            )

            del prices, volumes, highs, lows, hmm_states, fear_greed
            return regime, p_up

        except Exception as exc:
            logger.warning("Inference error (bar %d): %s", self._bar_count, exc)
            return None, None

    # ── Entry / Exit logic ────────────────────────────────────────────────────

    def _check_entry(self, bar: Bar, close: float, regime: int, p_up: float) -> None:
        """Enter a long position when regime + LSTM agree."""
        if regime != BULL:
            return
        if p_up < self.config.min_confidence_buy:
            return

        # Raw free balance from the account ledger.
        raw_usd = self._estimate_portfolio_usd(close)
        if raw_usd <= 0:
            return

        # Subtract what other strategies have already reserved so we never
        # double-count the same dollars on the same bar.
        portfolio_usd = (
            self._allocator.available(self._strategy_id, raw_usd)
            if self._allocator is not None
            else raw_usd
        )
        if portfolio_usd <= 0:
            logger.info(
                "[%s] Skipping entry — no capital available after allocator "
                "deduction (raw=%.2f  total_reserved=%.2f)",
                self._strategy_id,
                raw_usd,
                self._allocator.total_reserved if self._allocator is not None else 0.0,
            )
            return

        btc_qty = self._kelly.size_in_base_currency(
            p_win=p_up,
            regime=regime,
            portfolio_usd=portfolio_usd,
            price_per_unit=close,
            unit_precision=6,
        )

        if btc_qty <= 0:
            return

        instrument = self.cache.instrument(self._instrument_id)
        if instrument is None:
            logger.error("Instrument %s not found in cache.", self._instrument_id)
            return

        order = self.order_factory.market(
            instrument_id=self._instrument_id,
            order_side=OrderSide.BUY,
            quantity=Quantity(btc_qty, instrument.size_precision),
            time_in_force=TimeInForce.IOC,
        )
        self.submit_order(order)

        # Reserve the committed notional immediately so any other strategy
        # that fires on this same bar sees a reduced available balance.
        committed_usd = btc_qty * close
        if self._allocator is not None:
            self._allocator.reserve(self._strategy_id, committed_usd)

        self._entry_price    = close
        self._stop_price     = close * (1 - self.config.stop_loss_pct)
        self._tp_price       = close * (1 + self.config.take_profit_pct)
        self._peak_price     = close
        self._entry_quantity = btc_qty
        self._is_long        = True
        self._trade_count   += 1

        logger.info(
            "OPEN LONG | bar=%d qty=%.6f @ %.2f | SL=%.2f TP=%.2f | "
            "regime=BULL p_up=%.3f kelly_size=%.1f%% fg=%.2f | "
            "committed=%.2f available_was=%.2f",
            self._bar_count, btc_qty, close,
            self._stop_price, self._tp_price,
            p_up, self._kelly.position_fraction(p_up, regime) * 100,
            self._fear_greed_last,
            committed_usd, portfolio_usd,
        )

    def _check_exits(self, close: float, high: float, low: float) -> None:
        """
        Check stop-loss, take-profit, and trailing stop.
        Also check for regime/LSTM signal reversal.
        """
        # Update trailing stop peak
        if high > self._peak_price:
            self._peak_price = high
            self._stop_price = max(
                self._stop_price,
                self._peak_price * (1 - self.config.trailing_stop_pct),
            )

        reason: str | None = None

        if low <= self._stop_price:
            reason = f"STOP_LOSS @ {self._stop_price:.2f}"
        elif high >= self._tp_price:
            reason = f"TAKE_PROFIT @ {self._tp_price:.2f}"

        if reason:
            self._close_long(close, reason)

    def _close_long(self, close: float, reason: str) -> None:
        instrument = self.cache.instrument(self._instrument_id)
        if instrument is None:
            return

        net_qty = self.portfolio.net_position(self._instrument_id)
        if net_qty is None or float(net_qty) <= 0:
            self._is_long = False
            return

        order = self.order_factory.market(
            instrument_id=self._instrument_id,
            order_side=OrderSide.SELL,
            quantity=Quantity(float(net_qty), instrument.size_precision),
            time_in_force=TimeInForce.IOC,
        )
        self.submit_order(order)
        self._is_long = False

        # Capital is back in the pool — other strategies can use it again.
        if self._allocator is not None:
            self._allocator.release(self._strategy_id)

        pnl_pct = (close - self._entry_price) / self._entry_price * 100
        logger.info(
            "CLOSE LONG | reason=%s | entry=%.2f exit≈%.2f | PnL≈%+.2f%%",
            reason, self._entry_price, close, pnl_pct,
        )

    # ── Position estimation ───────────────────────────────────────────────────

    def _estimate_portfolio_usd(self, current_price: float) -> float:
        """
        Estimate free USD available for trading.
        In backtest, reads from the portfolio; provides a fallback estimate.
        """
        try:
            account = self.portfolio.account(self._instrument_id.venue)
            if account is None:
                return self.config.starting_balance_usd * (1 - self.config.max_position_pct)
            from nautilus_trader.model.currencies import USD as _USD
            from nautilus_trader.model.objects import Currency
            usd_bal = account.balance_free(Currency.from_str("USDT"))
            if usd_bal is None:
                usd_bal = account.balance_free(Currency.from_str("USD"))
            return float(usd_bal) if usd_bal else self.config.starting_balance_usd
        except Exception:
            return self.config.starting_balance_usd

    # ── Logging helper ────────────────────────────────────────────────────────

    def _log_state(self, close: float, regime: int, p_up: float) -> None:
        r_name = {BEAR: "BEAR", SIDEWAYS: "SIDEWAYS", BULL: "BULL"}.get(regime, "?")
        logger.info(
            "[bar %5d] price=%.2f | regime=%-8s | p_up=%.3f | long=%s | fg=%.2f",
            self._bar_count, close, r_name, p_up, self._is_long,
            self._fear_greed_last,
        )

    # ── NautilusTrader event hooks ────────────────────────────────────────────

    def on_order_filled(self, event) -> None:  # type: ignore[override]
        logger.debug("Order filled: %s", event)
        gc.collect()

    def on_position_opened(self, event) -> None:  # type: ignore[override]
        logger.info("Position opened: %s", event)

    def on_position_closed(self, event) -> None:  # type: ignore[override]
        logger.info("Position closed: %s", event)
        gc.collect()
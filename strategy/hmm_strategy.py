"""
strategy/hmm_strategy.py
HMM-only regime strategy for NautilusTrader (backtest + live).

Changes in this version
-----------------------
Fix 1 — EMA price-trend filter on entry
Fix 2 — Bear-exit persistence gate
Fix 3 — Reduced Kelly multiplier (0.50 → 0.40)
Fix 4 — Historical warm-up via request_bars()  (live only)
Fix 5 — Warm-up progress logging + HMM signal & entry-condition logging
Fix 6 — on_historical_data() override so request_bars() bars populate deques

  Root cause of Fix 6
  --------------------
  In NautilusTrader 1.224.0, bars returned by request_bars() are NOT
  routed through on_bar().  The data engine delivers them one-by-one via
  on_historical_data(), then fires the completion callback with the UUID4
  correlation ID.  Because on_historical_data() was not overridden, every
  historical bar was silently dropped, every deque stayed at 0, and the
  warm-up guard in on_bar() never cleared — so no HMM inference, no
  regime logs, and no entry signals ever appeared.

  Fix: override on_historical_data(data) to append each historical Bar to
  the OHLCV deques and emit warm-up progress lines.  on_bar() continues to
  handle live (WebSocket) bars and the organic backtest path unchanged.

  NautilusTrader 1.224.0 Cython signature (actor.pyx):

    request_bars(bar_type, start: datetime, end=None, limit=0, callback=None)

  `start` is the SECOND required positional argument — it has no default
  value in the compiled extension type.  Passing only `bar_type` (with
  callback and/or limit as keyword args) raises:
    TypeError: request_bars() takes at least 2 positional arguments (1 given)

  Fix: compute a UTC datetime far enough in the past to cover
  hmm_min_history bars (with a 50 % buffer for weekends / gaps), derive
  the bar duration from BarType.spec, and pass it as the second positional
  argument.  The callback is still passed as a keyword argument so that
  the completion banner fires once all bars have been delivered.

Fix 7 — _close_long() internal quantity tracking
  Root cause
  ----------
  generate_order_filled() raised TypeError (keyword-argument bug, now
  fixed in live_trading.py), so NautilusTrader never registered BUY
  fills.  portfolio.net_position() therefore returned 0, triggering the
  guard in _close_long():

      if net_qty is None or float(net_qty) <= 0: return   # ← no SELL sent

  self._is_long was reset to False, the strategy re-entered after cooldown
  with another BUY, and Roostoo accumulated open BUY positions with no
  corresponding SELL orders — appearing as if a BUY had been placed when
  a SELL was expected.

  Fix: track the purchased quantity in self._current_qty at entry time.
  _close_long() now falls back to self._current_qty when
  portfolio.net_position() returns 0, ensuring a SELL is always sent
  for every confirmed BUY.  self._current_qty is reset to 0.0 on every
  close path (whether a real SELL is submitted or truly no position
  exists).
"""

from __future__ import annotations

import gc
import logging
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import BarAggregation, OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

from models.hmm_model import BEAR, BULL, SIDEWAYS, RegimeHMM
from utils.capital_allocator import CapitalAllocator
from utils.kelly_criterion import KellyCriterion
from utils.trade_journal import TradeJournal

logger = logging.getLogger(__name__)

_REGIME_LABEL = {BEAR: "BEAR", SIDEWAYS: "SIDEWAYS", BULL: "BULL"}

# Seconds per bar-aggregation unit (used to compute warm-up start datetime)
_AGG_TO_SECONDS: dict[BarAggregation, int] = {
    BarAggregation.SECOND: 1,
    BarAggregation.MINUTE: 60,
    BarAggregation.HOUR:   3_600,
    BarAggregation.DAY:    86_400,
    BarAggregation.WEEK:   604_800,
}


# ── Config ────────────────────────────────────────────────────────────────────

class HMMStrategyConfig(StrategyConfig, frozen=True):
    instrument_id:           str
    bar_type:                str
    hmm_model_path:          str

    # Entry conditions
    min_bull_proba:          float = 0.45
    min_kelly_fraction:      float = 0.005

    # EMA trend filter (short-term: close > EMA(trend_ema_bars))
    trend_ema_bars:          int   = 2

    # 48-hour trend alignment
    trend_lookback_bars:     int   = 4

    # BULL entry persistence
    bull_entry_consecutive:  int   = 2

    # Kelly / position sizing
    kelly_fraction:          float = 0.70
    max_position_pct:        float = 0.70
    commission_rate:         float = 0.001

    # Exit — take-profit
    take_profit_pct:         float = 0.030

    # Exit — regime-adjusted trailing stops from peak
    trail_bull_pct:          float = 0.020
    trail_sideways_pct:      float = 0.012
    trail_bear_pct:          float = 0.006

    # Exit — immediate close on persistent bear signal
    bear_exit_proba:         float = 0.40
    bear_exit_consecutive:   int   = 2

    # Maximum bars to hold a position (192 bars = 48 hours at 15-min)
    max_holding_bars:        int   = 192

    # Re-entry cooldown
    min_bars_between_trades: int   = 8

    # HMM warm-up
    hmm_min_history:         int   = 150

    # Balance fallback
    starting_balance_usd:    float = 1_000_000.0


# ── Strategy ──────────────────────────────────────────────────────────────────

class HMMStrategy(Strategy):
    """
    Long-only HMM regime strategy.  All orders are MARKET (IOC).
    Compatible with BacktestEngine and TradingNode.
    """

    def __init__(
        self,
        config:    HMMStrategyConfig,
        allocator: Optional[CapitalAllocator] = None,
        journal:   Optional[TradeJournal]    = None,
    ) -> None:
        super().__init__(config)

        self._instrument_id = InstrumentId.from_str(config.instrument_id)
        self._bar_type      = BarType.from_str(config.bar_type)
        self._strategy_id   = config.instrument_id
        self._allocator     = allocator
        self._journal       = journal

        self._hmm: Optional[RegimeHMM] = None

        self._kelly = KellyCriterion(
            fraction         = config.kelly_fraction,
            max_position_pct = config.max_position_pct,
            take_profit_pct  = config.take_profit_pct,
            trail_entry_pct  = config.trail_bull_pct,
            commission_rate  = config.commission_rate,
        )

        # Rolling OHLCV buffers
        _buf = max(config.hmm_min_history, config.trend_lookback_bars) + 20
        self._closes  = deque(maxlen=_buf)
        self._highs   = deque(maxlen=_buf)
        self._lows    = deque(maxlen=_buf)
        self._volumes = deque(maxlen=_buf)

        # Position tracking
        self._is_long:          bool  = False
        self._entry_price:      float = 0.0
        self._stop_price:       float = 0.0
        self._tp_price:         float = 0.0
        self._peak_price:       float = 0.0
        self._bars_in_position: int   = 0

        # ── FIX 7: internally tracked buy quantity ────────────────────────────
        # Set to the purchased qty in _check_entry(); used as a fallback in
        # _close_long() when portfolio.net_position() returns 0 (e.g. because
        # the fill event was not registered due to the generate_order_filled()
        # keyword-argument bug).  Reset to 0.0 on every close path.
        self._current_qty: float = 0.0

        # Bar / trade counters
        self._bar_count:      int = 0   # live bars only (on_bar)
        self._hist_bar_count: int = 0   # historical bars (on_historical_data)
        self._last_exit_bar:  int = -999
        self._trade_count:    int = 0

        # Cached inference results
        self._last_regime:    int   = SIDEWAYS
        self._last_bull_prob: float = 1.0 / 3.0
        self._last_bear_prob: float = 1.0 / 3.0

        # Persistence counters
        self._consecutive_bear_bars: int = 0
        self._consecutive_bull_bars: int = 0

        # ── Observability state ───────────────────────────────────────────────
        #
        # _last_warmup_milestone  — last 10 % band logged (0–9); -1 = none yet
        # _warmup_complete_logged — one-shot "★ WARM-UP COMPLETE" guard
        # _last_logged_regime     — regime-transition detection; -999 = INIT
        self._last_warmup_milestone:  int  = -1
        self._warmup_complete_logged: bool = False
        self._last_logged_regime:     int  = -999   # sentinel


    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_start(self) -> None:
        self._hmm = RegimeHMM.load(Path(self.config.hmm_model_path))

        is_live = self._is_live_clock()

        if is_live:
            warmup_limit = self.config.hmm_min_history - 1
            start_dt = self._warmup_start_datetime(warmup_limit)
            try:
                self.request_bars(
                    self._bar_type,
                    start_dt,
                    # In NT 1.224.0 the callback receives the UUID4 correlation
                    # ID *after* all bars have been delivered to
                    # on_historical_data().  It is a completion signal only.
                    callback=self._on_historical_bars_complete,
                )
                logger.info(
                    "[%s] Requested ~%d historical warm-up bars since %s — "
                    "bars will be delivered via on_historical_data() "
                    "before the callback fires.",
                    self._instrument_id,
                    warmup_limit,
                    start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                )
            except Exception as exc:
                logger.warning(
                    "[%s] request_bars() raised %s — "
                    "falling back to organic warm-up from live bars "
                    "(first %d bars withheld from trading).",
                    self._instrument_id, exc,
                    self.config.hmm_min_history,
                )
        else:
            logger.info(
                "[%s] Backtest mode — warm-up handled organically by "
                "on_bar() guard (first %d bars withheld from trading).",
                self._instrument_id,
                self.config.hmm_min_history,
            )

        self.subscribe_bars(self._bar_type)

        logger.info(
            "HMMStrategy ready | %s | min_bull=%.2f | bear_exit=%.2f (×%d) | "
            "trail BULL=%.2f%%  SIDEWAYS=%.2f%%  BEAR=%.2f%% | "
            "kelly=%.0f%%  cap=%.0f%%  ema=%dbars | live=%s",
            self._instrument_id,
            self.config.min_bull_proba,
            self.config.bear_exit_proba,
            self.config.bear_exit_consecutive,
            self.config.trail_bull_pct     * 100,
            self.config.trail_sideways_pct * 100,
            self.config.trail_bear_pct     * 100,
            self.config.kelly_fraction     * 100,
            self.config.max_position_pct   * 100,
            self.config.trend_ema_bars,
            is_live,
        )
        logger.info(
            "Kelly break-even P(bull): %.3f  |  b: %.3f",
            self._kelly.breakeven_prob, self._kelly.win_loss_ratio,
        )

    def on_stop(self) -> None:
        if self._allocator is not None and self._is_long:
            self._allocator.release(self._strategy_id)
        logger.info(
            "HMMStrategy stopped | trades=%d | last_regime=%s | "
            "bull=%.3f  bear=%.3f",
            self._trade_count,
            _REGIME_LABEL.get(self._last_regime, "?"),
            self._last_bull_prob,
            self._last_bear_prob,
        )

    # ── Clock-type helper ─────────────────────────────────────────────────────

    def _is_live_clock(self) -> bool:
        clock_type_name = type(self.clock).__name__
        return "Live" in clock_type_name

    # ── Warm-up datetime helper ───────────────────────────────────────────────

    def _warmup_start_datetime(self, num_bars: int) -> datetime:
        """
        Return a timezone-aware UTC datetime far enough in the past to
        cover `num_bars` bars of this strategy's bar type.

        A 50 % buffer is added on top of the nominal lookback to absorb
        weekends, exchange downtime, and Binance data gaps.
        """
        spec = self._bar_type.spec
        secs_per_bar = _AGG_TO_SECONDS.get(spec.aggregation, 60) * spec.step
        # 1.5× buffer so weekends / gaps don't leave the buffer short
        total_secs = int(secs_per_bar * num_bars * 1.5)
        return datetime.now(timezone.utc) - timedelta(seconds=total_secs)

    # ── Warm-up progress helper (shared by both paths) ────────────────────────

    def _maybe_log_warmup_progress(self) -> None:
        """
        Emit one INFO line per 10 % of warm-up progress.

        Safe to call from both on_historical_data() and on_bar() — the
        milestone counter prevents duplicate lines across paths.

        Example output (hmm_min_history = 150):
            [BTCUSDT.BINANCE] WARM-UP   15/150  (10%) — 135 bars remaining (~2025 min)
            [BTCUSDT.BINANCE] WARM-UP   30/150  (20%) — 120 bars remaining (~1800 min)
            ...
            [BTCUSDT.BINANCE] WARM-UP  135/150  (90%) —  15 bars remaining  (~225 min)
        """
        n        = len(self._closes)
        required = self.config.hmm_min_history
        if n >= required:
            return
        milestone = (n * 10) // required   # 0 – 9
        if milestone <= self._last_warmup_milestone:
            return
        self._last_warmup_milestone = milestone
        remaining = required - n
        logger.info(
            "[%s] WARM-UP  %3d / %d bars  (%d%%)  —  "
            "%d bars remaining  (~%d min at 15-min cadence)",
            self._instrument_id,
            n, required,
            milestone * 10,
            remaining,
            remaining * 15,
        )

    # ── Historical bar handler (live only — Fix 6) ────────────────────────────

    def on_historical_data(self, data) -> None:
        """
        Called by NautilusTrader for every bar returned by request_bars().

        In NT 1.224.0 these bars are NOT routed through on_bar() — they
        arrive here instead.  We populate the OHLCV deques and log warm-up
        progress exactly as on_bar() would during an organic backtest
        warm-up, so the deques are full before the first live bar arrives.

        The method intentionally ignores any data type that is not a Bar
        for this bar type (e.g. trades, quotes) so it is safe to call from
        the base class hook without extra guards in the caller.
        """
        if not isinstance(data, Bar):
            return
        if data.bar_type != self._bar_type:
            return

        self._closes.append(float(data.close))
        self._highs.append(float(data.high))
        self._lows.append(float(data.low))
        self._volumes.append(float(data.volume))
        self._hist_bar_count += 1

        # Emit warm-up progress at every 10 % milestone
        self._maybe_log_warmup_progress()

        # One-shot WARM-UP COMPLETE banner the moment the deque is full
        if (not self._warmup_complete_logged
                and len(self._closes) >= self.config.hmm_min_history):
            self._warmup_complete_logged = True
            logger.info(
                "[%s] ★ WARM-UP COMPLETE (historical) — "
                "%d / %d bars loaded via on_historical_data(); "
                "HMM signals active from the next live bar.",
                self._instrument_id,
                len(self._closes),
                self.config.hmm_min_history,
            )

    # ── Historical bar completion callback (live only) ────────────────────────

    def _on_historical_bars_complete(self, correlation_id) -> None:
        """
        Fired by NautilusTrader with the UUID4 correlation ID once the
        entire historical bar response has been processed.

        Because the callback fires AFTER all on_historical_data() calls
        have completed, the deque count here reflects the true number of
        historical bars that were loaded.
        """
        n        = len(self._closes)
        required = self.config.hmm_min_history
        warmed   = n >= required

        logger.info(
            "[%s] Historical bar request done | correlation_id=%s | "
            "hist_bars=%d  deque=%d / required=%d | warmed=%s",
            self._instrument_id,
            correlation_id,
            self._hist_bar_count,
            n,
            required,
            warmed,
        )

        if not warmed:
            logger.warning(
                "[%s] Deque has only %d / %d bars after historical load — "
                "the gap will be filled organically from live bars; "
                "trading is withheld until the threshold is met.",
                self._instrument_id, n, required,
            )

    # ── Bar handler (live WebSocket bars + backtest) ───────────────────────────

    def on_bar(self, bar: Bar) -> None:
        self._bar_count += 1
        close  = float(bar.close)
        high   = float(bar.high)
        low    = float(bar.low)
        volume = float(bar.volume)

        self._closes.append(close)
        self._highs.append(high)
        self._lows.append(low)
        self._volumes.append(volume)

        # ── Heartbeat every 4 live bars (~1 h at 15-min cadence) ─────────────
        # Confirms the WebSocket feed is alive and shows current warm-up state.
        # Fires on live bars 1, 5, 9, 13, … (independent of historical bars).
        if self._bar_count % 4 == 1:
            logger.info(
                "[%s] ♥ live_bar=%-5d | close=%-12.4f | "
                "deque=%d/%d | warmed=%s | long=%s",
                self._instrument_id,
                self._bar_count,
                close,
                len(self._closes),
                self.config.hmm_min_history,
                len(self._closes) >= self.config.hmm_min_history,
                self._is_long,
            )

        # Step 1 — mechanical price exits (before HMM, always fire)
        if self._is_long:
            self._check_mechanical_exits(close, high, low)
            if not self._is_long:
                return

        # ── Step 2 — warm-up guard ────────────────────────────────────────────
        #
        # In live mode this guard clears on the first live bar after
        # on_historical_data() has populated the deque.
        #
        # In backtest / organic-fallback mode the deque fills from on_bar()
        # alone; _maybe_log_warmup_progress() emits the milestone lines and
        # the banner fires here once the threshold is met.
        if len(self._closes) < self.config.hmm_min_history:
            self._maybe_log_warmup_progress()
            return

        # One-shot banner for backtest / fallback (live path logs in
        # on_historical_data; _warmup_complete_logged prevents double-firing)
        if not self._warmup_complete_logged:
            self._warmup_complete_logged = True
            logger.info(
                "[%s] ★ WARM-UP COMPLETE — %d bars in deque, "
                "HMM signals now active.",
                self._instrument_id, len(self._closes),
            )

        # Step 3 — HMM inference
        regime, bull_prob, sideways_prob, bear_prob = self._run_inference()
        if regime is None:
            return

        # ── Regime-change log ─────────────────────────────────────────────────
        #
        # Emits one INFO line whenever the dominant regime flips.  The very
        # first inference uses the sentinel label "INIT".  Repeated bars at
        # the same regime are demoted to DEBUG to keep the INFO stream clean.
        if regime != self._last_logged_regime:
            prev_label = _REGIME_LABEL.get(self._last_logged_regime, "INIT")
            logger.info(
                "[%s] ▶ REGIME  %s → %s | "
                "bull=%.3f  side=%.3f  bear=%.3f | live_bar=%d",
                self._instrument_id,
                prev_label,
                _REGIME_LABEL.get(regime, "?"),
                bull_prob, sideways_prob, bear_prob,
                self._bar_count,
            )
            self._last_logged_regime = regime
        else:
            logger.debug(
                "[%s] regime=%s | bull=%.3f  side=%.3f  bear=%.3f",
                self._instrument_id,
                _REGIME_LABEL.get(regime, "?"),
                bull_prob, sideways_prob, bear_prob,
            )

        self._last_regime    = regime
        self._last_bull_prob = bull_prob
        self._last_bear_prob = bear_prob

        # Step 4 — regime-driven exit
        if self._is_long:
            self._check_regime_exit(close, regime, bear_prob)
            if not self._is_long:
                return

        # Step 5 — periodic full-state snapshot (every 20 live bars ≈ 5 h)
        if self._bar_count % 20 == 0:
            self._log_state(close, regime, bull_prob, bear_prob)

        # Step 6 — BULL entry persistence counter (while flat only)
        if not self._is_long:
            if regime == BULL:
                self._consecutive_bull_bars += 1
            else:
                self._consecutive_bull_bars = 0

        # Step 7 — entry (if flat and cooldown elapsed)
        if not self._is_long:
            bars_since_exit = self._bar_count - self._last_exit_bar
            cooldown_ok     = bars_since_exit >= self.config.min_bars_between_trades
            if cooldown_ok:
                self._check_entry(bar, close, regime, bull_prob, bear_prob)
            elif regime == BULL and bull_prob >= self.config.min_bull_proba:
                bars_left = self.config.min_bars_between_trades - bars_since_exit
                logger.info(
                    "[%s] BULL but NO ENTRY | cooldown: %d bar(s) remaining "
                    "(exited live_bar=%d, next eligible live_bar=%d)",
                    self._strategy_id,
                    bars_left,
                    self._last_exit_bar,
                    self._last_exit_bar + self.config.min_bars_between_trades,
                )

    # ── HMM inference ─────────────────────────────────────────────────────────

    def _run_inference(
        self,
    ) -> tuple[Optional[int], float, float, float]:
        try:
            prices  = np.array(self._closes,  dtype=np.float32)
            volumes = np.array(self._volumes, dtype=np.float32)

            proba_matrix  = self._hmm.predict_proba(prices, volumes)
            last          = proba_matrix[-1]

            bear_prob     = float(last[BEAR])
            sideways_prob = float(last[SIDEWAYS])
            bull_prob     = float(last[BULL])
            regime        = int(np.argmax(last))

            del prices, volumes, proba_matrix
            return regime, bull_prob, sideways_prob, bear_prob

        except Exception as exc:
            logger.warning(
                "[%s] HMM inference error (live_bar %d): %s",
                self._strategy_id, self._bar_count, exc,
            )
            return None, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0

    # ── Entry ─────────────────────────────────────────────────────────────────

    def _check_entry(
        self,
        bar:       Bar,
        close:     float,
        regime:    int,
        bull_prob: float,
        bear_prob: float,
    ) -> None:
        # ── Gate 1: regime must be BULL ───────────────────────────────────────
        # Non-BULL rejections are logged at DEBUG — expected and would flood
        # INFO at 15-min cadence during sideways / bear markets.
        if regime != BULL:
            logger.debug(
                "[%s] No entry | regime=%s (need BULL) | bull=%.3f  bear=%.3f",
                self._strategy_id,
                _REGIME_LABEL.get(regime, "?"),
                bull_prob, bear_prob,
            )
            return

        # ── From here regime IS BULL — log every rejection at INFO ────────────

        if bull_prob < self.config.min_bull_proba:
            logger.info(
                "[%s] BULL but NO ENTRY | bull_prob=%.3f < min_bull_proba=%.3f",
                self._strategy_id,
                bull_prob, self.config.min_bull_proba,
            )
            return

        if bear_prob >= self.config.bear_exit_proba:
            logger.info(
                "[%s] BULL but NO ENTRY | bear_prob=%.3f ≥ bear_exit_proba=%.3f",
                self._strategy_id,
                bear_prob, self.config.bear_exit_proba,
            )
            return

        if not self._price_above_ema(close):
            logger.info(
                "[%s] BULL but NO ENTRY | price=%.4f below EMA(%d)",
                self._strategy_id,
                close, self.config.trend_ema_bars,
            )
            return

        if not self._trend_aligned():
            logger.info(
                "[%s] BULL but NO ENTRY | trend not aligned "
                "(close=%.4f < close[-%d]=%.4f)",
                self._strategy_id,
                close,
                self.config.trend_lookback_bars,
                list(self._closes)[-(self.config.trend_lookback_bars + 1)],
            )
            return

        if self._consecutive_bull_bars < self.config.bull_entry_consecutive:
            logger.info(
                "[%s] BULL but NO ENTRY | bull_consec=%d < required=%d "
                "(%d more consecutive BULL bar(s) needed)",
                self._strategy_id,
                self._consecutive_bull_bars,
                self.config.bull_entry_consecutive,
                self.config.bull_entry_consecutive - self._consecutive_bull_bars,
            )
            return

        kelly_frac = self._kelly.position_fraction(bull_prob)
        if kelly_frac < self.config.min_kelly_fraction:
            logger.info(
                "[%s] BULL but NO ENTRY | kelly_frac=%.5f < min_kelly=%.5f",
                self._strategy_id,
                kelly_frac, self.config.min_kelly_fraction,
            )
            return

        raw_usd = self._estimate_free_usd()
        if raw_usd < 100.0:
            logger.info(
                "[%s] BULL but NO ENTRY | free_usd=%.2f < $100 minimum",
                self._strategy_id, raw_usd,
            )
            return

        portfolio_usd = (
            self._allocator.available(self._strategy_id, raw_usd)
            if self._allocator is not None
            else raw_usd
        )
        if portfolio_usd < 100.0:
            logger.info(
                "[%s] BULL but NO ENTRY | allocator reserves — "
                "available=$%.2f  raw_free=$%.2f",
                self._strategy_id, portfolio_usd, raw_usd,
            )
            return

        qty = self._kelly.size_in_base_currency(
            bull_prob=bull_prob,
            portfolio_usd=portfolio_usd,
            price=close,
        )
        if qty <= 0:
            logger.info(
                "[%s] BULL but NO ENTRY | kelly qty=%.8f ≤ 0",
                self._strategy_id, qty,
            )
            return

        instrument = self.cache.instrument(self._instrument_id)
        if instrument is None:
            logger.error(
                "[%s] BULL but NO ENTRY | instrument not found in cache.",
                self._strategy_id,
            )
            return

        # ── All conditions passed — log full signal before submitting ─────────
        committed_usd = qty * close
        logger.info(
            "[%s] ✓ ALL ENTRY CONDITIONS MET | "
            "bull=%.3f  bear=%.3f  kelly=%.4f  consec=%d  "
            "qty=%.6f  price=%.4f  committed=$%.2f — submitting BUY",
            self._strategy_id,
            bull_prob, bear_prob, kelly_frac,
            self._consecutive_bull_bars,
            qty, close, committed_usd,
        )

        order = self.order_factory.market(
            instrument_id = self._instrument_id,
            order_side    = OrderSide.BUY,
            quantity      = Quantity(qty, instrument.size_precision),
            time_in_force = TimeInForce.IOC,
        )
        self.submit_order(order)

        self._entry_price            = close
        self._tp_price               = close * (1.0 + self.config.take_profit_pct)
        self._peak_price             = close
        self._stop_price             = close * (1.0 - self.config.trail_bull_pct)
        self._is_long                = True
        self._bars_in_position       = 0
        self._consecutive_bear_bars  = 0
        self._consecutive_bull_bars  = 0
        self._trade_count           += 1
        # ── FIX 7: record quantity so _close_long() can fall back to it ───────
        self._current_qty            = qty

        if self._journal is not None:
            self._journal.open_trade(
                symbol      = self._strategy_id,
                entry_price = close,
                quantity    = qty,
                regime      = _REGIME_LABEL.get(self._last_regime, "UNKNOWN"),
                bull_prob   = bull_prob,
            )

        if self._allocator is not None:
            self._allocator.reserve(self._strategy_id, committed_usd)

        logger.info(
            "OPEN LONG | live_bar=%d | %s @ %.4f | qty=%.6f | "
            "TP=%.4f  stop=%.4f | "
            "bull=%.3f  bear=%.3f  kelly=%.2f%%  bull_consec=%d  $=%.2f",
            self._bar_count, self._strategy_id, close, qty,
            self._tp_price, self._stop_price,
            bull_prob, bear_prob, kelly_frac * 100,
            self._consecutive_bull_bars, committed_usd,
        )

    # ── Exits ─────────────────────────────────────────────────────────────────

    def _check_mechanical_exits(
        self, close: float, high: float, low: float
    ) -> None:
        self._bars_in_position += 1

        if self._bars_in_position >= self.config.max_holding_bars:
            self._close_long(
                close,
                f"MAX_HOLD@{self._bars_in_position}bars",
            )
            return

        if high > self._peak_price:
            self._peak_price = high

        trail_pct = self._trail_pct_for_regime(self._last_regime)
        new_stop  = self._peak_price * (1.0 - trail_pct)
        if new_stop > self._stop_price:
            self._stop_price = new_stop

        reason: Optional[str] = None
        if high >= self._tp_price:
            reason = f"TAKE_PROFIT @ {self._tp_price:.4f}"
        elif low <= self._stop_price:
            reason = f"TRAILING_STOP @ {self._stop_price:.4f}"

        if reason:
            self._close_long(close, reason)

    def _check_regime_exit(
        self, close: float, regime: int, bear_prob: float
    ) -> None:
        trail_pct = self._trail_pct_for_regime(regime)
        new_stop  = self._peak_price * (1.0 - trail_pct)
        if new_stop > self._stop_price:
            self._stop_price = new_stop
            logger.debug(
                "[%s] Stop tightened → %.4f (regime=%s  trail=%.2f%%)",
                self._strategy_id, self._stop_price,
                _REGIME_LABEL.get(regime, "?"), trail_pct * 100,
            )

        if bear_prob >= self.config.bear_exit_proba:
            self._consecutive_bear_bars += 1
            if self._consecutive_bear_bars >= self.config.bear_exit_consecutive:
                self._close_long(
                    close,
                    f"BEAR_SIGNAL_EXIT  consecutive={self._consecutive_bear_bars}"
                    f"  P(bear)={bear_prob:.3f}",
                )
        else:
            self._consecutive_bear_bars = 0

    def _close_long(self, close: float, reason: str) -> None:
        instrument = self.cache.instrument(self._instrument_id)
        if instrument is None:
            self._is_long                = False
            self._current_qty            = 0.0
            self._consecutive_bear_bars  = 0
            self._consecutive_bull_bars  = 0
            self._bars_in_position       = 0
            return

        # ── FIX 7: robust quantity resolution ────────────────────────────────
        #
        # Primary source: NautilusTrader portfolio (reflects confirmed fills).
        # Fallback:       self._current_qty (set at entry time).
        #
        # The fallback fires when portfolio.net_position() returns 0 because
        # the fill event was not registered (e.g. due to the
        # generate_order_filled() keyword-argument bug that has since been
        # fixed in live_trading.py).  Without the fallback, _close_long()
        # would silently skip the SELL order, leave the Roostoo position open,
        # reset _is_long to False, and allow the strategy to re-enter with
        # another BUY — resulting in multiple unclosed BUY positions on
        # Roostoo with no corresponding SELL orders.
        net_qty   = self.portfolio.net_position(self._instrument_id)
        qty_float = float(net_qty) if net_qty is not None else 0.0

        if qty_float <= 0.0 and self._current_qty > 0.0:
            logger.warning(
                "[%s] _close_long: portfolio.net_position() returned %s "
                "— falling back to internally tracked qty=%.8f for SELL. "
                "If this persists after restarting, check that "
                "generate_order_filled() is being called correctly.",
                self._strategy_id, net_qty, self._current_qty,
            )
            qty_float = self._current_qty

        if qty_float <= 0.0:
            # Truly no position (e.g. already closed elsewhere) — update
            # internal state without submitting a redundant SELL order.
            logger.warning(
                "[%s] _close_long called but qty=0 and _current_qty=0 "
                "— no SELL submitted (reason: %s).",
                self._strategy_id, reason,
            )
            self._is_long                = False
            self._current_qty            = 0.0
            self._consecutive_bear_bars  = 0
            self._consecutive_bull_bars  = 0
            self._bars_in_position       = 0
            self._last_exit_bar          = self._bar_count
            if self._allocator is not None:
                self._allocator.release(self._strategy_id)
            return

        order = self.order_factory.market(
            instrument_id = self._instrument_id,
            order_side    = OrderSide.SELL,
            quantity      = Quantity(qty_float, instrument.size_precision),
            time_in_force = TimeInForce.IOC,
        )

        if self._journal is not None:
            self._journal.close_trade(
                symbol      = self._strategy_id,
                exit_price  = close,
                exit_reason = reason,
            )

        self.submit_order(order)

        pnl_pct   = (close - self._entry_price) / (self._entry_price + 1e-12) * 100
        held_bars = self._bars_in_position

        self._is_long                = False
        self._current_qty            = 0.0   # ← reset on every real close
        self._consecutive_bear_bars  = 0
        self._consecutive_bull_bars  = 0
        self._bars_in_position       = 0
        self._last_exit_bar          = self._bar_count

        if self._allocator is not None:
            self._allocator.release(self._strategy_id)

        logger.info(
            "CLOSE LONG | %s | live_bar=%d | reason=%s | "
            "entry=%.4f  exit≈%.4f  PnL≈%+.2f%%  held=%d bars",
            self._strategy_id, self._bar_count, reason,
            self._entry_price, close, pnl_pct, held_bars,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _price_above_ema(self, close: float) -> bool:
        lb = self.config.trend_ema_bars
        if len(self._closes) < lb:
            return True
        prices = list(self._closes)
        alpha  = 2.0 / (lb + 1)
        ema    = prices[0]
        for p in prices[1:]:
            ema = alpha * p + (1.0 - alpha) * ema
        return close > ema

    def _trail_pct_for_regime(self, regime: int) -> float:
        if regime == BULL:
            return self.config.trail_bull_pct
        elif regime == SIDEWAYS:
            return self.config.trail_sideways_pct
        else:
            return self.config.trail_bear_pct

    def _trend_aligned(self) -> bool:
        lb = self.config.trend_lookback_bars
        if len(self._closes) <= lb:
            return True
        prices = list(self._closes)
        return prices[-1] > prices[-(lb + 1)]

    def _estimate_free_usd(self) -> float:
        try:
            account = self.portfolio.account(self._instrument_id.venue)
            if account is None:
                return self.config.starting_balance_usd
            from nautilus_trader.model.objects import Currency
            for ccy in ("USDT", "USD"):
                bal = account.balance_free(Currency.from_str(ccy))
                if bal is not None:
                    return float(bal)
            return self.config.starting_balance_usd
        except Exception:
            return self.config.starting_balance_usd

    def _log_state(
        self,
        close:     float,
        regime:    int,
        bull_prob: float,
        bear_prob: float,
    ) -> None:
        logger.info(
            "[live_bar %5d] %s | price=%.4f | %-8s | bull=%.3f bear=%.3f | "
            "long=%s | stop=%.4f | bear_consec=%d | bull_consec=%d | held=%d",
            self._bar_count, self._strategy_id, close,
            _REGIME_LABEL.get(regime, "?"),
            bull_prob, bear_prob,
            self._is_long, self._stop_price,
            self._consecutive_bear_bars,
            self._consecutive_bull_bars,
            self._bars_in_position,
        )

    # ── NautilusTrader event hooks ────────────────────────────────────────────

    def on_order_filled(self, event) -> None:   # type: ignore[override]
        logger.debug("Order filled: %s", event)
        gc.collect()

    def on_position_opened(self, event) -> None:  # type: ignore[override]
        logger.info("Position opened: %s", event)

    def on_position_closed(self, event) -> None:  # type: ignore[override]
        logger.info("Position closed: %s", event)
        gc.collect()
"""
evaluate_models.py
HMM-only diagnostics for all instruments in config.TRADING_INSTRUMENTS.

Changes in this version
-----------------------
Fix 1 — Multi-horizon calibration charts
    Added P(Bull) → forward-return charts at 6h (24-bar) and 24h (96-bar)
    horizons in addition to the existing 1-bar chart.  If the HMM detects
    genuine multi-day regimes the positive slope will appear most clearly at
    the longer horizons even when the 1-bar chart is flat.  If all horizons
    are flat or inverted the state labeling or features need to be fixed.

Fix 2 — Causal (rolling-window) HMM inference for simulation
    _simulate_pnl() uses posteriors from _compute_causal_proba(), which
    slides a fixed window and takes only the terminal row of each
    predict_proba() call — matching what the live strategy actually sees.

Fix 3 — Strategy guards mirrored in simulation
    EMA trend filter, bear-exit persistence, bull-entry persistence,
    48h lookback filter, and max holding bars are all included.

Usage
-----
    python evaluate_models.py
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.data import BarType
from nautilus_trader.persistence.catalog import ParquetDataCatalog

import config as cfg
from models.hmm_model import BEAR, BULL, SIDEWAYS, RegimeHMM
from utils.diagnostics import DiagnosticEngine
from utils.kelly_criterion import KellyCriterion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("evaluate_models")

_REGIME_NAMES  = {BEAR: "BEAR", SIDEWAYS: "SIDEWAYS", BULL: "BULL"}
_REGIME_COLORS = {BEAR: "#ef5350", SIDEWAYS: "#ffa726", BULL: "#66bb6a"}


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_arrays(
    catalog:      ParquetDataCatalog,
    bar_type_str: str,
    start:        str,
    end:          str,
) -> dict[str, np.ndarray]:
    start_ns = dt_to_unix_nanos(pd.Timestamp(start, tz="UTC"))
    end_ns   = dt_to_unix_nanos(pd.Timestamp(end,   tz="UTC"))
    bar_type = BarType.from_str(bar_type_str)
    bars     = catalog.bars(bar_types=[str(bar_type)], start=start_ns, end=end_ns)
    bars.sort(key=lambda b: b.ts_init)
    return {
        "closes":  np.array([float(b.close)  for b in bars], dtype=np.float32),
        "volumes": np.array([float(b.volume) for b in bars], dtype=np.float32),
    }


# ── Causal (rolling-window) HMM posteriors ────────────────────────────────────

def _compute_causal_proba(
    hmm_obj:     RegimeHMM,
    closes:      np.ndarray,
    volumes:     np.ndarray,
    window:      int = 300,
    min_history: int = cfg.HMM_MIN_HISTORY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute causal (rolling-window) HMM posteriors for every aligned bar.

    Slides a window of length `window` and extracts only the last row of each
    predict_proba() call — replicating the causal filtered estimate that the
    live strategy produces bar-by-bar.

    Returns
    -------
    bull_proba : float32 (N-1,)
    bear_proba : float32 (N-1,)
    regimes    : int32   (N-1,)
    """
    N_aligned  = len(closes) - 1
    bull_proba = np.full(N_aligned, 1.0 / 3.0, dtype=np.float32)
    bear_proba = np.full(N_aligned, 1.0 / 3.0, dtype=np.float32)
    regimes    = np.full(N_aligned, SIDEWAYS,   dtype=np.int32)

    log_step = max(1, N_aligned // 10)
    t0       = time.perf_counter()

    for i in range(N_aligned):
        start_i = max(0, i + 2 - window)
        c_win   = closes[start_i : i + 2]
        v_win   = volumes[start_i : i + 2]

        if len(c_win) < min_history:
            continue

        try:
            proba         = hmm_obj.predict_proba(c_win, v_win)
            last          = proba[-1]
            regimes[i]    = int(np.argmax(last))
            bull_proba[i] = float(last[BULL])
            bear_proba[i] = float(last[BEAR])
        except Exception as exc:
            logger.debug("Causal proba failed at bar %d: %s", i, exc)

        if (i + 1) % log_step == 0:
            elapsed = time.perf_counter() - t0
            logger.info(
                "Causal inference %d/%d  (%.0f%%)  elapsed=%.1fs",
                i + 1, N_aligned, 100.0 * (i + 1) / N_aligned, elapsed,
            )

    logger.info(
        "Causal inference complete — %.1fs for %d bars",
        time.perf_counter() - t0, N_aligned,
    )
    return bull_proba, bear_proba, regimes


# ── PnL simulation ────────────────────────────────────────────────────────────

def _simulate_pnl(
    aligned_closes:         np.ndarray,
    bull_proba:             np.ndarray,
    bear_proba:             np.ndarray,
    regimes:                np.ndarray,
    kelly:                  KellyCriterion,
    ema_bars:               int = cfg.TREND_EMA_BARS,
    bear_exit_consecutive:  int = cfg.BEAR_EXIT_CONSECUTIVE,
    bull_entry_consecutive: int = cfg.BULL_ENTRY_CONSECUTIVE,
    trend_lookback_bars:    int = cfg.TREND_LOOKBACK_BARS,
    max_holding_bars:       int = cfg.MAX_HOLDING_BARS,
) -> tuple[np.ndarray, pd.DataFrame]:

    N   = len(regimes)
    eq  = np.ones(N + 1, dtype=np.float64)
    bal = 1.0

    is_long          = False
    entry_price      = 0.0
    stop_price       = 0.0
    tp_price         = 0.0
    peak_price       = 0.0
    pos_frac         = 0.0
    last_exit        = -999
    last_regime      = SIDEWAYS
    bars_in_position = 0
    trades_list: list[dict] = []

    consecutive_bear = 0
    consecutive_bull = 0

    alpha = 2.0 / (ema_bars + 1)
    ema   = float(aligned_closes[0])

    for i in range(N):
        close  = float(aligned_closes[i])
        regime = int(regimes[i])
        bull_p = float(bull_proba[i])
        bear_p = float(bear_proba[i])

        ema       = alpha * close + (1.0 - alpha) * ema
        above_ema = close > ema

        above_trend = (
            i >= trend_lookback_bars
            and close > float(aligned_closes[i - trend_lookback_bars])
        )

        eq[i] = bal

        if is_long:
            bars_in_position += 1

            # Max holding period — close before checking stops
            if bars_in_position >= max_holding_bars:
                gross   = pos_frac * (close / entry_price - 1.0)
                comm    = 2.0 * cfg.COMMISSION_RATE * pos_frac
                net_pnl = gross - comm
                bal    += net_pnl
                trades_list.append({
                    "entry": entry_price, "exit": close,
                    "pnl":   net_pnl,
                    "reason": f"MAX_HOLD@{bars_in_position}bars",
                    "regime_at_exit": regime,
                })
                consecutive_bear = 0
                consecutive_bull = 0
                is_long          = False
                bars_in_position = 0
                last_exit        = i
                continue

            if close > peak_price:
                peak_price = close

            trail_pct = (
                cfg.TRAIL_BULL_PCT     if last_regime == BULL
                else cfg.TRAIL_SIDEWAYS_PCT if last_regime == SIDEWAYS
                else cfg.TRAIL_BEAR_PCT
            )
            new_stop = peak_price * (1.0 - trail_pct)
            if new_stop > stop_price:
                stop_price = new_stop

            rt_trail = (
                cfg.TRAIL_BULL_PCT     if regime == BULL
                else cfg.TRAIL_SIDEWAYS_PCT if regime == SIDEWAYS
                else cfg.TRAIL_BEAR_PCT
            )
            rt_stop = peak_price * (1.0 - rt_trail)
            if rt_stop > stop_price:
                stop_price = rt_stop
            last_regime = regime

            exit_price: Optional[float] = None
            reason:     Optional[str]   = None

            if close >= tp_price:
                exit_price, reason = close, "TAKE_PROFIT"
                consecutive_bear   = 0
            elif close <= stop_price:
                exit_price, reason = close, "TRAILING_STOP"
                consecutive_bear   = 0
            elif bear_p >= cfg.BEAR_EXIT_PROBA:
                consecutive_bear += 1
                if consecutive_bear >= bear_exit_consecutive:
                    exit_price = close
                    reason     = f"BEAR_SIGNAL (n={consecutive_bear})"
            else:
                consecutive_bear = 0

            if exit_price is not None:
                gross   = pos_frac * (exit_price / entry_price - 1.0)
                comm    = 2.0 * cfg.COMMISSION_RATE * pos_frac
                net_pnl = gross - comm
                bal    += net_pnl
                trades_list.append({
                    "entry": entry_price, "exit": exit_price,
                    "pnl":   net_pnl, "reason": reason,
                    "regime_at_exit": regime,
                })
                consecutive_bear = 0
                consecutive_bull = 0
                is_long          = False
                bars_in_position = 0
                last_exit        = i
                continue

        # Bull persistence tracking (while flat)
        if not is_long:
            if regime == BULL:
                consecutive_bull += 1
            else:
                consecutive_bull = 0

        # Entry
        if not is_long and (i - last_exit) >= cfg.MIN_BARS_BETWEEN_TRADES:
            if (
                regime   == BULL
                and bull_p  >= cfg.MIN_BULL_PROBA
                and bear_p  <  cfg.BEAR_EXIT_PROBA
                and above_ema
                and above_trend
                and consecutive_bull >= bull_entry_consecutive
            ):
                kf = kelly.position_fraction(bull_p)
                if kf >= cfg.MIN_KELLY_FRACTION:
                    entry_price      = close
                    tp_price         = close * (1.0 + cfg.TAKE_PROFIT_PCT)
                    stop_price       = close * (1.0 - cfg.TRAIL_BULL_PCT)
                    peak_price       = close
                    pos_frac         = kf
                    last_regime      = BULL
                    consecutive_bear = 0
                    consecutive_bull = 0
                    bars_in_position = 0
                    is_long          = True

    eq[N] = bal
    trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame(
        columns=["entry", "exit", "pnl", "reason", "regime_at_exit"]
    )
    return eq, trades_df


# ── Calibration chart helper ──────────────────────────────────────────────────

def _plot_calibration(
    causal_bull:    np.ndarray,
    forward_ret:    np.ndarray,
    n_valid:        int,
    horizon_label:  str,
    symbol:         str,
    out_dir:        Path,
    ticker:         str,
    filename_tag:   str,
) -> None:
    """
    Plot mean forward return per P(Bull) decile and save to out_dir.

    Parameters
    ----------
    causal_bull   : causal P(Bull) array, length ≥ n_valid
    forward_ret   : forward return array, length ≥ n_valid
    n_valid       : number of aligned bars to use
    horizon_label : human-readable horizon string, e.g. "1-bar (15 min)"
    symbol        : instrument symbol for title
    out_dir       : output directory
    ticker        : short ticker for filename
    filename_tag  : short tag for filename, e.g. "1bar", "24bar", "96bar"
    """
    if n_valid <= 0:
        return

    bins = np.percentile(causal_bull[:n_valid], np.linspace(0, 100, 11))
    bins = np.unique(bins)
    if len(bins) < 2:
        return

    means, centers = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (causal_bull[:n_valid] >= lo) & (causal_bull[:n_valid] < hi)
        if mask.sum() > 10:
            means.append(float(forward_ret[:n_valid][mask].mean()) * 100)
            centers.append(float((lo + hi) / 2))

    if not means:
        return

    width = float(bins[1] - bins[0]) * 0.8

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        centers, means, width=width,
        color=["#66bb6a" if m >= 0 else "#ef5350" for m in means],
        edgecolor="k", alpha=0.85,
    )
    ax.axhline(0, color="black", lw=1)
    ax.axvline(
        cfg.MIN_BULL_PROBA, color="darkgreen", ls="--",
        label=f"Entry threshold ({cfg.MIN_BULL_PROBA})",
    )
    ax.set_title(
        f"Causal P(Bull) → {horizon_label} Forward Return — {symbol} (OOS)\n"
        f"(rolling-window inference, no look-ahead bias)"
    )
    ax.set_xlabel("P(Bull)  [causal]")
    ax.set_ylabel(f"Mean {horizon_label} forward log-return (%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"bull_prob_vs_returns_{filename_tag}_{ticker}.png", dpi=150)
    plt.close(fig)
    logger.info("Saved calibration chart: %s horizon", horizon_label)


# ── Per-instrument evaluation ─────────────────────────────────────────────────

def evaluate_instrument(
    catalog:  ParquetDataCatalog,
    inst_cfg: dict,
) -> None:
    """Run the full diagnostic suite for a single instrument."""
    symbol  = inst_cfg["binance_symbol"]
    ticker  = inst_cfg["ticker"]
    out_dir = cfg.RESULTS_DIR / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    diag    = DiagnosticEngine(out_dir)

    logger.info("=" * 60)
    logger.info("Evaluating  %s  →  %s", symbol, out_dir)
    logger.info("=" * 60)

    hmm_path = inst_cfg["hmm_model_path"]
    if not hmm_path.exists():
        logger.error("[%s] HMM not found at %s. Run train_models.py.", symbol, hmm_path)
        return

    hmm_obj = RegimeHMM.load(hmm_path)

    logger.info(
        "[%s] Loading OOS data (%s → %s)…",
        symbol, cfg.BACKTEST_START, cfg.BACKTEST_END,
    )
    data    = _load_arrays(catalog, inst_cfg["bar_type_str"], cfg.BACKTEST_START, cfg.BACKTEST_END)
    closes  = data["closes"]
    volumes = data["volumes"]

    if len(closes) < cfg.HMM_MIN_HISTORY + 100:
        logger.error(
            "[%s] Only %d bars; need ≥ %d.",
            symbol, len(closes), cfg.HMM_MIN_HISTORY + 100,
        )
        return

    # ── Batch inference — visualisation plots only ────────────────────────────
    # Uses the full forward-backward algorithm (smoothed estimate).
    # Do NOT use these arrays for the simulation or calibration chart.
    logger.info("[%s] Running batch HMM inference for visualisation…", symbol)
    batch_regimes      = hmm_obj.predict(closes, volumes)
    batch_proba_matrix = hmm_obj.predict_proba(closes, volumes)
    ll                 = hmm_obj.log_likelihood(closes, volumes)

    logger.info("[%s] OOS log-likelihood: %.2f", symbol, ll)
    regime_counts = {
        _REGIME_NAMES[r]: int((batch_regimes == r).sum())
        for r in range(cfg.HMM_N_STATES)
    }
    logger.info("[%s] Regime distribution: %s", symbol, regime_counts)

    diag.plot_regime_overlay(
        closes, batch_regimes,
        title=f"OOS Regime Detection — {symbol}",
        filename=f"regime_overlay_{ticker}.png",
    )
    diag.plot_transition_matrix(
        hmm_obj.get_transition_matrix(),
        filename=f"transition_matrix_{ticker}.png",
    )
    diag.plot_regime_probabilities(
        proba_matrix=batch_proba_matrix,
        prices=closes,
        entry_threshold=cfg.MIN_BULL_PROBA,
        bear_threshold=cfg.BEAR_EXIT_PROBA,
        filename=f"regime_probabilities_{ticker}.png",
    )

    # Regime distribution bar chart
    n      = len(batch_regimes)
    fig, ax = plt.subplots(figsize=(8, 4))
    names  = [_REGIME_NAMES[r] for r in range(cfg.HMM_N_STATES)]
    counts = [int((batch_regimes == r).sum()) for r in range(cfg.HMM_N_STATES)]
    colors = [_REGIME_COLORS[r] for r in range(cfg.HMM_N_STATES)]
    bars   = ax.bar(names, counts, color=colors, edgecolor="k", alpha=0.85)
    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f"{c}\n({100*c/max(n,1):.1f}%)",
            ha="center", fontsize=9,
        )
    ax.set_title(f"Regime Distribution — {symbol} (OOS)")
    ax.set_ylabel("Bars")
    plt.tight_layout()
    plt.savefig(out_dir / f"regime_distribution_{ticker}.png", dpi=150)
    plt.close(fig)

    # ── Causal inference — simulation and calibration ─────────────────────────
    logger.info(
        "[%s] Computing causal (rolling-window) posteriors for simulation…",
        symbol,
    )
    causal_bull, causal_bear, causal_regimes = _compute_causal_proba(
        hmm_obj     = hmm_obj,
        closes      = closes,
        volumes     = volumes,
        window      = 300,
        min_history = cfg.HMM_MIN_HISTORY,
    )

    # ── Kelly sizer ───────────────────────────────────────────────────────────
    kelly = KellyCriterion(
        fraction         = cfg.KELLY_FRACTION,
        max_position_pct = cfg.MAX_POSITION_PCT,
        take_profit_pct  = cfg.TAKE_PROFIT_PCT,
        trail_entry_pct  = cfg.TRAIL_BULL_PCT,
        commission_rate  = cfg.COMMISSION_RATE,
    )
    logger.info(
        "[%s] Kelly | break-even P(bull)=%.3f  b=%.3f",
        symbol, kelly.breakeven_prob, kelly.win_loss_ratio,
    )

    # ── Kelly distribution (BULL bars, causal proba) ──────────────────────────
    bull_mask   = causal_regimes == BULL
    bull_probs_ = causal_bull[bull_mask]
    kelly_fracs = np.array([kelly.position_fraction(p) for p in bull_probs_])

    if len(kelly_fracs) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(kelly_fracs * 100, bins=30,
                color="#42a5f5", edgecolor="k", alpha=0.85)
        ax.axvline(np.mean(kelly_fracs) * 100, color="red", ls="--",
                   label=f"Mean = {np.mean(kelly_fracs)*100:.1f}%")
        ax.set_title(f"Kelly Position Size Distribution (BULL bars) — {symbol}")
        ax.set_xlabel("Position Size (% of portfolio)")
        ax.set_ylabel("Frequency")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"kelly_distribution_{ticker}.png", dpi=150)
        plt.close(fig)
        logger.info(
            "[%s] Kelly | mean=%.2f%%  min=%.2f%%  max=%.2f%%",
            symbol,
            np.mean(kelly_fracs) * 100,
            np.min(kelly_fracs)  * 100,
            np.max(kelly_fracs)  * 100,
        )

    # ── Multi-horizon P(Bull) calibration charts ──────────────────────────────
    # Three charts: 1-bar (15 min), 24-bar (6h), 96-bar (24h).
    #
    # How to read these charts
    # -------------------------
    # A working signal shows a clear upward slope: higher P(Bull) → higher
    # mean forward return.  The entry threshold line (0.55) should be to the
    # left of where the bars reliably turn positive.
    #
    # If the 1-bar chart is flat but the 96-bar chart slopes upward, the HMM
    # detects genuine multi-day regimes but not 15-min momentum — which is
    # expected and acceptable.  The strategy holding period should match the
    # horizon where the slope appears.
    #
    # If ALL horizons are flat or inverted, the state labeling or features
    # need to be fixed before any strategy-level tuning is meaningful.
    aligned_closes = closes[1:]

    # 1-bar forward return (existing)
    fwd_1 = np.diff(np.log(aligned_closes + 1e-9))
    n_1   = min(len(causal_bull) - 1, len(fwd_1))

    _plot_calibration(
        causal_bull   = causal_bull,
        forward_ret   = fwd_1,
        n_valid       = n_1,
        horizon_label = "1-bar (15 min)",
        symbol        = symbol,
        out_dir       = out_dir,
        ticker        = ticker,
        filename_tag  = "1bar",
    )

    # 6-hour (24-bar) and 24-hour (96-bar) forward returns
    for horizon, tag, label in [
        (24,  "24bar",  "24-bar (6 h)"),
        (96,  "96bar",  "96-bar (24 h)"),
    ]:
        if len(aligned_closes) <= horizon:
            continue
        fwd_h = np.array([
            np.log((aligned_closes[i + horizon] + 1e-9)
                   / (aligned_closes[i]          + 1e-9))
            for i in range(len(aligned_closes) - horizon)
        ], dtype=np.float64)
        n_h = min(len(causal_bull) - horizon, len(fwd_h))

        _plot_calibration(
            causal_bull   = causal_bull,
            forward_ret   = fwd_h,
            n_valid       = n_h,
            horizon_label = label,
            symbol        = symbol,
            out_dir       = out_dir,
            ticker        = ticker,
            filename_tag  = tag,
        )

    # ── Per-regime return statistics (causal regimes) ─────────────────────────
    logger.info("[%s] Per-regime 1-bar forward return statistics (causal):", symbol)
    for r in range(cfg.HMM_N_STATES):
        mask = causal_regimes[:n_1] == r
        if mask.sum() > 0:
            rets = fwd_1[:n_1][mask]
            logger.info(
                "[%s]   %-8s | n=%5d | mean=%+.5f | std=%.5f | win_rate=%.3f",
                symbol, _REGIME_NAMES[r], int(mask.sum()),
                float(rets.mean()), float(rets.std()),
                float((rets > 0).mean()),
            )

    # ── Simulation (causal proba + all strategy guards) ───────────────────────
    logger.info(
        "[%s] Running causal simulation "
        "(EMA=%d bars, bear_consec=%d, bull_consec=%d, lookback=%d, max_hold=%d)…",
        symbol,
        cfg.TREND_EMA_BARS, cfg.BEAR_EXIT_CONSECUTIVE,
        cfg.BULL_ENTRY_CONSECUTIVE, cfg.TREND_LOOKBACK_BARS,
        cfg.MAX_HOLDING_BARS,
    )
    equity_curve, trade_df = _simulate_pnl(
        aligned_closes         = aligned_closes,
        bull_proba             = causal_bull,
        bear_proba             = causal_bear,
        regimes                = causal_regimes,
        kelly                  = kelly,
        ema_bars               = cfg.TREND_EMA_BARS,
        bear_exit_consecutive  = cfg.BEAR_EXIT_CONSECUTIVE,
        bull_entry_consecutive = cfg.BULL_ENTRY_CONSECUTIVE,
        trend_lookback_bars    = cfg.TREND_LOOKBACK_BARS,
        max_holding_bars       = cfg.MAX_HOLDING_BARS,
    )

    logger.info(
        "[%s] Simulation | trades=%d | total_return=%.4f",
        symbol, len(trade_df), float(equity_curve[-1] - 1.0),
    )

    if len(trade_df) > 0:
        logger.info("[%s] Performance metrics:", symbol)
        diag.compute_strategy_metrics(equity_curve=equity_curve, trades=trade_df)

        if "reason" in trade_df.columns:
            logger.info(
                "[%s] Exit reasons:\n%s",
                symbol, trade_df["reason"].value_counts().to_string(),
            )

    # ── Equity curve plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(equity_curve, lw=1.5, color="#1565c0", label="HMM Strategy (causal)")
    ax.axhline(1.0, ls="--", color="gray", lw=1, label="Starting value")
    ax.fill_between(
        range(len(equity_curve)), equity_curve, 1.0,
        where=equity_curve >= 1.0, alpha=0.15, color="green",
    )
    ax.fill_between(
        range(len(equity_curve)), equity_curve, 1.0,
        where=equity_curve < 1.0, alpha=0.15, color="red",
    )
    ax.set_title(
        f"Simulated Equity Curve (normalised) — {symbol} (OOS, causal inference)\n"
        f"Kelly={cfg.KELLY_FRACTION:.0%}  EMA={cfg.TREND_EMA_BARS}bars  "
        f"BearConsec={cfg.BEAR_EXIT_CONSECUTIVE}"
    )
    ax.set_xlabel("Bar")
    ax.set_ylabel("Portfolio Value (normalised)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"equity_curve_{ticker}.png", dpi=150)
    plt.close(fig)

    logger.info("[%s] All outputs saved → %s", symbol, out_dir)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    catalog = ParquetDataCatalog(cfg.CATALOG_PATH)
    symbols = [i["binance_symbol"] for i in cfg.INSTRUMENTS]
    logger.info("Evaluating %d instrument(s): %s", len(cfg.INSTRUMENTS), symbols)

    for inst_cfg in cfg.INSTRUMENTS:
        evaluate_instrument(catalog, inst_cfg)

    logger.info("All evaluations complete.  Results: %s", cfg.RESULTS_DIR)


if __name__ == "__main__":
    main()
"""
utils/diagnostics.py
Diagnostic and performance-metric tools for the HMM regime strategy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

REGIME_NAMES  = ["BEAR", "SIDEWAYS", "BULL"]
REGIME_COLORS = {0: "#ef5350", 1: "#ffa726", 2: "#66bb6a"}   # BEAR=0, SIDEWAYS=1, BULL=2


class DiagnosticEngine:
    """
    Diagnostic plots and performance metrics for the HMM-only trading system.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── HMM regime plots ──────────────────────────────────────────────────────

    def plot_regime_overlay(
        self,
        prices:   np.ndarray,
        regimes:  np.ndarray,
        title:    str = "HMM Regime Detection",
        filename: str = "regime_overlay.png",
    ) -> None:
        """Price chart colour-coded by regime."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

        idx    = np.arange(len(regimes))
        colors = [REGIME_COLORS[r] for r in regimes]

        ax1.plot(prices[1:], lw=0.7, color="#1565c0", zorder=2)
        ax1.scatter(idx, prices[1:], c=colors, s=1.5, zorder=3)
        ax1.set_ylabel("Price (USD)")
        ax1.set_title(title)

        ax2.plot(regimes, drawstyle="steps-post", lw=0.9, color="#333")
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(REGIME_NAMES)
        ax2.set_ylabel("Regime")

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close(fig)
        logger.info("Saved regime overlay → %s", filename)

    def plot_transition_matrix(
        self,
        T:        np.ndarray,
        filename: str = "transition_matrix.png",
    ) -> None:
        """Heatmap of the HMM transition probability matrix."""
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            T,
            annot=True, fmt=".3f",
            xticklabels=REGIME_NAMES,
            yticklabels=REGIME_NAMES,
            cmap="Blues", ax=ax,
        )
        ax.set_title("HMM Transition Matrix")
        ax.set_xlabel("To State")
        ax.set_ylabel("From State")
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close(fig)
        logger.info("Saved transition matrix → %s", filename)

    def plot_regime_probabilities(
        self,
        proba_matrix:    np.ndarray,
        prices:          np.ndarray,
        entry_threshold: float = 0.45,
        bear_threshold:  float = 0.40,
        filename:        str   = "regime_probabilities.png",
    ) -> None:
        """
        4-panel plot: price, P(Bear), P(Sideways), P(Bull) over time.

        Column layout of proba_matrix matches hmm_model.py constants:
            column 0  →  P(Bear)     (BEAR     = 0)
            column 1  →  P(Sideways) (SIDEWAYS = 1)
            column 2  →  P(Bull)     (BULL     = 2)

        The bear-exit threshold is shown on the P(Bear) panel and the entry
        threshold is shown on the P(Bull) panel, matching the strategy logic.

        Bug fix: previous version used labels ["P(Bull)", "P(Sideways)", "P(Bear)"]
        which mapped the BEAR column (0) to "P(Bull)" and vice versa, causing every
        diagnostic reader to draw exactly the wrong conclusions about regime activity.
        """
        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
        n   = proba_matrix.shape[0]
        idx = np.arange(n)

        # Panel 1 — price
        axes[0].plot(prices[1 : n + 1], lw=0.7, color="#1565c0")
        axes[0].set_ylabel("Price (USD)")
        axes[0].set_title("Price")

        # Panels 2-4 — regime posteriors in correct column order.
        # col 0 = P(Bear), col 1 = P(Sideways), col 2 = P(Bull)
        # ↓ FIXED: was ["P(Bull)", "P(Sideways)", "P(Bear)"] — reversed first/last
        labels     = ["P(Bear)", "P(Sideways)", "P(Bull)"]
        # ↓ FIXED: thresholds now follow the correct column assignments
        thresholds = [
            (bear_threshold,  "darkred",   f"Exit ≥ {bear_threshold}"),   # col 0 = P(Bear)
            (None, None, None),                                             # col 1 = P(Sideways)
            (entry_threshold, "darkgreen", f"Entry ≥ {entry_threshold}"), # col 2 = P(Bull)
        ]

        for ax, col, label, (thresh, tc, tl) in zip(
            axes[1:], range(3), labels, thresholds
        ):
            ax.fill_between(
                idx, proba_matrix[:, col],
                alpha=0.75, color=REGIME_COLORS[col], label=label,
            )
            ax.set_ylabel(label)
            ax.set_ylim(0, 1)
            if thresh is not None:
                ax.axhline(thresh, ls="--", color=tc, lw=1.2, label=tl)
            ax.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close(fig)
        logger.info("Saved regime probability time series → %s", filename)

    # ── Strategy-level performance metrics ────────────────────────────────────

    def compute_strategy_metrics(
        self,
        equity_curve:  np.ndarray | pd.Series,
        trades:        pd.DataFrame | None = None,
        bars_per_year: int   = 252 * 24 * 4,   # 15-min bars, 24/7
        risk_free:     float = 0.0,
    ) -> dict:
        """
        Compute Sharpe, Sortino, Calmar, max-drawdown and trade statistics.
        """
        eq   = np.array(equity_curve, dtype=np.float64)
        rets = np.diff(eq) / (eq[:-1] + 1e-12)

        mean_r  = rets.mean()
        std_r   = rets.std() + 1e-12
        sharpe  = (mean_r - risk_free) / std_r * np.sqrt(bars_per_year)

        down_r  = rets[rets < 0]
        sort_d  = down_r.std() + 1e-12 if len(down_r) > 1 else 1e-12
        sortino = (mean_r - risk_free) / sort_d * np.sqrt(bars_per_year)

        peak   = np.maximum.accumulate(eq)
        dd     = (eq - peak) / (peak + 1e-12)
        max_dd = float(dd.min())

        total_return  = float(eq[-1] / eq[0] - 1.0)
        n_bars        = len(eq)
        annual_factor = bars_per_year / max(n_bars, 1)
        cagr          = (1.0 + total_return) ** annual_factor - 1.0
        calmar        = cagr / (abs(max_dd) + 1e-12)

        metrics: dict = {
            "total_return": total_return,
            "cagr":         cagr,
            "sharpe":       sharpe,
            "sortino":      sortino,
            "calmar":       calmar,
            "max_drawdown": max_dd,
        }

        if trades is not None and len(trades) > 0 and "pnl" in trades.columns:
            wins  = trades[trades["pnl"] > 0]
            loss  = trades[trades["pnl"] <= 0]
            metrics["n_trades"]      = len(trades)
            metrics["win_rate"]      = len(wins) / len(trades)
            metrics["profit_factor"] = (
                wins["pnl"].sum() / (abs(loss["pnl"].sum()) + 1e-12)
            )
            metrics["avg_win"]  = float(wins["pnl"].mean()) if len(wins)  > 0 else 0.0
            metrics["avg_loss"] = float(loss["pnl"].mean()) if len(loss)  > 0 else 0.0

        for k, v in metrics.items():
            logger.info("  %-22s %+.4f", k, v)

        return metrics
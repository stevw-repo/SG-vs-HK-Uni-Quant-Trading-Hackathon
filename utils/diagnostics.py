"""
utils/diagnostics.py
Diagnostic and evaluation tools for the HMM-LSTM pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

if TYPE_CHECKING:
    from models.hmm_model import RegimeHMM
    from models.lstm_model import DirectionalLSTM

logger = logging.getLogger(__name__)

REGIME_NAMES = ["BEAR", "SIDEWAYS", "BULL"]


class DiagnosticEngine:
    """
    Collects and plots diagnostics for both the HMM and LSTM models,
    and computes strategy-level metrics (Sharpe, win-rate, etc.).
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── HMM diagnostics ──────────────────────────────────────────────────────

    def plot_regime_overlay(
        self,
        prices:   np.ndarray,
        regimes:  np.ndarray,
        title:    str = "HMM Regime Detection",
        filename: str = "regime_overlay.png",
    ) -> None:
        """Overlay regime colours on price chart."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

        idx    = np.arange(len(regimes))
        cmap   = {0: "#ef5350", 1: "#ffa726", 2: "#66bb6a"}  # red/orange/green
        colors = [cmap[r] for r in regimes]

        ax1.plot(prices[1:], lw=0.8, color="#1565c0")
        ax1.scatter(idx, prices[1:], c=colors, s=2, zorder=3)
        ax1.set_ylabel("Price (USD)")
        ax1.set_title(title)

        ax2.plot(regimes, drawstyle="steps-post", lw=1, color="#333")
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(REGIME_NAMES)
        ax2.set_ylabel("Regime")

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close(fig)
        logger.info("Saved regime overlay → %s", filename)

    def plot_transition_matrix(
        self,
        T: np.ndarray,
        filename: str = "transition_matrix.png",
    ) -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            T,
            annot=True,
            fmt=".3f",
            xticklabels=REGIME_NAMES,
            yticklabels=REGIME_NAMES,
            cmap="Blues",
            ax=ax,
        )
        ax.set_title("HMM Transition Matrix")
        ax.set_xlabel("To")
        ax.set_ylabel("From")
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close(fig)
        logger.info("Saved transition matrix → %s", filename)

    # ── LSTM diagnostics ──────────────────────────────────────────────────────

    def plot_training_history(
        self,
        history:  dict,
        filename: str = "lstm_training_history.png",
    ) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

        ax1.plot(history.get("loss", []),     label="Train loss")
        ax1.plot(history.get("val_loss", []), label="Val loss")
        ax1.set_title("LSTM Loss")
        ax1.legend()

        ax2.plot(history.get("val_acc", []), color="green")
        ax2.axhline(0.5, ls="--", color="gray")
        ax2.set_title("Val Accuracy")

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close(fig)
        logger.info("Saved training history → %s", filename)

    def plot_lstm_accuracy_by_regime(
        self,
        true_labels:  np.ndarray,
        pred_proba:   np.ndarray,
        regimes:      np.ndarray,
        filename:     str = "lstm_accuracy_by_regime.png",
    ) -> None:
        """Bar chart of LSTM accuracy broken down by HMM regime."""
        pred = (pred_proba > 0.5).astype(int)
        fig, ax = plt.subplots(figsize=(8, 4))
        accs, counts = [], []
        for r, name in enumerate(REGIME_NAMES):
            mask = regimes == r
            if mask.sum() == 0:
                accs.append(0.0)
                counts.append(0)
                continue
            acc = (pred[mask] == true_labels[mask]).mean()
            accs.append(acc)
            counts.append(mask.sum())

        bars = ax.bar(REGIME_NAMES, accs, color=["#ef5350", "#ffa726", "#66bb6a"])
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"n={c}", ha="center", fontsize=9)
        ax.axhline(0.5, ls="--", color="gray", label="Random baseline")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title("LSTM Directional Accuracy by Regime")
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close(fig)
        logger.info("Saved accuracy-by-regime → %s", filename)

    # ── Strategy-level metrics ────────────────────────────────────────────────

    def compute_strategy_metrics(
        self,
        equity_curve: np.ndarray | pd.Series,
        trades:       pd.DataFrame | None = None,
        risk_free:    float = 0.0,
    ) -> dict:
        """
        Compute and log key performance metrics.

        Returns a dict with:
            total_return, cagr, sharpe, sortino, max_drawdown,
            win_rate, profit_factor  (last two only if trades provided)
        """
        eq = np.array(equity_curve, dtype=np.float64)
        rets = np.diff(eq) / eq[:-1]

        total_return  = (eq[-1] / eq[0]) - 1
        ann_factor    = 252 * 24 * 60          # 1-min bars per year
        mean_r        = rets.mean()
        std_r         = rets.std() + 1e-12
        sharpe        = (mean_r - risk_free / ann_factor) / std_r * np.sqrt(ann_factor)

        down_r        = rets[rets < 0]
        sortino_denom = down_r.std() + 1e-12
        sortino       = (mean_r - risk_free / ann_factor) / sortino_denom * np.sqrt(ann_factor)

        running_max   = np.maximum.accumulate(eq)
        drawdowns     = (eq - running_max) / running_max
        max_dd        = drawdowns.min()

        metrics = {
            "total_return": total_return,
            "sharpe":       sharpe,
            "sortino":      sortino,
            "max_drawdown": max_dd,
        }

        if trades is not None and len(trades) > 0:
            wins    = trades[trades["pnl"] > 0]
            losses  = trades[trades["pnl"] <= 0]
            metrics["win_rate"]      = len(wins) / len(trades)
            metrics["profit_factor"] = (
                wins["pnl"].sum() / (abs(losses["pnl"].sum()) + 1e-12)
            )
            metrics["n_trades"] = len(trades)

        for k, v in metrics.items():
            logger.info("  %-20s %+.4f", k, v)

        return metrics

    def walk_forward_evaluation(
        self,
        closes:     np.ndarray,
        hmm:        "RegimeHMM",
        lstm:       "DirectionalLSTM",
        highs:      np.ndarray | None = None,
        lows:       np.ndarray | None = None,
        volumes:    np.ndarray | None = None,
        fear_greed: np.ndarray | None = None,
        window:     int = 3000,
        step:       int = 500,
        min_conf:   float = 0.53,
    ) -> pd.DataFrame:
        """
        Slide a window over the data and record signal accuracy.
        Returns a DataFrame with per-window metrics.
        """
        records = []
        max_history = hmm.n_states * 100  # rough warm-up

        for start in range(max_history, len(closes) - window, step):
            end = start + window

            chunk_closes  = closes[start:end]
            chunk_highs   = highs[start:end]   if highs      is not None else None
            chunk_lows    = lows[start:end]     if lows       is not None else None
            chunk_volumes = volumes[start:end]  if volumes    is not None else None
            chunk_fg      = fear_greed[start:end] if fear_greed is not None else None

            try:
                # Pass volumes to HMM to match the signature used in training
                if chunk_volumes is not None:
                    regimes = hmm.predict(chunk_closes, chunk_volumes)
                else:
                    regimes = hmm.predict(chunk_closes)

                n        = len(regimes)
                p_up_arr = []

                for i in range(lstm.lookback, n - lstm.prediction_horizon):
                    p = lstm.predict_proba(
                        closes=chunk_closes[:i + 1],
                        highs=chunk_highs[:i + 1]   if chunk_highs   is not None else None,
                        lows=chunk_lows[:i + 1]     if chunk_lows    is not None else None,
                        volumes=chunk_volumes[:i + 1] if chunk_volumes is not None else None,
                        hmm_regimes=regimes[:i + 1],
                        fear_greed=chunk_fg[:i + 1] if chunk_fg      is not None else None,
                    )
                    p_up_arr.append(p)

                p_up_arr = np.array(p_up_arr)
                lbs_arr  = np.array([
                    1.0 if chunk_closes[i + lstm.prediction_horizon] > chunk_closes[i] else 0.0
                    for i in range(lstm.lookback, n - lstm.prediction_horizon)
                ])

                mask = p_up_arr > min_conf
                if mask.sum() == 0:
                    continue
                acc = ((p_up_arr[mask] > 0.5) == (lbs_arr[mask] > 0.5)).mean()
                records.append({
                    "window_start": start,
                    "window_end":   end,
                    "n_signals":    int(mask.sum()),
                    "accuracy":     float(acc),
                    "mean_p_up":    float(p_up_arr[mask].mean()),
                })
            except Exception as exc:
                logger.warning("Walk-forward window [%d:%d] failed: %s", start, end, exc)

        df = pd.DataFrame(records)
        logger.info("Walk-forward evaluation | %d windows | mean acc=%.3f",
                    len(df), df["accuracy"].mean() if len(df) else 0.0)
        return df

    def print_classification_report(
        self,
        true_labels: np.ndarray,
        pred_proba:  np.ndarray,
    ) -> None:
        pred = (pred_proba > 0.5).astype(int)
        print(classification_report(true_labels, pred, target_names=["DOWN", "UP"]))

        cm = confusion_matrix(true_labels, pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"])
        ax.set_title("LSTM Confusion Matrix")
        plt.tight_layout()
        path = self.output_dir / "lstm_confusion_matrix.png"
        plt.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved confusion matrix → %s", path)
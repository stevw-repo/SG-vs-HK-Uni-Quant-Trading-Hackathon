"""
utils/kelly_criterion.py
Commission-adjusted Kelly Criterion sized from HMM regime probabilities.

Derivation
----------
For a binary outcome trade on a long-only market order:

  net_win  = take_profit_pct  − 2 × commission_rate   (enter + exit)
  net_loss = trail_entry_pct  + 2 × commission_rate   (initial stop at entry)
  b        = net_win / net_loss                        (win/loss odds ratio)

  Kelly*   = (p_win × b − p_lose) / b                 (full Kelly fraction)
           where p_win = P(bull) from HMM posterior
                 p_lose = 1 − P(bull)

  f        = max(0, Kelly* × fraction)                 (fractional Kelly)
  f        = min(f, max_position_pct)                  (position cap)

Break-even probability (Kelly* = 0):
  p_break  = 1 / (b + 1)

Example — trail=2 %, take_profit=4 %, commission=0.1 %:
  b        = (4% − 0.2%) / (2% + 0.2%) = 3.8/2.2 ≈ 1.727
  p_break  = 1/2.727 ≈ 0.367

Setting min_bull_proba=0.45 ensures we only trade when P(bull) is well above
the break-even probability, giving a genuine positive expected value after fees.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class KellyCriterion:
    """
    Commission-aware fractional Kelly position sizer.

    Parameters
    ----------
    fraction         : Fractional Kelly multiplier (0 < f ≤ 1).
    max_position_pct : Hard cap on position size as a fraction of capital.
    take_profit_pct  : Target gain before round-trip commission.
    trail_entry_pct  : Trailing stop distance at entry — the initial risk.
    commission_rate  : One-way commission rate (0.001 = 0.1 %).
    """

    def __init__(
        self,
        fraction:         float = 0.50,
        max_position_pct: float = 0.70,
        take_profit_pct:  float = 0.040,
        trail_entry_pct:  float = 0.020,
        commission_rate:  float = 0.001,
    ) -> None:
        self.fraction         = fraction
        self.max_position_pct = max_position_pct

        # Commission-adjusted win and loss amounts
        self._net_win  = max(1e-6, take_profit_pct - 2.0 * commission_rate)
        self._net_loss = max(1e-6, trail_entry_pct  + 2.0 * commission_rate)
        self._b        = self._net_win / self._net_loss
        self._p_break  = 1.0 / (self._b + 1.0)

        logger.info(
            "KellyCriterion | net_win=%.4f  net_loss=%.4f  b=%.4f  "
            "p_breakeven=%.4f  fraction=%.2f  cap=%.2f",
            self._net_win, self._net_loss, self._b,
            self._p_break, fraction, max_position_pct,
        )

    # ── Core formula ──────────────────────────────────────────────────────────

    def full_kelly(self, bull_prob: float) -> float:
        """
        Full (un-capped) Kelly fraction for a given P(bull).
        Returns 0.0 when bull_prob is below the break-even probability.
        """
        p = float(bull_prob)
        return max(0.0, (p * self._b - (1.0 - p)) / self._b)

    def position_fraction(self, bull_prob: float) -> float:
        """
        Fractional Kelly position size, capped at max_position_pct.

        Parameters
        ----------
        bull_prob : P(bull) from the HMM posterior — used directly as p_win.

        Returns
        -------
        float in [0, max_position_pct]
        """
        fk = self.full_kelly(bull_prob) * self.fraction
        fk = min(fk, self.max_position_pct)
        logger.debug(
            "Kelly | bull_prob=%.3f  full_kelly=%.3f  → fraction=%.3f",
            bull_prob, self.full_kelly(bull_prob), fk,
        )
        return fk

    def size_in_base_currency(
        self,
        bull_prob:     float,
        portfolio_usd: float,
        price:         float,
        precision:     int = 6,
    ) -> float:
        """
        Convert Kelly fraction → quantity of the base currency to buy.

        Parameters
        ----------
        bull_prob     : P(bull) from HMM.
        portfolio_usd : Available USD capital.
        price         : Current price of the base asset in USD.
        precision     : Decimal rounding precision for the returned quantity.

        Returns
        -------
        float : quantity to buy (0.0 when no positive edge).
        """
        frac  = self.position_fraction(bull_prob)
        units = (frac * portfolio_usd) / max(price, 1e-12)
        return round(units, precision)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def breakeven_prob(self) -> float:
        """Minimum P(bull) for a positive expected value after commission."""
        return self._p_break

    @property
    def win_loss_ratio(self) -> float:
        """Commission-adjusted win/loss ratio (b)."""
        return self._b
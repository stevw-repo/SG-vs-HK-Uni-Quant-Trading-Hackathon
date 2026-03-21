"""
utils/kelly_criterion.py
Kelly Criterion position sizing with regime-based multipliers.
"""

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

# Regime constants (must match hmm_model.py)
BEAR, SIDEWAYS, BULL = 0, 1, 2


class KellyCriterion:
    """
    Fractional Kelly position sizer.

    Parameters
    ----------
    fraction         : float  Fraction of full Kelly to use (0 < f ≤ 1)
    win_loss_ratio   : float  Expected win / loss ratio (b in Kelly formula)
    max_position_pct : float  Hard cap on position as fraction of portfolio
    regime_multipliers : dict Multiply Kelly by this factor per regime
    """

    def __init__(
        self,
        fraction:          float = 0.40,
        win_loss_ratio:    float = 2.67,
        max_position_pct:  float = 0.65,
        regime_multipliers: dict | None = None,
    ) -> None:
        self.fraction          = fraction
        self.win_loss_ratio    = win_loss_ratio
        self.max_position_pct  = max_position_pct
        self.regime_multipliers = regime_multipliers or {
            BEAR:     0.0,   # No position in bear regime
            SIDEWAYS: 0.6,   # Reduced position in sideways
            BULL:     1.2,   # Slight boost in bull regime (still capped)
        }

    # ── Core formula ─────────────────────────────────────────────────────────

    def full_kelly(self, p_win: float) -> float:
        """
        Classic Kelly formula:
            f* = (p * b - q) / b
        where b = win/loss ratio, q = 1 - p.
        """
        b = self.win_loss_ratio
        q = 1.0 - p_win
        f = (p_win * b - q) / b
        return max(0.0, f)

    def position_fraction(
        self,
        p_win:  float,
        regime: int = BULL,
    ) -> float:
        """
        Return the fraction of portfolio to deploy.

        Steps
        -----
        1. Compute full Kelly f*
        2. Apply fractional Kelly
        3. Apply regime multiplier
        4. Clip to [0, max_position_pct]
        """
        if p_win <= 0.5:
            return 0.0

        fk  = self.full_kelly(p_win)
        fk *= self.fraction
        fk *= self.regime_multipliers.get(regime, 1.0)
        fk  = min(fk, self.max_position_pct)
        fk  = max(0.0, fk)

        logger.debug(
            "Kelly | p_win=%.3f regime=%d → full_kelly=%.3f → final=%.3f",
            p_win, regime, self.full_kelly(p_win), fk,
        )
        return fk

    def size_in_base_currency(
        self,
        p_win:          float,
        regime:         int,
        portfolio_usd:  float,
        price_per_unit: float,
        unit_precision: int = 6,
    ) -> float:
        """
        Convert fraction → units of base currency (e.g. BTC).

        Returns
        -------
        float : quantity to buy, rounded to unit_precision decimal places.
        """
        frac       = self.position_fraction(p_win, regime)
        usd_to_use = frac * portfolio_usd
        units      = usd_to_use / (price_per_unit + 1e-12)
        return round(units, unit_precision)
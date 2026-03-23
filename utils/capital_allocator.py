"""
utils/capital_allocator.py
Thread-safe capital reservation tracker for multi-strategy portfolios.

When multiple HMMStrategy instances share the same brokerage account, each
strategy calls available() to find how much capital remains after all other
strategies have reserved their positions.  This prevents the account from
being over-committed when two strategies fire entry signals on the same bar.

Usage
-----
    allocator = CapitalAllocator()

    # Strategy A — before entering:
    free = allocator.available("BTCUSDT.BINANCE", account_free_usd)
    allocator.reserve("BTCUSDT.BINANCE", committed_usd)

    # Strategy A — after position closes:
    allocator.release("BTCUSDT.BINANCE")
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class CapitalAllocator:
    """
    Tracks USD capital reservations across concurrent strategy instances.

    Each strategy is keyed by its instrument_id_str so reservations are
    unambiguous even when two strategies trade on the same venue.
    """

    def __init__(self) -> None:
        self._reservations: dict[str, float] = {}
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def reserve(self, strategy_id: str, amount_usd: float) -> None:
        """Record that strategy_id has committed amount_usd of capital."""
        with self._lock:
            self._reservations[strategy_id] = max(0.0, amount_usd)
            logger.debug(
                "Reserved $%.2f for %s | total_reserved=$%.2f",
                amount_usd, strategy_id, self._total_locked(),
            )

    def release(self, strategy_id: str) -> None:
        """Remove the reservation for strategy_id (position has closed)."""
        with self._lock:
            released = self._reservations.pop(strategy_id, 0.0)
            logger.debug(
                "Released $%.2f for %s | total_reserved=$%.2f",
                released, strategy_id, self._total_locked(),
            )

    def available(self, strategy_id: str, total_usd: float) -> float:
        """
        How much can strategy_id deploy given the current free account balance?

        Deducts all OTHER strategies' reservations.  This strategy's own prior
        reservation (if any) is excluded because we are about to replace it.

        Parameters
        ----------
        strategy_id : str    Caller's unique strategy identifier.
        total_usd   : float  Current free USD balance from the broker.

        Returns
        -------
        float ≥ 0 — amount available to this strategy.
        """
        with self._lock:
            other = sum(
                v for k, v in self._reservations.items() if k != strategy_id
            )
            return max(0.0, total_usd - other)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def total_reserved(self) -> float:
        with self._lock:
            return self._total_locked()

    def snapshot(self) -> dict[str, float]:
        """Diagnostic copy of the current reservations."""
        with self._lock:
            return dict(self._reservations)

    def _total_locked(self) -> float:
        return sum(self._reservations.values())
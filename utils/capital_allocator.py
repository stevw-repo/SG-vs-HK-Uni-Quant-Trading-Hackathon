"""
utils/capital_allocator.py
Shared capital reservation tracker for multiple concurrent strategy instances
that draw from the same account balance.
"""
from __future__ import annotations

import logging
from threading import Lock

logger = logging.getLogger(__name__)


class CapitalAllocator:
    """
    Prevents multiple strategies from committing the same dollars simultaneously.

    Each strategy calls available() to find out how much it can use, then
    reserve() immediately after submitting an entry order, and release() when
    the position closes.  The allocator subtracts every other strategy's live
    reservation from the real account free balance before returning the answer,
    so two strategies firing on the same bar each see a correctly reduced figure.

    Thread-safe via an internal Lock for live trading; in backtest the engine is
    single-threaded so the lock is a negligible no-op.
    """

    def __init__(self) -> None:
        self._reserved: dict[str, float] = {}   # strategy_id → committed USD
        self._lock = Lock()

    def available(self, strategy_id: str, account_free_usd: float) -> float:
        """
        Return the USD available to strategy_id.

        Parameters
        ----------
        strategy_id      : Unique key for the calling strategy instance.
        account_free_usd : Raw free balance from the account, as returned by
                           _estimate_portfolio_usd().  Other strategies'
                           reservations are subtracted from this figure.
        """
        with self._lock:
            other_reserved = sum(
                amt for sid, amt in self._reserved.items()
                if sid != strategy_id
            )
            result = max(0.0, account_free_usd - other_reserved)
            logger.debug(
                "[%s] available=%.2f  (account_free=%.2f  other_reserved=%.2f)",
                strategy_id, result, account_free_usd, other_reserved,
            )
            return result

    def reserve(self, strategy_id: str, usd_amount: float) -> None:
        """Record that strategy_id has committed usd_amount. Overwrites any
        previous reservation for this strategy."""
        with self._lock:
            self._reserved[strategy_id] = usd_amount
            logger.debug(
                "[%s] reserved=%.2f  (total_reserved=%.2f)",
                strategy_id, usd_amount, sum(self._reserved.values()),
            )

    def release(self, strategy_id: str) -> None:
        """Release the reservation held by strategy_id (position closed)."""
        with self._lock:
            released = self._reserved.pop(strategy_id, 0.0)
            logger.debug(
                "[%s] released=%.2f  (total_reserved=%.2f)",
                strategy_id, released, sum(self._reserved.values()),
            )

    @property
    def total_reserved(self) -> float:
        with self._lock:
            return sum(self._reserved.values())

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(self._reserved)
"""
utils/trade_journal.py

Append-only trade journal.  Writes are atomic (tmp → rename) so the
dashboard HTTP server never reads a partial file.

Usage
-----
    journal = TradeJournal(starting_balance=50_000.0)

    trade_id = journal.open_trade("BTCUSDT", 84200.0, 0.062,
                                   regime="BULL", bull_prob=0.71)
    journal.close_trade("BTCUSDT", 87150.0, exit_reason="TAKE_PROFIT")
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)
JOURNAL_PATH = Path("trade_journal.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TradeRecord:
    trade_id:    str
    symbol:      str
    entry_time:  str
    entry_price: float
    quantity:    float
    regime:      str   = "UNKNOWN"
    bull_prob:   float = 0.0
    exit_time:   str   = ""
    exit_price:  float = 0.0
    exit_reason: str   = ""
    pnl_pct:     float = 0.0
    pnl_usd:     float = 0.0
    open:        bool  = True


class TradeJournal:
    """
    Single shared journal instance across all strategy instances.

    Thread-safety note: all writes happen inside the single asyncio event
    loop thread; the only reader is the daemon HTTP server thread which
    reads the already-flushed JSON file, so no lock is needed.
    """

    def __init__(
        self,
        path:             Path  = JOURNAL_PATH,
        starting_balance: float = 1_000_000.0,
    ) -> None:
        self._path             = path
        self._starting_balance = starting_balance
        self._trades: list[TradeRecord] = []
        self._load()

    # ── Public API ──────────────────────────────────────────────────────────

    def open_trade(
        self,
        symbol:      str,
        entry_price: float,
        quantity:    float,
        regime:      str   = "UNKNOWN",
        bull_prob:   float = 0.0,
    ) -> str:
        """Record an entry. Returns a short trade_id."""
        trade_id = str(uuid4())[:8]
        self._trades.append(TradeRecord(
            trade_id    = trade_id,
            symbol      = symbol,
            entry_time  = _now_iso(),
            entry_price = entry_price,
            quantity    = quantity,
            regime      = regime,
            bull_prob   = bull_prob,
        ))
        self._save()
        return trade_id

    def close_trade(
        self,
        symbol:      str,
        exit_price:  float,
        exit_reason: str = "",
    ) -> None:
        """Mark the most-recent open trade for *symbol* as closed."""
        for t in reversed(self._trades):
            if t.symbol == symbol and t.open:
                t.exit_time   = _now_iso()
                t.exit_price  = exit_price
                t.exit_reason = exit_reason
                t.pnl_pct     = (exit_price - t.entry_price) / (t.entry_price + 1e-12) * 100
                t.pnl_usd     = (exit_price - t.entry_price) * t.quantity
                t.open        = False
                break
        self._save()

    def to_dict(self) -> dict:
        return {
            "starting_balance": self._starting_balance,
            "trades":           [asdict(t) for t in self._trades],
        }

    @property
    def open_trades(self) -> list[TradeRecord]:
        return [t for t in self._trades if t.open]

    # ── Internals ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw    = json.loads(self._path.read_text())
            fields = set(TradeRecord.__dataclass_fields__)
            self._trades = [
                TradeRecord(**{k: v for k, v in t.items() if k in fields})
                for t in raw.get("trades", [])
            ]
            logger.info(
                "TradeJournal: resumed %d trades from %s",
                len(self._trades), self._path,
            )
        except Exception as exc:
            logger.warning("TradeJournal: could not load %s: %s", self._path, exc)

    def _save(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(self.to_dict(), indent=2))
            tmp.replace(self._path)
        except Exception as exc:
            logger.warning("TradeJournal: save failed: %s", exc)
"""
strategy/roostoo_client.py
Synchronous HTTP client for the Roostoo mock-trading API.
Only exposes the three operations used by this project:
  - place_order  (BUY / SELL)
  - get_balance
  - cancel_order
  - query_order  (for diagnostics / fill confirmation)
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)


class RoostooClient:
    """
    Thread-safe Roostoo REST client.

    Authentication: HMAC-SHA256 over alphabetically sorted params,
    with RST-API-KEY and MSG-SIGNATURE in the HTTP headers.
    """

    def __init__(
        self,
        api_key:    str,
        secret_key: str,
        base_url:   str = "https://mock-api.roostoo.com",
        timeout:    int = 15,
    ) -> None:
        self._api_key    = api_key
        self._secret_key = secret_key.encode()
        self._base_url   = base_url.rstrip("/")
        self._timeout    = timeout
        self._session    = requests.Session()

    # ── Auth helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _timestamp() -> str:
        return str(int(time.time() * 1000))

    def _sign(self, params: dict) -> tuple[dict, str]:
        """
        Add timestamp, build sorted query string, compute HMAC-SHA256.
        Returns (signed_params, total_params_string).
        """
        params["timestamp"] = self._timestamp()
        total = "&".join(f"{k}={params[k]}" for k in sorted(params))
        sig   = hmac.new(self._secret_key, total.encode(), hashlib.sha256).hexdigest()
        return params, total, sig

    def _signed_headers(self, sig: str) -> dict:
        return {
            "RST-API-KEY":     self._api_key,
            "MSG-SIGNATURE":   sig,
            "Content-Type":    "application/x-www-form-urlencoded",
        }

    # ── Public API calls ──────────────────────────────────────────────────────

    def get_balance(self) -> dict[str, Any] | None:
        """
        GET /v3/balance
        Returns Roostoo wallet dict or None on error.
        """
        params, _, sig = self._sign({})
        try:
            r = self._session.get(
                f"{self._base_url}/v3/balance",
                headers=self._signed_headers(sig),
                params=params,
                timeout=self._timeout,
            )
            r.raise_for_status()
            data = r.json()
            if not data.get("Success"):
                logger.warning("get_balance: %s", data.get("ErrMsg"))
            return data
        except Exception as exc:
            logger.error("get_balance failed: %s", exc)
            return None

    def place_order(
        self,
        pair:       str,
        side:       str,
        quantity:   float,
        price:      float | None = None,
        order_type: str   | None = None,
    ) -> dict[str, Any] | None:
        """
        POST /v3/place_order
        side:       "BUY" | "SELL"
        order_type: "MARKET" | "LIMIT"  (auto-detected if None)
        price:      required only for LIMIT orders
        """
        if order_type is None:
            order_type = "LIMIT" if price is not None else "MARKET"
        if order_type == "LIMIT" and price is None:
            raise ValueError("price required for LIMIT orders")

        payload: dict[str, Any] = {
            "pair":     pair,
            "side":     side.upper(),
            "type":     order_type.upper(),
            "quantity": str(round(quantity, 6)),
        }
        if order_type == "LIMIT":
            payload["price"] = str(price)

        params, total_params, sig = self._sign(payload)

        try:
            r = self._session.post(
                f"{self._base_url}/v3/place_order",
                headers=self._signed_headers(sig),
                data=total_params,
                timeout=self._timeout,
            )
            r.raise_for_status()
            data = r.json()
            if data.get("Success"):
                detail = data.get("OrderDetail", {})
                logger.info(
                    "Order placed | id=%s status=%s side=%s qty=%s price=%s",
                    detail.get("OrderID"),
                    detail.get("Status"),
                    side, quantity,
                    detail.get("FilledAverPrice", "—"),
                )
            else:
                logger.error("place_order failed: %s", data.get("ErrMsg"))
            return data
        except Exception as exc:
            logger.error("place_order exception: %s", exc)
            return None

    def cancel_order(
        self,
        order_id: str | None = None,
        pair:     str | None = None,
    ) -> dict[str, Any] | None:
        """
        POST /v3/cancel_order
        Cancels by order_id, by pair, or ALL if both are None.
        """
        payload: dict[str, Any] = {}
        if order_id:
            payload["order_id"] = str(order_id)
        elif pair:
            payload["pair"] = pair

        params, total_params, sig = self._sign(payload)

        try:
            r = self._session.post(
                f"{self._base_url}/v3/cancel_order",
                headers=self._signed_headers(sig),
                data=total_params,
                timeout=self._timeout,
            )
            r.raise_for_status()
            data = r.json()
            if data.get("Success"):
                logger.info("Cancelled orders: %s", data.get("CanceledList"))
            else:
                logger.warning("cancel_order: %s", data.get("ErrMsg"))
            return data
        except Exception as exc:
            logger.error("cancel_order exception: %s", exc)
            return None

    def query_order(
        self,
        order_id:     str | None  = None,
        pair:         str | None  = None,
        pending_only: bool | None = None,
        limit:        int  | None = None,
    ) -> dict[str, Any] | None:
        """POST /v3/query_order — for diagnostics and fill confirmation."""
        payload: dict[str, Any] = {}
        if order_id:
            payload["order_id"] = str(order_id)
        elif pair:
            payload["pair"] = pair
        if pending_only is not None:
            payload["pending_only"] = "TRUE" if pending_only else "FALSE"
        if limit is not None:
            payload["limit"] = str(limit)

        params, total_params, sig = self._sign(payload)

        try:
            r = self._session.post(
                f"{self._base_url}/v3/query_order",
                headers=self._signed_headers(sig),
                data=total_params,
                timeout=self._timeout,
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.error("query_order exception: %s", exc)
            return None

    # ── Convenience helpers ───────────────────────────────────────────────────

    def free_usd(self) -> float:
        data = self.get_balance()
        if not data or not data.get("Success"):
            return 0.0
        return float(data["Wallet"].get("USD", {}).get("Free", 0.0))

    def free_btc(self) -> float:
        data = self.get_balance()
        if not data or not data.get("Success"):
            return 0.0
        return float(data["Wallet"].get("BTC", {}).get("Free", 0.0))

    def total_portfolio_usd(self, btc_price: float) -> float:
        data = self.get_balance()
        if not data or not data.get("Success"):
            return 0.0
        wallet = data["Wallet"]
        usd = float(wallet.get("USD", {}).get("Free", 0.0)) + \
              float(wallet.get("USD", {}).get("Lock", 0.0))
        btc = float(wallet.get("BTC", {}).get("Free", 0.0)) + \
              float(wallet.get("BTC", {}).get("Lock", 0.0))
        return usd + btc * btc_price
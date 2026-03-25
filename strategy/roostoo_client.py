"""
strategy/roostoo_client.py
Synchronous HTTP client for the Roostoo mock-trading API.
Only exposes the three operations used by this project:
  - place_order  (BUY / SELL)
  - get_balance
  - cancel_order
  - query_order  (for diagnostics / fill confirmation)

Fix 1 — quantity step size error
  Added get_exchange_info() and _load_amount_precision() so that place_order()
  rounds quantity to Roostoo's per-pair AmountPrecision (e.g. BNB/USD = 2 d.p.)
  instead of a hard-coded 6, preventing 'quantity step size error' rejections.

Fix 2 — _sign() return-type annotation corrected (tuple[dict, str] → tuple[dict, str, str])

Fix 3 — free_usd / free_btc / total_portfolio_usd now handle both the
  live-API 'SpotWallet' key and the documented 'Wallet' key defensively.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from typing import Any
import config as cfg
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
        api_key:    str = cfg.ROOSTOO_API_KEY,
        secret_key: str = cfg.ROOSTOO_SECRET_KEY,
        base_url:   str = "https://mock-api.roostoo.com",
        timeout:    int = 15,
    ) -> None:
        self._api_key    = api_key
        self._secret_key = secret_key.encode()
        self._base_url   = base_url.rstrip("/")
        self._timeout    = timeout
        self._session    = requests.Session()

        # Per-pair AmountPrecision fetched from /v3/exchangeInfo on init.
        # place_order() rounds quantity to this precision before sending.
        # Falls back to 6 for any pair not found in the response.
        self._amount_precision: dict[str, int] = {}
        self._load_amount_precision()

    # ── Auth helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _timestamp() -> str:
        return str(int(time.time() * 1000))

    def _sign(self, params: dict) -> tuple[dict, str, str]:
        """
        Add timestamp, build sorted query string, compute HMAC-SHA256.
        Returns (signed_params, total_params_string, signature).
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

    # ── Exchange info (public, no auth) ───────────────────────────────────────

    def get_exchange_info(self) -> dict[str, Any] | None:
        """
        GET /v3/exchangeInfo  (Auth: RCL_NoVerification)
        Returns trading rules including per-pair AmountPrecision and MiniOrder.
        No authentication required.
        """
        try:
            r = self._session.get(
                f"{self._base_url}/v3/exchangeInfo",
                timeout=self._timeout,
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.error("get_exchange_info failed: %s", exc)
            return None

    def _load_amount_precision(self) -> None:
        """
        Populate self._amount_precision from /v3/exchangeInfo.

        The Roostoo API enforces an AmountPrecision per pair — for example,
        BNB/USD allows only 2 decimal places.  Sending more decimals returns
        'quantity step size error'.  This method fetches the live rules once
        at construction time so place_order() can truncate correctly.

        Falls back silently to precision=6 for any pair whose info cannot
        be loaded (e.g. network error at startup).
        """
        info = self.get_exchange_info()
        if not info:
            logger.warning(
                "Could not load Roostoo exchangeInfo at startup — "
                "defaulting to AmountPrecision=6 for all pairs.  "
                "Orders may be rejected if the true precision is stricter."
            )
            return
        for pair, meta in info.get("TradePairs", {}).items():
            self._amount_precision[pair] = int(meta.get("AmountPrecision", 6))
        logger.info("Roostoo AmountPrecision per pair: %s", self._amount_precision)

    def _pair_amount_precision(self, pair: str) -> int:
        """Return the AmountPrecision for pair, defaulting to 6 if unknown."""
        return self._amount_precision.get(pair, 6)

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

        Quantity is rounded to the Roostoo-specified AmountPrecision for this
        pair before the request is sent, preventing 'quantity step size error'
        rejections that occur when too many decimal places are submitted.
        """
        if order_type is None:
            order_type = "LIMIT" if price is not None else "MARKET"
        if order_type == "LIMIT" and price is None:
            raise ValueError("price required for LIMIT orders")

        # ── Round to this pair's AmountPrecision ──────────────────────────────
        # e.g. BNB/USD: AmountPrecision=2 → round(620.607171, 2) = "620.61"
        #      BTC/USD: AmountPrecision=6 → round(3.392237, 6)   = "3.392237"
        prec    = self._pair_amount_precision(pair)
        qty_str = str(round(quantity, prec))

        payload: dict[str, Any] = {
            "pair":     pair,
            "side":     side.upper(),
            "type":     order_type.upper(),
            "quantity": qty_str,
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
                    side, qty_str,
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

    @staticmethod
    def _wallet_from(data: dict) -> dict:
        """
        Extract the wallet sub-dict from a balance response.
        Handles both the live-API 'SpotWallet' key and the documented 'Wallet' key.
        """
        return data.get("SpotWallet") or data.get("Wallet") or {}

    def free_usd(self) -> float:
        data = self.get_balance()
        if not data or not data.get("Success"):
            return 0.0
        return float(self._wallet_from(data).get("USD", {}).get("Free", 0.0))

    def free_btc(self) -> float:
        data = self.get_balance()
        if not data or not data.get("Success"):
            return 0.0
        return float(self._wallet_from(data).get("BTC", {}).get("Free", 0.0))

    def total_portfolio_usd(self, btc_price: float) -> float:
        data = self.get_balance()
        if not data or not data.get("Success"):
            return 0.0
        wallet = self._wallet_from(data)
        usd = float(wallet.get("USD", {}).get("Free", 0.0)) + \
              float(wallet.get("USD", {}).get("Lock", 0.0))
        btc = float(wallet.get("BTC", {}).get("Free", 0.0)) + \
              float(wallet.get("BTC", {}).get("Lock", 0.0))
        return usd + btc * btc_price
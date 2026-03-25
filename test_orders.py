#!/usr/bin/env python3
"""
test_orders.py
─────────────────────────────────────────────────────────────────────
Interactive CLI for manually testing every Roostoo API endpoint.

Option 9  — Quick Test  — MARKET BUY then immediately MARKET SELL.
Option 10 — Sell All    — scans every non-USD balance and MARKET SELLs
             each one, then prints a consolidated proceeds summary.
Option 11 — Sell One    — scans every non-USD balance, lets you choose
             a single asset, then MARKET SELLs it.

Usage
─────
    python test_orders.py
"""
from __future__ import annotations

import json
import time
from typing import Any

from strategy.roostoo_client import RoostooClient

_W = 58          # visual width used by all bars and the menu box
_QUOTE = "USD"   # quote currency; positions in this asset are never sold


# ── Output helpers ────────────────────────────────────────────────────────────

def pp(data: Any) -> None:
    """Pretty-print JSON."""
    print(json.dumps(data, indent=2, default=str))


def bar(title: str = "") -> None:
    if title:
        pad   = (_W - len(title) - 2) // 2
        right = _W - pad - len(title) - 2
        print(f"\n{'─' * pad} {title} {'─' * right}")
    else:
        print("─" * _W)


def ask(label: str, default: str | None = None) -> str:
    """Prompt with an optional shown default; repeats if the field is required."""
    hint = f" [{default}]" if default is not None else ""
    while True:
        val = input(f"  {label}{hint}: ").strip()
        if val:
            return val
        if default is not None:
            return default
        print("    ↳ required")


def ask_opt(label: str) -> str | None:
    """Optional prompt; returns None if the user presses Enter."""
    return input(f"  {label} [optional]: ").strip() or None


def confirm(msg: str) -> bool:
    return input(f"\n  {msg} (y/N): ").strip().lower() == "y"


# ── Balance parsing helper ────────────────────────────────────────────────────

def _parse_assets(balance: dict) -> dict[str, dict[str, float]]:
    """
    Return {ASSET: {free: float, locked: float}} from whatever structure
    the API hands back.  Tries the most common key names in order.
    """
    raw: dict | None = None
    for key in ("AssetInfo", "Balance", "Data", "Balances", "balances"):
        if key in balance and isinstance(balance[key], dict):
            raw = balance[key]
            break

    if raw is None:
        return {}

    result: dict[str, dict[str, float]] = {}
    for asset, info in raw.items():
        if isinstance(info, dict):
            free   = float(info.get("Free",   info.get("free",   info.get("Available",  0))))
            locked = float(info.get("Locked", info.get("locked", info.get("Freeze",      0))))
        else:
            free, locked = float(info), 0.0
        result[asset] = {"free": free, "locked": locked}

    return result


def _get_sellable_positions(
    c: RoostooClient,
) -> tuple[list[tuple[str, float, float]], list[tuple[str, float, str]]] | None:
    """
    Fetch balance and return (positions, skipped).

    positions : list of (asset, qty_rounded, qty_raw)  — ready to sell
    skipped   : list of (asset, free, reason)           — will not be sold

    Returns None if the balance fetch itself failed.
    """
    print("  Fetching balance…\n")
    balance = c.get_balance()
    pp(balance)

    if not balance or not balance.get("Success"):
        print(f"\n  ✗  Balance fetch failed : {(balance or {}).get('ErrMsg', 'no response')}")
        return None

    assets = _parse_assets(balance)
    if not assets:
        print("\n  ✗  Could not parse balance response — see raw output above.")
        return None

    positions: list[tuple[str, float, float]] = []
    skipped:   list[tuple[str, float, str]]   = []

    for asset, info in sorted(assets.items()):
        free = info["free"]

        if asset == _QUOTE:
            continue

        if free <= 0:
            skipped.append((asset, free, "zero / negative balance"))
            continue

        pair = f"{asset}/{_QUOTE}"
        prec = c._pair_amount_precision(pair)
        qty  = round(free, prec)

        if qty <= 0:
            skipped.append((asset, free, f"rounds to 0 at precision={prec}"))
            continue

        positions.append((asset, qty, free))

    return positions, skipped


def _print_positions_table(
    positions: list[tuple[str, float, float]],
    skipped:   list[tuple[str, float, str]],
    col: int = 10,
) -> None:
    """Render the positions table (and optional skipped list) to stdout."""
    print(f"  {'Asset':<{col}} {'Free':>16}  {'Will Sell':>16}  Pair")
    bar()
    for asset, qty, raw_qty in positions:
        pair = f"{asset}/{_QUOTE}"
        print(f"  {asset:<{col}} {raw_qty:>16.8f}  {qty:>16.8f}  → {pair}")

    if skipped:
        bar()
        print("  Skipped (zero / unsellable):")
        for asset, free, reason in skipped:
            print(f"    {asset:<{col}}  free={free:.8f}  ({reason})")


def _execute_single_sell(
    c: RoostooClient,
    asset: str,
    qty: float,
) -> None:
    """Place one MARKET SELL and print a single-position result summary."""
    pair = f"{asset}/{_QUOTE}"
    bar(f"MARKET SELL  ·  {qty} {pair}")
    res = c.place_order(pair=pair, side="SELL", quantity=qty, order_type="MARKET")
    pp(res)

    bar("Result")
    if not res or not res.get("Success"):
        err = (res or {}).get("ErrMsg", "unknown error")
        print(f"  ✗  Order failed : {err}")
        print("     Use option [7] Query Order to inspect the position.")
        return

    detail   = res.get("OrderDetail", {})
    fill_px  = float(detail.get("FilledAverPrice") or detail.get("Price", 0))
    comm     = float(detail.get("CommissionChargeValue", 0))
    proceeds = fill_px * qty
    net      = proceeds - comm

    print(f"  Pair            : {pair}")
    print(f"  Qty sold        : {qty}")
    print(f"  Fill price      : ${fill_px:>12,.4f}")
    bar()
    print(f"  Gross proceeds  : ${proceeds:>12,.4f} {_QUOTE}")
    print(f"  Commission      : ${comm:>12,.6f} {_QUOTE}")
    print(f"  Net proceeds    : ${net:>12,.4f} {_QUOTE}")
    bar()
    print(f"  Order ID        : {detail.get('OrderID', '?')}")
    print(f"\n  ✓  Sell complete.")


# ── Thin wrappers for endpoints not yet on RoostooClient ──────────────────────

def _server_time(c: RoostooClient) -> dict:
    r = c._session.get(f"{c._base_url}/v3/serverTime", timeout=c._timeout)
    r.raise_for_status()
    return r.json()


def _ticker(c: RoostooClient, pair: str | None = None) -> dict:
    params: dict[str, str] = {"timestamp": str(int(time.time() * 1000))}
    if pair:
        params["pair"] = pair
    r = c._session.get(f"{c._base_url}/v3/ticker", params=params, timeout=c._timeout)
    r.raise_for_status()
    return r.json()


def _pending_count(c: RoostooClient) -> dict:
    params, _, sig = c._sign({})
    r = c._session.get(
        f"{c._base_url}/v3/pending_count",
        headers={"RST-API-KEY": c._api_key, "MSG-SIGNATURE": sig},
        params=params,
        timeout=c._timeout,
    )
    r.raise_for_status()
    return r.json()


# ── Menu handlers ─────────────────────────────────────────────────────────────

def h_server_time(c: RoostooClient) -> None:
    bar("Server Time")
    pp(_server_time(c))


def h_exchange_info(c: RoostooClient) -> None:
    bar("Exchange Info")
    pp(c.get_exchange_info())


def h_ticker(c: RoostooClient) -> None:
    bar("Market Ticker")
    pair = ask_opt("Pair (e.g. BTC/USD) — blank for all")
    pp(_ticker(c, pair))


def h_balance(c: RoostooClient) -> None:
    bar("Balance")
    pp(c.get_balance())


def h_pending_count(c: RoostooClient) -> None:
    bar("Pending Order Count")
    pp(_pending_count(c))


def h_place_order(c: RoostooClient) -> None:
    bar("Place Order")
    pair  = ask("Pair",     "BTC/USD")
    side  = ask("Side",     "BUY").upper()
    otype = ask("Type",     "MARKET").upper()
    qty   = float(ask("Quantity", "0.001"))
    price: float | None = float(ask("Price")) if otype == "LIMIT" else None
    pp(c.place_order(pair=pair, side=side, quantity=qty,
                     price=price, order_type=otype))


def h_query_order(c: RoostooClient) -> None:
    bar("Query Order")
    print("  All parameters optional — press Enter to skip.\n")
    order_id = ask_opt("Order ID")
    pair:         str  | None = ask_opt("Pair (e.g. BTC/USD)") if not order_id else None
    pending_only: bool | None = None
    limit:        int  | None = None
    if not order_id:
        raw = input("  Pending only? (y / n / Enter = skip): ").strip().lower()
        pending_only = True if raw == "y" else (False if raw == "n" else None)
        raw = input("  Limit [default 100]: ").strip()
        limit = int(raw) if raw else None
    pp(c.query_order(order_id=order_id, pair=pair,
                     pending_only=pending_only, limit=limit))


def h_cancel_order(c: RoostooClient) -> None:
    bar("Cancel Order")
    print("  Leave both blank to cancel ALL pending orders.\n")
    order_id = ask_opt("Order ID")
    pair: str | None = None
    if not order_id:
        pair = ask_opt("Pair (e.g. BTC/USD)")
    if not order_id and not pair:
        if not confirm("⚠  Cancel ALL pending orders?"):
            print("  Aborted.")
            return
    pp(c.cancel_order(order_id=order_id, pair=pair))


def h_quick_test(c: RoostooClient) -> None:
    """MARKET BUY then immediately MARKET SELL the same quantity on the same pair."""
    bar("Quick Test  ★  Buy → Sell")

    info  = c.get_exchange_info()
    pairs = list((info or {}).get("TradePairs", {}).keys())
    if pairs:
        print(f"  Available pairs : {', '.join(pairs)}")

    pair      = ask("Pair",     "BNB/USD")
    qty       = float(ask("Quantity", "1"))
    prec      = c._pair_amount_precision(pair)
    qty_shown = round(qty, prec)

    print(f"\n  Order 1  →  MARKET BUY  {qty_shown} {pair}")
    print(f"  Order 2  →  MARKET SELL {qty_shown} {pair}  (immediately after)")
    if not confirm("Proceed?"):
        print("  Aborted.")
        return

    bar("Order 1 — BUY")
    buy = c.place_order(pair=pair, side="BUY", quantity=qty, order_type="MARKET")
    pp(buy)

    if not buy or not buy.get("Success"):
        print(f"\n  ✗  BUY failed : {(buy or {}).get('ErrMsg', 'no response from API')}")
        print("     SELL skipped — no position was opened.")
        return

    bar("Order 2 — SELL")
    sell = c.place_order(pair=pair, side="SELL", quantity=qty, order_type="MARKET")
    pp(sell)

    bar("Round-trip Summary")
    if not sell or not sell.get("Success"):
        print(f"  ✗  SELL failed : {(sell or {}).get('ErrMsg', 'no response from API')}")
        print("     Use option [7] Query Order to inspect the open position.")
        return

    bd, sd  = buy["OrderDetail"], sell["OrderDetail"]
    buy_px  = float(bd.get("FilledAverPrice") or bd.get("Price", 0))
    sell_px = float(sd.get("FilledAverPrice") or sd.get("Price", 0))
    gross   = (sell_px - buy_px) * qty_shown
    comm    = (float(bd.get("CommissionChargeValue", 0)) +
               float(sd.get("CommissionChargeValue", 0)))
    net     = gross - comm

    def signed(n: float) -> str:
        return ("+$" if n >= 0 else "-$") + f"{abs(n):.6f}"

    print(f"  Pair              : {pair}")
    print(f"  Qty traded        : {qty_shown}")
    print(f"  Buy  fill price   : ${buy_px:>14,.4f}")
    print(f"  Sell fill price   : ${sell_px:>14,.4f}")
    bar()
    print(f"  Gross P&L         : {signed(gross)}")
    print(f"  Commission  (×2)  : -${comm:.6f}")
    print(f"  Net P&L           : {signed(net)}")
    bar()
    print(f"  BUY  order ID     : {bd.get('OrderID', '?')}")
    print(f"  SELL order ID     : {sd.get('OrderID', '?')}")
    print(f"\n  ✓  Round-trip complete.")


def h_sell_all_positions(c: RoostooClient) -> None:
    """Scan balance then MARKET SELL every non-USD position."""
    bar("Sell All Positions")

    result = _get_sellable_positions(c)
    if result is None:
        return
    positions, skipped = result

    bar("Positions Found")
    if not positions:
        print(f"  ℹ  No sellable non-{_QUOTE} positions found.\n")
        if skipped:
            print("  Skipped assets:")
            for asset, free, reason in skipped:
                print(f"    {asset:<10}  free={free:.8f}  ({reason})")
        return

    _print_positions_table(positions, skipped)
    bar()
    print(f"\n  {len(positions)} MARKET SELL order(s) will be placed.")
    if not confirm(f"⚠  Proceed with selling ALL {len(positions)} position(s)?"):
        print("  Aborted — no orders placed.")
        return

    results: list[tuple[str, float, dict | None]] = []
    for i, (asset, qty, _) in enumerate(positions, 1):
        pair = f"{asset}/{_QUOTE}"
        bar(f"Sell {i}/{len(positions)}  ·  {qty} {pair}")
        res = c.place_order(pair=pair, side="SELL", quantity=qty, order_type="MARKET")
        pp(res)
        results.append((pair, qty, res))

    bar("Sell-All Summary")
    ok   = [(pair, qty, res) for pair, qty, res in results if res and res.get("Success")]
    fail = [(pair, qty, res) for pair, qty, res in results if not (res and res.get("Success"))]

    print(f"  Positions found   : {len(positions)}")
    print(f"  ✓ Orders placed   : {len(ok)}")
    if fail:
        print(f"  ✗ Orders failed   : {len(fail)}")

    if ok:
        bar("Filled Orders")
        total_proceeds   = 0.0
        total_commission = 0.0
        print(f"  {'Pair':<14} {'Qty':>10}  {'Fill Px':>12}  {'Proceeds':>14}  {'Comm':>12}")
        bar()
        for pair, qty, res in ok:
            detail   = (res or {}).get("OrderDetail", {})
            fill_px  = float(detail.get("FilledAverPrice") or detail.get("Price", 0))
            comm     = float(detail.get("CommissionChargeValue", 0))
            proceeds = fill_px * qty
            total_proceeds   += proceeds
            total_commission += comm
            print(
                f"  {pair:<14} {qty:>10.6f}  "
                f"${fill_px:>11,.4f}  "
                f"${proceeds:>13,.4f}  "
                f"${comm:>11,.6f}"
            )
        bar()
        net = total_proceeds - total_commission
        print(f"  {'Gross proceeds':<30}: ${total_proceeds:>13,.4f} {_QUOTE}")
        print(f"  {'Total commission':<30}: ${total_commission:>13,.6f} {_QUOTE}")
        print(f"  {'Net proceeds':<30}: ${net:>13,.4f} {_QUOTE}")

    if fail:
        bar("Failed Orders")
        for pair, qty, res in fail:
            err = (res or {}).get("ErrMsg", "unknown error")
            print(f"  ✗  {pair}  qty={qty}  →  {err}")
        print("\n  Run option [7] Query Order to inspect any open positions.")

    bar()
    if not fail:
        print("  ✓  All positions closed successfully.")
    else:
        print("  ⚠  Some positions could not be closed — see above.")


def h_sell_one_position(c: RoostooClient) -> None:
    """Scan balance, let the user pick a single asset, then MARKET SELL it."""
    bar("Sell One Position")

    result = _get_sellable_positions(c)
    if result is None:
        return
    positions, skipped = result

    bar("Positions Found")
    if not positions:
        print(f"  ℹ  No sellable non-{_QUOTE} positions found.\n")
        if skipped:
            print("  Skipped assets:")
            for asset, free, reason in skipped:
                print(f"    {asset:<10}  free={free:.8f}  ({reason})")
        return

    # ── Numbered pick list ────────────────────────────────────────────────────
    col = 10
    print(f"  #   {'Asset':<{col}} {'Free':>16}  {'Will Sell':>16}  Pair")
    bar()
    for i, (asset, qty, raw_qty) in enumerate(positions, 1):
        pair = f"{asset}/{_QUOTE}"
        print(f"  [{i}] {asset:<{col}} {raw_qty:>16.8f}  {qty:>16.8f}  → {pair}")

    if skipped:
        bar()
        print("  Skipped (zero / unsellable):")
        for asset, free, reason in skipped:
            print(f"    {asset:<{col}}  free={free:.8f}  ({reason})")

    bar()

    # ── Asset selection ───────────────────────────────────────────────────────
    while True:
        raw = input(f"\n  Enter number to sell [1–{len(positions)}], or 0 to cancel: ").strip()
        if raw == "0":
            print("  Aborted — no orders placed.")
            return
        try:
            idx = int(raw) - 1
        except ValueError:
            print("  Not a number — try again.")
            continue
        if not (0 <= idx < len(positions)):
            print(f"  Out of range (1–{len(positions)}) — try again.")
            continue
        break

    asset, qty, raw_qty = positions[idx]
    pair = f"{asset}/{_QUOTE}"

    # ── Optional partial quantity ─────────────────────────────────────────────
    bar("Quantity")
    print(f"  Full position : {raw_qty:.8f} {asset}  (will sell {qty} after rounding)")
    raw_amt = input(
        f"  Enter quantity to sell, or press Enter for full amount [{qty}]: "
    ).strip()

    if raw_amt:
        prec        = c._pair_amount_precision(pair)
        custom_qty  = round(float(raw_amt), prec)
        if custom_qty <= 0:
            print("  ✗  Quantity must be greater than zero.  Aborted.")
            return
        if custom_qty > qty:
            print(
                f"  ✗  {custom_qty} exceeds available rounded balance of {qty}.  Aborted."
            )
            return
        qty = custom_qty

    # ── Confirm & execute ─────────────────────────────────────────────────────
    if not confirm(f"⚠  MARKET SELL {qty} {asset} → {pair}?"):
        print("  Aborted — no orders placed.")
        return

    _execute_single_sell(c, asset, qty)


# ── Menu definition ───────────────────────────────────────────────────────────

MENU: list[tuple[str, Any]] = [
    ("Check Server Time",                    h_server_time),
    ("Exchange Info",                        h_exchange_info),
    ("Get Ticker",                           h_ticker),
    ("Get Balance",                          h_balance),
    ("Pending Order Count",                  h_pending_count),
    ("Place Order",                          h_place_order),
    ("Query Order",                          h_query_order),
    ("Cancel Order",                         h_cancel_order),
    ("★  Quick Test: Buy → Sell now",        h_quick_test),
    ("★  Sell All Positions",                h_sell_all_positions),
    ("★  Sell One Position  (pick an asset)", h_sell_one_position),
]

_DIVIDERS_AFTER = {4, 7}


def print_menu() -> None:
    inner   = _W - 2
    label_w = inner - 7

    top    = "╔" + "═" * inner + "╗"
    mid    = "╠" + "═" * inner + "╣"
    bottom = "╚" + "═" * inner + "╝"
    div    = "║  " + "·" * (inner - 4) + "  ║"
    title  = "Roostoo API  ·  Manual Test Console"

    print(f"\n{top}")
    print(f"║{title.center(inner)}║")
    print(mid)
    for i, (label, _) in enumerate(MENU):
        print(f"║  [{i + 1}]  {label:<{label_w}}║")
        if i in _DIVIDERS_AFTER:
            print(div)
    print(f"║  [0]  {'Exit':<{label_w}}║")
    print(bottom)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    client = RoostooClient()
    pairs  = list(client._amount_precision.keys())

    print(f"\n  API endpoint : {client._base_url}")
    print(f"  API key      : {client._api_key[:10]}…")
    print(f"  Known pairs  : {', '.join(pairs) if pairs else '(none loaded)'}")

    while True:
        print_menu()
        try:
            raw = input(f"\n  Choose [0–{len(MENU)}]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye.\n")
            break

        if raw == "0":
            print("\n  Goodbye.\n")
            break

        try:
            idx = int(raw) - 1
        except ValueError:
            print("  Not a number — try again.")
            continue

        if not (0 <= idx < len(MENU)):
            print(f"  Out of range (1–{len(MENU)}) — try again.")
            continue

        _, handler = MENU[idx]
        try:
            handler(client)
        except KeyboardInterrupt:
            print("\n  (interrupted — back to menu)")
        except Exception as exc:
            print(f"\n  ✗  Unexpected error: {exc}")


if __name__ == "__main__":
    main()
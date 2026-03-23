#!/usr/bin/env python3
"""
terminal_monitor.py  ─  Real-time terminal dashboard for the HMM trading bot.

Run from the project root (separate terminal from live_trading.py):
    python terminal_monitor.py [--refresh N]   (default: 10 s)

Keys:
    q / ESC  ─ quit
    r        ─ force refresh immediately
"""
from __future__ import annotations

import argparse
import curses
import hashlib
import hmac
import importlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests as _req
    _HAS_REQUESTS = True
except ImportError:
    _req = None               # type: ignore[assignment]
    _HAS_REQUESTS = False

# ─── Default / fallback configuration ─────────────────────────────────────────
# These are overwritten on every refresh cycle from config.py, so changing
# config.py while the monitor is running takes effect at the next poll.
JOURNAL_PATH       = Path("trade_journal.json")
ROOSTOO_BASE_URL   = "https://mock-api.roostoo.com"
ROOSTOO_API_KEY    = ""
ROOSTOO_SECRET_KEY = ""
_CONFIGURED_SYMS: list[str] = []

# Try a first-load so we have values before the first refresh fires
try:
    import config as _cfg_module
    ROOSTOO_BASE_URL   = getattr(_cfg_module, "ROOSTOO_BASE_URL",   ROOSTOO_BASE_URL)
    ROOSTOO_API_KEY    = getattr(_cfg_module, "ROOSTOO_API_KEY",    ROOSTOO_API_KEY)
    ROOSTOO_SECRET_KEY = getattr(_cfg_module, "ROOSTOO_SECRET_KEY", ROOSTOO_SECRET_KEY)
    _CONFIGURED_SYMS   = [
        i["instrument_id_str"]
        for i in getattr(_cfg_module, "INSTRUMENTS", [])
    ]
    _HAS_CONFIG = True
except ImportError:
    _cfg_module  = None   # type: ignore[assignment]
    _HAS_CONFIG  = False

BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/price"
DEFAULT_REFRESH    = 10


# ─── Config hot-reload ────────────────────────────────────────────────────────

def reload_config() -> str:
    """
    Re-read config.py from disk and update all credential / instrument globals.
    Returns an error string on failure, "" on success.
    This is called at the start of every refresh cycle so a mid-run API switch
    is picked up without restarting the monitor.
    """
    global ROOSTOO_BASE_URL, ROOSTOO_API_KEY, ROOSTOO_SECRET_KEY, _CONFIGURED_SYMS
    if not _HAS_CONFIG or _cfg_module is None:
        return ""
    try:
        importlib.reload(_cfg_module)
        ROOSTOO_BASE_URL   = getattr(_cfg_module, "ROOSTOO_BASE_URL",   ROOSTOO_BASE_URL)
        ROOSTOO_API_KEY    = getattr(_cfg_module, "ROOSTOO_API_KEY",    ROOSTOO_API_KEY)
        ROOSTOO_SECRET_KEY = getattr(_cfg_module, "ROOSTOO_SECRET_KEY", ROOSTOO_SECRET_KEY)
        _CONFIGURED_SYMS   = [
            i["instrument_id_str"]
            for i in getattr(_cfg_module, "INSTRUMENTS", [])
        ]
        return ""
    except Exception as exc:
        return f"config reload: {exc}"


# ─── Symbol helpers ───────────────────────────────────────────────────────────

def _display(journal_sym: str) -> str:
    """'BTCUSDT.BINANCE' → 'BTCUSDT'"""
    return journal_sym.split(".")[0]


def _to_binance(journal_sym: str) -> str:
    """
    Convert a journal/config symbol to a Binance ticker.
      'BTCUSDT.BINANCE' → 'BTCUSDT'
      'BTC/USD'         → 'BTCUSDT'
      'DOGE/USD'        → 'DOGEUSDT'
    """
    s = journal_sym.split(".")[0].replace("/", "").upper()
    if s.endswith("USD") and not s.endswith("USDT"):
        s = s[:-3] + "USDT"
    return s


# ─── Data fetchers ────────────────────────────────────────────────────────────

def _sign(params: dict) -> tuple[dict, str]:
    params = {**params, "timestamp": str(int(time.time() * 1000))}
    total  = "&".join(f"{k}={params[k]}" for k in sorted(params))
    sig    = hmac.new(
        ROOSTOO_SECRET_KEY.encode(), total.encode(), hashlib.sha256
    ).hexdigest()
    return params, sig


def fetch_wallet() -> tuple[dict, str]:
    """
    Returns (SpotWallet dict, error_string).
    SpotWallet is {} on any error; error_string is "" on success.
    Uses the ROOSTOO_API_KEY / ROOSTOO_SECRET_KEY globals, which are refreshed
    from config.py before this is called each cycle.
    """
    if not _HAS_REQUESTS:
        return {}, "requests not installed"
    if not ROOSTOO_API_KEY or not ROOSTOO_SECRET_KEY:
        return {}, "API key not configured"
    try:
        params, sig = _sign({})
        r = _req.get(
            f"{ROOSTOO_BASE_URL}/v3/balance",
            headers={"RST-API-KEY": ROOSTOO_API_KEY, "MSG-SIGNATURE": sig},
            params=params,
            timeout=6,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("Success"):
            return data.get("SpotWallet", {}), ""
        return {}, f"Roostoo: {data.get('ErrMsg', 'unknown error')}"
    except Exception as exc:
        return {}, f"wallet: {exc}"


def fetch_prices(extra_symbols: list[str]) -> tuple[dict[str, float], str]:
    """
    Fetch Binance prices for all configured instruments plus any extra symbols.
    Returns (prices_dict, error_string).
    """
    if not _HAS_REQUESTS:
        return {}, "requests not installed"

    want: set[str] = set(_CONFIGURED_SYMS) | set(extra_symbols)

    try:
        r = _req.get(BINANCE_TICKER_URL, timeout=5)
        r.raise_for_status()
        all_px: dict[str, float] = {
            item["symbol"]: float(item["price"]) for item in r.json()
        }
    except Exception as exc:
        return {}, f"Binance ticker: {exc}"

    result:  dict[str, float] = {}
    missing: list[str]        = []

    for sym in want:
        b = _to_binance(sym)
        if b in all_px:
            result[sym] = all_px[b]
        else:
            missing.append(b)

    if "BTCUSDT" in all_px:
        result["_BTC_USD"] = all_px["BTCUSDT"]

    if not result and missing:
        return {}, f"no matches on Binance for: {', '.join(missing[:4])}"

    return result, ""


def load_journal() -> dict:
    try:
        if JOURNAL_PATH.exists():
            return json.loads(JOURNAL_PATH.read_text())
    except Exception:
        pass
    return {"starting_balance": 1_000_000.0, "trades": []}


# ─── Curses colour helpers ────────────────────────────────────────────────────

_P_GREEN  = 1
_P_RED    = 2
_P_YELLOW = 3
_P_CYAN   = 4
_P_BOLD   = 5
_P_DIM    = 6


def _init_colors() -> None:
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(_P_GREEN,  curses.COLOR_GREEN,  -1)
    curses.init_pair(_P_RED,    curses.COLOR_RED,    -1)
    curses.init_pair(_P_YELLOW, curses.COLOR_YELLOW, -1)
    curses.init_pair(_P_CYAN,   curses.COLOR_CYAN,   -1)
    curses.init_pair(_P_BOLD,   curses.COLOR_WHITE,  -1)
    curses.init_pair(_P_DIM,    curses.COLOR_WHITE,  -1)


def _a(pair: int, bold: bool = False, dim: bool = False) -> int:
    attr = curses.color_pair(pair)
    if bold: attr |= curses.A_BOLD
    if dim:  attr |= curses.A_DIM
    return attr


def _put(win, row: int, col: int, text: str, attr: int = 0) -> int:
    h, w = win.getmaxyx()
    if row >= h - 1 or col >= w:
        return row
    try:
        win.addstr(row, col, text[: w - col - 1], attr)
    except curses.error:
        pass
    return row + 1


def _rule(win, row: int, ch: str = "─", attr: int = 0) -> int:
    _, w = win.getmaxyx()
    return _put(win, row, 0, ch * (w - 1), attr)


# ─── Renderer ─────────────────────────────────────────────────────────────────

def render(
    win,
    journal:   dict,
    wallet:    dict,
    prices:    dict[str, float],
    refresh:   int,
    last_errs: list[str],
) -> None:
    win.erase()
    h, w = win.getmaxyx()

    NORM  = 0
    GREEN = _a(_P_GREEN,  bold=True)
    RED   = _a(_P_RED,    bold=True)
    YEL   = _a(_P_YELLOW, bold=True)
    CYAN  = _a(_P_CYAN)
    BOLD  = _a(_P_BOLD,   bold=True)
    DIM   = _a(_P_DIM,    dim=True)

    row = 0

    # ── Header ────────────────────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d  %H:%M:%S UTC")
    row = _put(win, row, 0,
               "  HMM REGIME STRATEGY  —  LIVE TERMINAL MONITOR".center(w - 1), BOLD)
    row = _put(win, row, 0, ts.center(w - 1), CYAN)
    row = _rule(win, row, "═", CYAN)

    trades        = journal.get("trades", [])
    start_bal     = journal.get("starting_balance", 50_000.0)
    open_trades   = [t for t in trades if     t.get("open")]
    closed_trades = [t for t in trades if not t.get("open")]
    btc_px        = prices.get("_BTC_USD", 0.0)

    # ── Wallet ────────────────────────────────────────────────────────────────
    api_label = f"  (key: …{ROOSTOO_API_KEY[-6:]})" if ROOSTOO_API_KEY else ""
    row = _put(win, row, 0, f" WALLET  (Roostoo live){api_label}", CYAN)

    if wallet:
        usd_free = float(wallet.get("USD", {}).get("Free", 0))
        usd_lock = float(wallet.get("USD", {}).get("Lock", 0))
        btc_free = float(wallet.get("BTC", {}).get("Free", 0))
        btc_lock = float(wallet.get("BTC", {}).get("Lock", 0))
        total    = (usd_free + usd_lock) + (btc_free + btc_lock) * btc_px
        pnl_u    = total - start_bal
        pnl_p    = pnl_u / (start_bal + 1e-12) * 100
        pa       = GREEN if pnl_u >= 0 else RED
        sign     = "+" if pnl_u >= 0 else ""

        row = _put(win, row, 2,
                   f"USD   Free: ${usd_free:>14,.2f}    Locked: ${usd_lock:>12,.2f}", NORM)
        row = _put(win, row, 2,
                   f"BTC   Free: {btc_free:>16.6f}    Locked: {btc_lock:>14.6f}", NORM)
        if btc_px:
            row = _put(win, row, 2,
                       f"Portfolio: ${total:>14,.2f}    "
                       f"P&L: {sign}${pnl_u:,.2f}  ({sign}{pnl_p:.2f}%)", pa)
    else:
        note = (
            "[install requests: pip install requests]"
            if not _HAS_REQUESTS else
            "[balance unavailable — check API key / network]"
        )
        row = _put(win, row, 2, note, DIM)

    row = _rule(win, row, "─", CYAN)

    # ── Live prices ───────────────────────────────────────────────────────────
    row = _put(win, row, 0, " LIVE PRICES  (Binance)", CYAN)

    display_px = {k: v for k, v in prices.items() if not k.startswith("_")}
    if display_px:
        items  = sorted(display_px.items())
        n_cols = min(len(items), 3)
        col_w  = max(32, (w - 2) // n_cols)
        base_r = row
        for idx, (sym, px) in enumerate(items):
            col_pos = 2 + (idx % n_cols) * col_w
            row_pos = base_r + idx // n_cols
            _put(win, row_pos, col_pos,
                 f"{_display(sym):<14}  ${px:>12,.4f}", NORM)
        row = base_r + (len(items) - 1) // n_cols + 1
    else:
        err_detail = next((e for e in last_errs if e), "")
        msg = f" [price feed unavailable{f'  ({err_detail})' if err_detail else ''}]"
        row = _put(win, row, 2, msg, RED)

    row = _rule(win, row, "─", CYAN)

    # ── Open positions ────────────────────────────────────────────────────────
    row = _put(win, row, 0, f" OPEN POSITIONS  ({len(open_trades)})", CYAN)

    if open_trades:
        row = _put(win, row, 0,
                   f"  {'Symbol':<20} {'Entry':>12} {'Now':>12}"
                   f" {'Qty':>10} {'Unreal P&L':>12} {'Regime':<10} {'Bull%':>6}", YEL)
        for t in open_trades:
            if row >= h - 4:
                break
            sym      = t.get("symbol",      "?")
            entry_px = t.get("entry_price", 0.0)
            qty      = t.get("quantity",    0.0)
            curr_px  = prices.get(sym, entry_px)
            unr_u    = (curr_px - entry_px) * qty
            unr_p    = (curr_px - entry_px) / (entry_px + 1e-12) * 100
            regime   = t.get("regime",    "?")
            bull_p   = t.get("bull_prob", 0.0) * 100
            attr     = GREEN if unr_u >= 0 else RED
            sign     = "+" if unr_u >= 0 else ""
            row = _put(win, row, 0,
                       f"  {_display(sym):<20} {entry_px:>12,.4f} {curr_px:>12,.4f}"
                       f" {qty:>10.6f} {sign}{unr_u:>+11,.2f}"
                       f" {regime:<10} {bull_p:>5.1f}%", attr)
    else:
        row = _put(win, row, 2, " No open positions.", DIM)

    row = _rule(win, row, "─", CYAN)

    # ── Closed trade summary ──────────────────────────────────────────────────
    row = _put(win, row, 0, f" CLOSED TRADES  ({len(closed_trades)})", CYAN)

    if closed_trades:
        pnls   = [t.get("pnl_usd", 0.0) for t in closed_trades]
        wins   = [p for p in pnls if p >  0]
        losses = [p for p in pnls if p <= 0]
        total  = sum(pnls)
        wr     = len(wins) / len(pnls) * 100
        pf     = sum(wins) / (abs(sum(losses)) + 1e-12)
        avg_w  = sum(wins)   / (len(wins)   + 1e-12)
        avg_l  = sum(losses) / (len(losses) + 1e-12)
        sign   = "+" if total >= 0 else ""
        row = _put(win, row, 2,
                   f"Total P&L: {sign}${total:,.2f}   Win Rate: {wr:.1f}%   "
                   f"Profit Factor: {pf:.2f}   Avg Win: +${avg_w:,.2f}   Avg Loss: ${avg_l:,.2f}",
                   GREEN if total >= 0 else RED)
        row += 1

        max_rows = max(0, h - row - 3)
        recent   = sorted(
            closed_trades,
            key=lambda t: t.get("exit_time", ""),
            reverse=True,
        )[:max_rows]

        row = _put(win, row, 0,
                   f"  {'Symbol':<20} {'Entry':>10} {'Exit':>10}"
                   f" {'Qty':>8} {'P&L $':>10} {'%':>7}  Reason", YEL)
        for t in recent:
            if row >= h - 3:
                break
            sym   = t.get("symbol",      "?")
            ep    = t.get("entry_price", 0.0)
            xp    = t.get("exit_price",  0.0)
            qty   = t.get("quantity",    0.0)
            pnl_u = t.get("pnl_usd",    0.0)
            pnl_p = t.get("pnl_pct",    0.0)
            rsn   = t.get("exit_reason", "-")[:26]
            sign  = "+" if pnl_u >= 0 else ""
            row   = _put(win, row, 0,
                         f"  {_display(sym):<20} {ep:>10,.4f} {xp:>10,.4f}"
                         f" {qty:>8.6f} {sign}{pnl_u:>+9,.2f} {sign}{pnl_p:>+6.2f}%  {rsn}",
                         GREEN if pnl_u >= 0 else RED)
    else:
        row = _put(win, row, 2, " No closed trades yet.", DIM)

    # ── Footer / error bar ────────────────────────────────────────────────────
    err_summary = "  |  ERR: " + "  /  ".join(e for e in last_errs if e)[:60] \
                  if any(last_errs) else ""
    footer = f"  [q] quit   [r] refresh now   auto-refresh every {refresh}s{err_summary}"
    try:
        win.addstr(h - 1, 0, footer[: w - 1], CYAN if not any(last_errs) else RED)
    except curses.error:
        pass

    win.refresh()


# ─── Event loop ───────────────────────────────────────────────────────────────

def _curses_main(stdscr, refresh_secs: int) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    _init_colors()

    journal:   dict             = {"starting_balance": 50_000.0, "trades": []}
    wallet:    dict             = {}
    prices:    dict[str, float] = {}
    last_errs: list[str]        = ["", "", "", ""]  # journal, config, wallet, prices
    last_ts:   float            = 0.0

    while True:
        key = stdscr.getch()
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("r"), ord("R")):
            last_ts = 0.0

        now = time.time()
        if now - last_ts >= refresh_secs:

            # 0. Hot-reload config.py ← fixes stale API key after a switch
            last_errs[1] = reload_config()

            # 1. Journal
            try:
                journal = load_journal()
                last_errs[0] = ""
            except Exception as exc:
                last_errs[0] = f"journal: {exc}"

            # 2. Wallet  (uses freshly reloaded credentials)
            wallet, last_errs[2] = fetch_wallet()

            # 3. Prices
            journal_syms = list({t["symbol"] for t in journal.get("trades", [])})
            prices, last_errs[3] = fetch_prices(journal_syms)

            last_ts = now

        render(stdscr, journal, wallet, prices, refresh_secs, last_errs)
        curses.napms(250)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="HMM Strategy live terminal dashboard"
    )
    ap.add_argument(
        "--refresh", "-r", type=int, default=DEFAULT_REFRESH, metavar="SECS",
        help=f"Refresh interval in seconds (default: {DEFAULT_REFRESH})",
    )
    args = ap.parse_args()
    try:
        curses.wrapper(_curses_main, args.refresh)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
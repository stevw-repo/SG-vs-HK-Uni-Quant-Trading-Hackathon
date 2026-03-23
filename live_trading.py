"""
live_trading.py
Live trading:
  Binance WebSocket (NautilusTrader public data, no API key)
  +
  Roostoo REST API (order execution and balance only)

Usage
-----
    python live_trading.py

Dashboard (auto-starts on port 8080)
-------------------------------------
    http://<server-ip>:8080/data  →  raw JSON feed for the React dashboard
"""

from __future__ import annotations

import asyncio
import gc
import http.server
import json
import logging
import threading
from pathlib import Path
from uuid import uuid4

from nautilus_trader.adapters.binance.common.enums import BinanceAccountType
from nautilus_trader.adapters.binance.config import BinanceDataClientConfig
from nautilus_trader.adapters.binance.factories import BinanceLiveDataClientFactory
from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.component import LiveClock, MessageBus
from nautilus_trader.common.providers import InstrumentProvider
from nautilus_trader.config import (
    InstrumentProviderConfig,
    LiveExecClientConfig,
    TradingNodeConfig,
)
from nautilus_trader.live.execution_client import LiveExecutionClient
from nautilus_trader.live.factories import LiveExecClientFactory
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.currencies import Currency
from nautilus_trader.model.enums import AccountType, LiquiditySide, OmsType, OrderSide, OrderType
from nautilus_trader.model.identifiers import (
    AccountId,
    ClientId,
    InstrumentId,
    TradeId,
    VenueOrderId,
    Venue,
)
from nautilus_trader.model.objects import AccountBalance, Money, Price, Quantity

import config as cfg
from strategy.hmm_strategy import HMMStrategy, HMMStrategyConfig
from strategy.roostoo_client import RoostooClient
from utils.binance_instruments import get_instruments_sync
from utils.capital_allocator import CapitalAllocator
from utils.trade_journal import TradeJournal

import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("live_trading")


# ══════════════════════════════════════════════════════════════════════════════
#  Dashboard HTTP server
# ══════════════════════════════════════════════════════════════════════════════

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HMM Strategy Live Dashboard</title>
<style>
  :root{--bg:#0d1117;--card:#161b22;--border:#30363d;--text:#e6edf3;
        --muted:#8b949e;--green:#3fb950;--red:#f85149;--yellow:#d29922;--cyan:#79c0ff}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font:13px/1.5 'SF Mono','Fira Code','Courier New',monospace;padding:16px}
  h1{color:var(--cyan);font-size:16px;letter-spacing:.05em}
  .ts{color:var(--muted);font-size:11px;margin-bottom:20px}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;margin-bottom:20px}
  .card{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:12px}
  .clabel{color:var(--cyan);font-size:10px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px}
  .cval{font-size:18px;font-weight:700}
  .green{color:var(--green)}.red{color:var(--red)}.yellow{color:var(--yellow)}.cyan{color:var(--cyan)}
  section{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:14px;margin-bottom:14px}
  .stitle{color:var(--cyan);font-size:10px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:12px}
  table{width:100%;border-collapse:collapse}
  th{color:var(--muted);font-size:10px;text-align:left;padding:3px 8px;border-bottom:1px solid var(--border);white-space:nowrap}
  td{padding:4px 8px;border-bottom:1px solid #21262d;font-size:12px;white-space:nowrap}
  tr:last-child td{border-bottom:none}
  tr:hover td{background:#1c2128}
  .badge{display:inline-block;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:700}
  .badge.BULL{background:#0d4429;color:var(--green)}
  .badge.BEAR{background:#3d0f0a;color:var(--red)}
  .badge.SIDEWAYS,.badge.UNKNOWN{background:#3a2d00;color:var(--yellow)}
  .empty{color:var(--muted);text-align:center;padding:16px;font-size:12px}
  #err{color:var(--red);font-size:11px;margin-top:10px;min-height:16px}
</style>
</head>
<body>
<h1>&#9889; HMM Regime Strategy &#8212; Live Dashboard</h1>
<div class="ts" id="ts">Connecting&#8230;</div>
<div class="grid" id="cards"></div>
<section><div class="stitle">Open Positions</div><div id="open"></div></section>
<section><div class="stitle">Closed Trade History</div><div id="closed"></div></section>
<div id="err"></div>
<script>
const $=id=>document.getElementById(id);
const f=(n,d=2)=>Number(n).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const pct=n=>(n>=0?'+':'')+f(n,2)+'%';
const usd=n=>(n>=0?'+$':'-$')+f(Math.abs(n),2);
const cls=n=>n>=0?'green':'red';
const badge=r=>`<span class="badge ${r||'UNKNOWN'}">${r||'UNKNOWN'}</span>`;

function renderCards(d){
  const trades=d.trades||[],open=trades.filter(t=>t.open),closed=trades.filter(t=>!t.open);
  const sb=d.starting_balance||50000;
  const totalPnl=closed.reduce((s,t)=>s+(t.pnl_usd||0),0);
  const wins=closed.filter(t=>t.pnl_usd>0);
  const losses=closed.filter(t=>t.pnl_usd<=0);
  const wr=closed.length?(wins.length/closed.length*100).toFixed(1)+'%':'—';
  const pf=losses.length?(wins.reduce((s,t)=>s+t.pnl_usd,0)/(Math.abs(losses.reduce((s,t)=>s+t.pnl_usd,0))+1e-9)).toFixed(2):'—';
  $('cards').innerHTML=[
    ['Starting Balance','$'+f(sb),''],
    ['Realised P&L',usd(totalPnl),cls(totalPnl)],
    ['Win Rate',wr,''],
    ['Profit Factor',pf,''],
    ['Open Positions',open.length,open.length?'yellow':''],
    ['Total Trades',closed.length,''],
  ].map(([l,v,c])=>`<div class="card"><div class="clabel">${l}</div><div class="cval ${c}">${v}</div></div>`).join('');
}

function renderOpen(d){
  const open=(d.trades||[]).filter(t=>t.open);
  if(!open.length){$('open').innerHTML='<div class="empty">No open positions</div>';return;}
  $('open').innerHTML=`<table><thead><tr>
    <th>Symbol</th><th>Entry Price</th><th>Qty</th><th>Regime</th><th>Bull Prob</th><th>Entry Time</th>
  </tr></thead><tbody>${open.map(t=>`<tr>
    <td>${t.symbol}</td><td>$${f(t.entry_price,4)}</td><td>${f(t.quantity,6)}</td>
    <td>${badge(t.regime)}</td><td>${((t.bull_prob||0)*100).toFixed(1)}%</td>
    <td style="color:var(--muted)">${(t.entry_time||'').slice(0,19).replace('T',' ')}</td>
  </tr>`).join('')}</tbody></table>`;
}

function renderClosed(d){
  const closed=(d.trades||[]).filter(t=>!t.open)
    .sort((a,b)=>(b.exit_time||'').localeCompare(a.exit_time||'')).slice(0,100);
  if(!closed.length){$('closed').innerHTML='<div class="empty">No closed trades yet</div>';return;}
  $('closed').innerHTML=`<table><thead><tr>
    <th>Symbol</th><th>Entry</th><th>Exit</th><th>Qty</th><th>P&amp;L USD</th><th>P&amp;L %</th><th>Reason</th><th>Exit Time</th>
  </tr></thead><tbody>${closed.map(t=>`<tr>
    <td>${t.symbol}</td><td>$${f(t.entry_price,4)}</td><td>$${f(t.exit_price,4)}</td>
    <td>${f(t.quantity,6)}</td>
    <td class="${cls(t.pnl_usd)}">${usd(t.pnl_usd)}</td>
    <td class="${cls(t.pnl_pct)}">${pct(t.pnl_pct)}</td>
    <td style="color:var(--muted);font-size:11px">${t.exit_reason||'&#8212;'}</td>
    <td style="color:var(--muted);font-size:11px">${(t.exit_time||'').slice(0,19).replace('T',' ')}</td>
  </tr>`).join('')}</tbody></table>`;
}

async function refresh(){
  try{
    const r=await fetch('/data');
    if(!r.ok)throw new Error('HTTP '+r.status);
    const d=await r.json();
    renderCards(d);renderOpen(d);renderClosed(d);
    $('ts').textContent='Last updated: '+new Date().toUTCString();
    $('err').textContent='';
  }catch(e){$('err').textContent='&#9888; '+e.message;}
}
refresh();
setInterval(refresh,10000);
</script>
</body>
</html>"""

class _JournalHandler(http.server.BaseHTTPRequestHandler):
    """Minimal single-endpoint server: GET /data → journal JSON (CORS open)."""

    _journal: TradeJournal | None = None  # set before server starts

    def do_GET(self) -> None:
        path = self.path.split("?")[0].rstrip("/") or "/"

        if path == "/data":
            body = json.dumps(
                self._journal.to_dict() if self._journal else {}
            ).encode()
            self._send(200, "application/json", body)

        elif path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8",
                       _DASHBOARD_HTML.encode())

        else:
            self._send(404, "text/plain", b"Not Found")

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type",                content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length",              str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_) -> None:
        pass

def start_dashboard_server(journal: TradeJournal, port: int = 8080) -> None:
    _JournalHandler._journal = journal
    server = http.server.HTTPServer(("0.0.0.0", port), _JournalHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("Dashboard data server → http://0.0.0.0:%d/data", port)


# ══════════════════════════════════════════════════════════════════════════════
#  Roostoo execution client
# ══════════════════════════════════════════════════════════════════════════════

class RoostooExecClientConfig(LiveExecClientConfig, frozen=True):
    """
    Execution client config for the Roostoo mock exchange.

    Note: there is no `pair` field here.  Each order carries its own
    instrument_id; _submit_order() looks up the correct Roostoo pair from
    cfg.INSTRUMENT_MAP at order time, so a single execution client instance
    correctly handles every instrument in config.TRADING_INSTRUMENTS.
    """
    api_key:    str = cfg.ROOSTOO_API_KEY
    secret_key: str = cfg.ROOSTOO_SECRET_KEY
    base_url:   str = cfg.ROOSTOO_BASE_URL


class RoostooLiveExecutionClient(LiveExecutionClient):
    """
    Routes NautilusTrader SubmitOrder commands to the Roostoo REST API.

    Supports all pairs listed in config.TRADING_INSTRUMENTS.  The Roostoo
    pair is resolved per-order from cfg.INSTRUMENT_MAP so no per-instrument
    client instances are required.

    Only MARKET orders are sent.  Fills are confirmed synchronously from
    the Roostoo JSON response so no polling loop is needed.

    Memory notes (cloud deployment)
    ---------------------------------
    • No internal order queue is maintained.
    • Account state is synced only on connect and after each fill.
    • gc.collect() is called after every fill event.
    """

    _ACCOUNT_ID = AccountId("BINANCE-001")

    def __init__(
        self,
        loop:      asyncio.AbstractEventLoop,
        client_id: ClientId,
        msgbus:    MessageBus,
        cache:     Cache,
        clock:     LiveClock,
        roostoo:   RoostooClient,
        config:    RoostooExecClientConfig,
    ) -> None:
        _stub_provider = InstrumentProvider(config=InstrumentProviderConfig())

        super().__init__(
            loop                = loop,
            client_id           = client_id,
            venue               = Venue("BINANCE"),
            oms_type            = OmsType.NETTING,
            account_type        = AccountType.CASH,
            base_currency       = None,
            msgbus              = msgbus,
            cache               = cache,
            clock               = clock,
            instrument_provider = _stub_provider,
        )
        self._roostoo = roostoo

    async def _connect(self) -> None:
        logger.info("Connecting Roostoo execution client…")
        self._set_account_id(self._ACCOUNT_ID)
        self._sync_account()
        self._set_connected(True)
        logger.info("Roostoo execution client connected.")

    async def _disconnect(self) -> None:
        self._set_connected(False)
        logger.info("Roostoo execution client disconnected.")

    def _sync_account(self) -> None:
        balances: list[AccountBalance] = []

        data = self._roostoo.get_balance()
        if not data or not data.get("Success"):
            logger.warning(
                "Could not sync Roostoo account state (API unreachable or "
                "Success=false) — registering zero-USD fallback account so "
                "that reconciliation does not crash."
            )
        else:
            wallet = data.get("SpotWallet", {})
            for ccy_code, amounts in wallet.items():
                try:
                    ccy  = Currency.from_str(ccy_code)
                    free = float(amounts.get("Free", 0.0))
                    lock = float(amounts.get("Lock", 0.0))
                    balances.append(AccountBalance(
                        total =Money(free + lock, ccy),
                        locked=Money(lock,        ccy),
                        free  =Money(free,        ccy),
                    ))
                except Exception as e:
                    logger.debug("Skipping currency %s: %s", ccy_code, e)

        if not balances:
            try:
                usd = Currency.from_str("USD")
                balances = [AccountBalance(
                    total =Money(0.0, usd),
                    locked=Money(0.0, usd),
                    free  =Money(0.0, usd),
                )]
            except Exception as exc:
                logger.error(
                    "Could not construct fallback USD balance: %s — "
                    "reconciliation may still crash.", exc,
                )

        if balances:
            self.generate_account_state(
                balances=balances,
                margins=[],
                reported=True,
                ts_event=self._clock.timestamp_ns(),
            )

    async def _submit_order(self, command) -> None:
        order = command.order
        if order.order_type not in (OrderType.MARKET, OrderType.LIMIT):
            logger.warning("Order type %s not supported.", order.order_type)
            return

        # Resolve the Roostoo trading pair from the order's instrument_id.
        # cfg.INSTRUMENT_MAP covers every pair in config.TRADING_INSTRUMENTS,
        # so this works correctly for BTC, ETH, BNB, DOGE, LINK, or any other
        # pair added to that list in the future without touching this file.
        instr_key = str(order.instrument_id)
        instr_cfg = cfg.INSTRUMENT_MAP.get(instr_key)
        if instr_cfg is None:
            logger.error(
                "No Roostoo pair configured for instrument %s — rejecting order.",
                instr_key,
            )
            self.generate_order_rejected(
                order=order,
                reason=f"No Roostoo pair configured for {instr_key}",
                ts_event=self._clock.timestamp_ns(),
            )
            return
        roostoo_pair = instr_cfg["roostoo_pair"]

        side  = "BUY" if order.side == OrderSide.BUY else "SELL"
        qty   = float(order.quantity)
        price = float(order.price) if order.order_type == OrderType.LIMIT else None

        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._roostoo.place_order(
                pair=roostoo_pair,
                side=side,
                quantity=qty,
                price=price,
            ),
        )

        if not result:
            self.generate_order_rejected(
                order=order, reason="No response from Roostoo API",
                ts_event=self._clock.timestamp_ns(),
            )
            return
        if not result.get("Success"):
            self.generate_order_rejected(
                order=order, reason=result.get("ErrMsg", "Roostoo error"),
                ts_event=self._clock.timestamp_ns(),
            )
            return

        detail     = result.get("OrderDetail", {})
        status     = detail.get("Status", "")
        roostoo_id = str(detail.get("OrderID", uuid4()))
        filled_qty = float(detail.get("FilledQuantity",  qty))
        filled_px  = float(detail.get("FilledAverPrice", detail.get("Price", 0.0)))
        commission = float(detail.get("CommissionChargeValue", 0.0))
        comm_ccy   = detail.get("CommissionCoin", "USD")

        instrument = self._cache.instrument(order.instrument_id)
        if instrument is None:
            logger.error(
                "Instrument %s not found in cache — cannot report fill.",
                order.instrument_id,
            )
            return

        try:
            comm_currency = Currency.from_str(comm_ccy)
        except Exception:
            comm_currency = instrument.quote_currency

        if status == "FILLED":
            self.generate_order_filled(
                order          = order,
                venue_order_id = VenueOrderId(roostoo_id),
                trade_id       = TradeId(str(uuid4())),
                position_id    = None,
                last_qty       = Quantity(filled_qty, instrument.size_precision),
                last_px        = Price(filled_px,     instrument.price_precision),
                quote_currency = instrument.quote_currency,
                commission     = Money(commission, comm_currency),
                liquidity_side = LiquiditySide.TAKER,
                ts_event       = self._clock.timestamp_ns(),
            )
            self._sync_account()
            gc.collect()
        else:
            logger.warning("Order %s status=%s — not FILLED.", roostoo_id, status)

    async def _cancel_order(self, command) -> None:
        vid = command.venue_order_id
        if vid:
            self._roostoo.cancel_order(order_id=str(vid))

    async def _cancel_all_orders(self, command) -> None:
        self._roostoo.cancel_order()

    async def _submit_order_list(self,   command) -> None: pass
    async def _modify_order(self,        command) -> None: pass
    async def _batch_cancel_orders(self, command) -> None: pass
    async def _query_order(self,         command) -> None: pass

    async def generate_order_status_reports(
        self,
        instrument_id=None,
        start=None,
        end=None,
        open_only: bool = False,
    ) -> list:
        return []

    async def generate_fill_reports(
        self,
        instrument_id=None,
        venue_order_id=None,
        start=None,
        end=None,
    ) -> list:
        return []

    async def generate_position_status_reports(
        self,
        instrument_id=None,
        start=None,
        end=None,
    ) -> list:
        return []


class RoostooLiveExecClientFactory(LiveExecClientFactory):
    @staticmethod
    def create(
        loop:   asyncio.AbstractEventLoop,
        name:   str,
        config: RoostooExecClientConfig,
        msgbus: MessageBus,
        cache:  Cache,
        clock:  LiveClock,
        **kw,
    ) -> RoostooLiveExecutionClient:
        roostoo = RoostooClient(
            api_key   =config.api_key,
            secret_key=config.secret_key,
            base_url  =config.base_url,
        )
        return RoostooLiveExecutionClient(
            loop=loop, client_id=ClientId(name), msgbus=msgbus,
            cache=cache, clock=clock, roostoo=roostoo, config=config,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Pre-flight checks
# ══════════════════════════════════════════════════════════════════════════════

def preflight_checks() -> bool:
    ok = True

    for inst_cfg in cfg.INSTRUMENTS:
        if not inst_cfg["hmm_model_path"].exists():
            logger.error(
                "HMM model not found: %s — run train_models.py first.",
                inst_cfg["hmm_model_path"],
            )
            ok = False

    if cfg.ROOSTOO_API_KEY == "YOUR_API_KEY_HERE":
        logger.error("Roostoo API key not set in config.py.")
        ok = False

    client = RoostooClient(cfg.ROOSTOO_API_KEY, cfg.ROOSTOO_SECRET_KEY, cfg.ROOSTOO_BASE_URL)
    bal    = client.get_balance()
    if not bal or not bal.get("Success"):
        logger.error("Cannot reach Roostoo API. Check credentials and network.")
        ok = False
    else:
        w   = bal.get("SpotWallet", {})
        usd = float(w.get("USD", {}).get("Free", 0.0))
        btc = float(w.get("BTC", {}).get("Free", 0.0))
        logger.info("Roostoo balance OK | USD free=%.2f | BTC free=%.6f", usd, btc)

    return ok


# ══════════════════════════════════════════════════════════════════════════════
#  TradingNode assembly
# ══════════════════════════════════════════════════════════════════════════════

def build_trading_node() -> tuple[TradingNode, TradeJournal]:

    journal = TradeJournal(
        path             = Path("trade_journal.json"),
        starting_balance = cfg.STARTING_BALANCE,
    )

    # ── Step 1: Resolve Binance instrument specs for all configured pairs ─────
    #
    # Mirrors backtest.py exactly: get_instruments_sync() checks the in-memory
    # cache, then the disk cache at INSTRUMENT_CACHE_PATH, and only hits the
    # live Binance exchange-info endpoint on the very first run.  Subsequent
    # launches load from the pickle file in < 1 ms.
    #
    # This step is what gives us accurate price_precision and size_precision
    # for every pair — not just BTCUSDT.
    all_symbols = [i["binance_symbol"] for i in cfg.INSTRUMENTS]
    logger.info(
        "Resolving instrument definitions for: %s  (disk cache: %s)",
        all_symbols,
        cfg.INSTRUMENT_CACHE_PATH,
    )
    binance_instruments = get_instruments_sync(
        symbols=all_symbols,
        venue=cfg.VENUE,
        cache_path=cfg.INSTRUMENT_CACHE_PATH,
    )
    if not binance_instruments:
        raise RuntimeError(
            "Could not resolve any Binance instrument definitions.  "
            "Check your internet connection or run fetch_data.py first."
        )

    # ── Step 2: Determine which instruments are actually tradeable ────────────
    #
    # An instrument is active only when BOTH conditions hold:
    #   (a) the HMM model file exists on disk
    #   (b) the Binance instrument spec was successfully resolved
    #
    # This is the same guard used in backtest.py, so the two modes stay in sync.
    active: list[dict] = []
    for inst_cfg in cfg.INSTRUMENTS:
        symbol = inst_cfg["binance_symbol"]
        if not inst_cfg["hmm_model_path"].exists():
            logger.warning(
                "[%s] Skipping — HMM model not found at %s.  "
                "Run train_models.py first.",
                symbol, inst_cfg["hmm_model_path"],
            )
            continue
        if symbol not in binance_instruments:
            logger.warning(
                "[%s] Skipping — instrument spec could not be resolved from Binance.",
                symbol,
            )
            continue
        active.append(inst_cfg)
        logger.info(
            "[%s] Instrument resolved: price_precision=%d  size_precision=%d",
            symbol,
            binance_instruments[symbol].price_precision,
            binance_instruments[symbol].size_precision,
        )

    if not active:
        raise RuntimeError(
            "No tradeable instruments remain after filtering.  "
            "Check that HMM models exist and Binance symbols are correct."
        )

    active_syms = [i["binance_symbol"] for i in active]
    logger.info("Active instruments for live trading: %s", active_syms)

    # ── Step 3: Build the TradingNode ─────────────────────────────────────────
    #
    # InstrumentProviderConfig.load_ids tells the Binance data client exactly
    # which instrument specs to load on connect — one entry per active pair.
    # Without this, load_all=False would leave the cache empty until the first
    # subscription fires, which can race with on_start() strategy calls.
    active_instrument_ids = frozenset(i["instrument_id_str"] for i in active)

    node_config = TradingNodeConfig(
        trader_id="ROOSTOO-HMM-001",
        data_clients={
            "BINANCE": BinanceDataClientConfig(
                api_key      = None,
                api_secret   = None,
                account_type = BinanceAccountType.SPOT,
                instrument_provider=InstrumentProviderConfig(
                    load_all  = False,
                    load_ids  = active_instrument_ids,
                ),
            ),
        },
        exec_clients={
            # One execution client handles all pairs — the Roostoo pair is
            # resolved per-order inside _submit_order() via cfg.INSTRUMENT_MAP.
            "BINANCE": RoostooExecClientConfig(
                api_key    = cfg.ROOSTOO_API_KEY,
                secret_key = cfg.ROOSTOO_SECRET_KEY,
                base_url   = cfg.ROOSTOO_BASE_URL,
            ),
        },
        timeout_connection     = 30.0,
        timeout_reconciliation = 10.0,
        timeout_portfolio      = 10.0,
        timeout_disconnection  = 10.0,
        timeout_post_stop      = 5.0,
    )

    node = TradingNode(config=node_config)
    node.add_data_client_factory("BINANCE", BinanceLiveDataClientFactory)
    node.add_exec_client_factory("BINANCE", RoostooLiveExecClientFactory)

    # ── Step 4: Register one strategy per active instrument ───────────────────
    #
    # Identical pattern to backtest.py: iterate active instruments, build a
    # per-instrument HMMStrategyConfig, and add a separate HMMStrategy instance
    # for each one.  The shared CapitalAllocator and TradeJournal are passed
    # through to every strategy so capital and journaling stay coordinated.
    allocator = CapitalAllocator()

    for inst_cfg in active:
        strategy_cfg = HMMStrategyConfig(
            instrument_id           = inst_cfg["instrument_id_str"],
            bar_type                = inst_cfg["bar_type_str"],
            hmm_model_path          = str(inst_cfg["hmm_model_path"]),
            min_bull_proba          = cfg.MIN_BULL_PROBA,
            min_kelly_fraction      = cfg.MIN_KELLY_FRACTION,
            trend_ema_bars          = cfg.TREND_EMA_BARS,
            kelly_fraction          = cfg.KELLY_FRACTION,
            max_position_pct        = cfg.MAX_POSITION_PCT,
            commission_rate         = cfg.COMMISSION_RATE,
            take_profit_pct         = cfg.TAKE_PROFIT_PCT,
            trail_bull_pct          = cfg.TRAIL_BULL_PCT,
            trail_sideways_pct      = cfg.TRAIL_SIDEWAYS_PCT,
            trail_bear_pct          = cfg.TRAIL_BEAR_PCT,
            bear_exit_proba         = cfg.BEAR_EXIT_PROBA,
            bear_exit_consecutive   = cfg.BEAR_EXIT_CONSECUTIVE,
            min_bars_between_trades = cfg.MIN_BARS_BETWEEN_TRADES,
            hmm_min_history         = cfg.HMM_MIN_HISTORY,
            trend_lookback_bars     = cfg.TREND_LOOKBACK_BARS,
            bull_entry_consecutive  = cfg.BULL_ENTRY_CONSECUTIVE,
            max_holding_bars        = cfg.MAX_HOLDING_BARS,
            starting_balance_usd    = cfg.STARTING_BALANCE,
        )
        node.trader.add_strategy(
            HMMStrategy(config=strategy_cfg, allocator=allocator, journal=journal)
        )
        logger.info("Strategy registered: %s", inst_cfg["binance_symbol"])

    node.build()
    return node, journal


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    logger.info("=" * 60)
    logger.info("  HMM Regime Strategy — Live Trading")
    logger.info("  Binance data  |  Roostoo execution")
    logger.info("=" * 60)

    if not preflight_checks():
        logger.critical("Pre-flight checks failed. Aborting.")
        return

    node, journal = build_trading_node()
    start_dashboard_server(journal, port=8080)

    logger.info(
        "Each strategy will request %d historical bars (~%.1f h) from Binance "
        "REST on start — ready to trade on first live bar.",
        cfg.HMM_MIN_HISTORY - 1,
        (cfg.HMM_MIN_HISTORY - 1) * 15 / 60,
    )

    try:
        node.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — stopping…")
    finally:
        node.stop()
        node.dispose()
        gc.collect()
        logger.info("Node stopped cleanly.")


if __name__ == "__main__":
    main()
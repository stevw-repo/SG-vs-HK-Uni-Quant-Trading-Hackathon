"""
live_trading.py
Live trading for ALL instruments defined in config.TRADING_INSTRUMENTS.

  - One HMMLSTMStrategy is created per instrument.
  - A single RoostooLiveExecutionClient handles all instruments; the correct
    Roostoo pair is resolved per-order from the order's instrument_id using
    a pair_map populated from config at startup.
  - Adding a new asset requires only an edit to config.TRADING_INSTRUMENTS
    — no changes are needed in this file.

Memory-optimisation notes
--------------------------
  - All price/volume buffers are bounded deques (maxlen).
  - LSTM inference runs on CPU with torch.no_grad().
  - gc.collect() is triggered after fills and periodically.

Usage
-----
    python live_trading.py
"""

from __future__ import annotations

import asyncio
import gc
import logging
from uuid import uuid4

from nautilus_trader.adapters.binance.common.enums import BinanceAccountType
from nautilus_trader.adapters.binance.config import BinanceLiveDataClientConfig
from nautilus_trader.adapters.binance.factories import BinanceLiveDataClientFactory
from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.component import LiveClock, MessageBus
from nautilus_trader.config import (
    InstrumentProviderConfig,
    LiveExecClientConfig,
    TradingNodeConfig,
)
from nautilus_trader.live.execution_client import LiveExecutionClient
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.currencies import Currency
from nautilus_trader.model.enums import AccountType, LiquiditySide, OmsType, OrderSide, OrderType
from nautilus_trader.model.identifiers import (
    ClientId,
    TradeId,
    VenueOrderId,
    Venue,
)
from nautilus_trader.model.objects import AccountBalance, Money, Price, Quantity

import config as cfg
from strategy.hmm_lstm_strategy import HMMLSTMStrategy, HMMLSTMStrategyConfig
from strategy.roostoo_client import RoostooClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("live_trading")


# ══════════════════════════════════════════════════════════════════════════════
#  Roostoo Execution Client
# ══════════════════════════════════════════════════════════════════════════════

class RoostooExecClientConfig(LiveExecClientConfig, frozen=True):
    api_key:   str
    secret_key: str
    base_url:  str               = "https://mock-api.roostoo.com"
    # Maps instrument_id_str → roostoo_pair, e.g.
    #   {"BTCUSDT.BINANCE": "BTC/USD", "ETHUSDT.BINANCE": "ETH/USD"}
    # Populated automatically from cfg.INSTRUMENTS — do not set manually.
    pair_map:  dict[str, str]    = {}


class RoostooLiveExecutionClient(LiveExecutionClient):
    """
    Routes NautilusTrader SubmitOrder commands to the Roostoo REST API.

    The correct Roostoo trading pair is resolved per-order from the order's
    instrument_id via the pair_map injected at construction.  No instrument-
    specific logic lives here — adding a new asset only requires an entry in
    config.TRADING_INSTRUMENTS.
    """

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
        super().__init__(
            loop=loop,
            client_id=client_id,
            venue=Venue("BINANCE"),
            oms_type=OmsType.NETTING,
            account_type=AccountType.CASH,
            base_currency=None,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            instrument_provider=None,
        )
        self._roostoo  = roostoo
        self._pair_map = dict(config.pair_map)   # instrument_id_str → roostoo_pair

    # ── Connection ────────────────────────────────────────────────────────────

    async def _connect(self) -> None:
        logger.info("Connecting Roostoo execution client …")
        logger.info("  Instrument → pair mappings: %s", self._pair_map)
        self._sync_account()
        self._set_connected(True)
        logger.info("Roostoo execution client connected.")

    async def _disconnect(self) -> None:
        self._set_connected(False)
        logger.info("Roostoo execution client disconnected.")

    # ── Account ───────────────────────────────────────────────────────────────

    def _sync_account(self) -> None:
        """Pull Roostoo balance and push an AccountState into NautilusTrader."""
        data = self._roostoo.get_balance()
        if not data or not data.get("Success"):
            logger.warning("Could not sync Roostoo account state.")
            return

        wallet   = data.get("Wallet", {})
        balances = []
        for ccy_code, amounts in wallet.items():
            try:
                ccy  = Currency.from_str(ccy_code)
                free = float(amounts.get("Free", 0.0))
                lock = float(amounts.get("Lock", 0.0))
                balances.append(AccountBalance(
                    total=Money(free + lock, ccy),
                    locked=Money(lock,       ccy),
                    free=Money(free,         ccy),
                ))
            except Exception as e:
                logger.debug("Skipping currency %s: %s", ccy_code, e)

        if balances:
            self.generate_account_state(
                balances=balances,
                margins=[],
                reported=True,
                ts_event=self._clock.timestamp_ns(),
            )

    # ── Order submission ──────────────────────────────────────────────────────

    async def _submit_order(self, command) -> None:
        """Send a MARKET or LIMIT order to Roostoo and report the fill."""
        order = command.order

        if order.order_type not in (OrderType.MARKET, OrderType.LIMIT):
            self._log.warning(
                "Order type %s not supported — skipping.", order.order_type
            )
            return

        instrument_id_str = str(order.instrument_id)
        pair = self._pair_map.get(instrument_id_str)
        if pair is None:
            logger.error(
                "No Roostoo pair configured for '%s' — order rejected. "
                "Check TRADING_INSTRUMENTS in config.py.",
                instrument_id_str,
            )
            self.generate_order_rejected(
                order=order,
                reason=f"No Roostoo pair for {instrument_id_str}",
                ts_event=self._clock.timestamp_ns(),
            )
            return

        side     = "BUY" if order.side == OrderSide.BUY else "SELL"
        quantity = float(order.quantity)
        price    = float(order.price) if order.order_type == OrderType.LIMIT else None

        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._roostoo.place_order(
                pair=pair,
                side=side,
                quantity=quantity,
                price=price,
            ),
        )

        if not result:
            self.generate_order_rejected(
                order=order,
                reason="No response from Roostoo API",
                ts_event=self._clock.timestamp_ns(),
            )
            return

        if not result.get("Success"):
            self.generate_order_rejected(
                order=order,
                reason=result.get("ErrMsg", "Roostoo error"),
                ts_event=self._clock.timestamp_ns(),
            )
            return

        detail       = result.get("OrderDetail", {})
        status       = detail.get("Status", "")
        roostoo_id   = str(detail.get("OrderID", uuid4()))
        filled_qty   = float(detail.get("FilledQuantity",  quantity))
        filled_price = float(detail.get("FilledAverPrice", detail.get("Price", 0.0)))
        commission   = float(detail.get("CommissionChargeValue", 0.0))
        comm_ccy     = detail.get("CommissionCoin", "USD")

        instrument = self._cache.instrument(order.instrument_id)
        if instrument is None:
            logger.error(
                "Instrument '%s' not in cache — cannot report fill.",
                instrument_id_str,
            )
            return

        try:
            comm_currency = Currency.from_str(comm_ccy)
        except Exception:
            comm_currency = instrument.quote_currency

        if status == "FILLED":
            self.generate_order_filled(
                order=order,
                venue_order_id=VenueOrderId(roostoo_id),
                trade_id=TradeId(str(uuid4())),
                position_id=None,
                last_qty=Quantity(filled_qty,   instrument.size_precision),
                last_px=Price(filled_price,     instrument.price_precision),
                quote_currency=instrument.quote_currency,
                commission=Money(commission,    comm_currency),
                liquidity_side=LiquiditySide.TAKER,
                ts_event=self._clock.timestamp_ns(),
            )
            self._sync_account()
            gc.collect()

        elif status == "PENDING":
            logger.warning(
                "Order %s is PENDING on Roostoo. "
                "Use MARKET orders for guaranteed execution.",
                roostoo_id,
            )

    # ── Cancel ────────────────────────────────────────────────────────────────

    async def _cancel_order(self, command) -> None:
        vid = command.venue_order_id
        if vid:
            self._roostoo.cancel_order(order_id=str(vid))

    async def _cancel_all_orders(self, command) -> None:
        self._roostoo.cancel_order()

    # ── Unused stubs ──────────────────────────────────────────────────────────

    async def _submit_order_list(self, command) -> None: pass
    async def _modify_order(self, command) -> None: pass
    async def _batch_cancel_orders(self, command) -> None: pass
    async def _query_order(self, command) -> None: pass


# ══════════════════════════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════════════════════════

class RoostooLiveExecClientFactory:
    """Factory consumed by TradingNode to build the Roostoo execution client."""

    @staticmethod
    def create(
        loop:    asyncio.AbstractEventLoop,
        name:    str,
        config:  RoostooExecClientConfig,
        msgbus:  MessageBus,
        cache:   Cache,
        clock:   LiveClock,
        **kwargs,
    ) -> RoostooLiveExecutionClient:
        roostoo = RoostooClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
            base_url=config.base_url,
        )
        return RoostooLiveExecutionClient(
            loop=loop,
            client_id=ClientId(name),
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            roostoo=roostoo,
            config=config,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Pre-flight checks
# ══════════════════════════════════════════════════════════════════════════════

def preflight_checks() -> bool:
    """Verify model files exist for every instrument and Roostoo is reachable."""
    ok = True

    for inst_cfg in cfg.INSTRUMENTS:
        symbol = inst_cfg["binance_symbol"]
        if not inst_cfg["hmm_model_path"].exists():
            logger.error(
                "[%s] HMM model not found at %s — run train_models.py first.",
                symbol, inst_cfg["hmm_model_path"],
            )
            ok = False
        if not inst_cfg["lstm_model_path"].exists():
            logger.error(
                "[%s] LSTM model not found at %s — run train_models.py first.",
                symbol, inst_cfg["lstm_model_path"],
            )
            ok = False

    if not cfg.FEAR_GREED_PATH.exists():
        logger.warning(
            "Fear & Greed file not found at %s — "
            "strategies will use 0.5 neutral fallback. "
            "Run fetch_data.py to populate it.",
            cfg.FEAR_GREED_PATH,
        )

    if cfg.ROOSTOO_API_KEY == "YOUR_API_KEY_HERE":
        logger.error("Roostoo API key not set in config.py.")
        ok = False

    client = RoostooClient(
        cfg.ROOSTOO_API_KEY, cfg.ROOSTOO_SECRET_KEY, cfg.ROOSTOO_BASE_URL
    )
    bal = client.get_balance()
    if not bal or not bal.get("Success"):
        logger.error("Cannot reach Roostoo API. Check credentials and network.")
        ok = False
    else:
        wallet   = bal.get("Wallet", {})
        usd_free = float(wallet.get("USD", {}).get("Free", 0.0))
        logger.info("Roostoo balance OK | USD free=%.2f", usd_free)
        for inst_cfg in cfg.INSTRUMENTS:
            ticker_upper = inst_cfg["ticker"].upper()
            asset_free   = float(wallet.get(ticker_upper, {}).get("Free", 0.0))
            logger.info("  %s free=%.8f", ticker_upper, asset_free)

    return ok


# ══════════════════════════════════════════════════════════════════════════════
#  Node builder
# ══════════════════════════════════════════════════════════════════════════════

def build_trading_node() -> TradingNode:
    """Assemble and return a fully configured TradingNode (not yet started)."""

    node_config = TradingNodeConfig(
        trader_id="ROOSTOO-TRADER-001",

        # Binance data client — public endpoints, no API key required
        data_clients={
            "BINANCE": BinanceLiveDataClientConfig(
                api_key=None,
                api_secret=None,
                account_type=BinanceAccountType.SPOT,
                instrument_provider=InstrumentProviderConfig(load_all=False),
            ),
        },

        exec_clients={
            "BINANCE": RoostooExecClientConfig(
                api_key=cfg.ROOSTOO_API_KEY,
                secret_key=cfg.ROOSTOO_SECRET_KEY,
                base_url=cfg.ROOSTOO_BASE_URL,
                pair_map={
                    inst["instrument_id_str"]: inst["roostoo_pair"]
                    for inst in cfg.INSTRUMENTS
                },
            ),
        },
        timeout_connection=30.0,
        timeout_reconciliation=10.0,
        timeout_portfolio=10.0,
        timeout_disconnection=10.0,
        timeout_post_stop=5.0,
    )

    node = TradingNode(config=node_config)
    node.add_data_client_factory("BINANCE", BinanceLiveDataClientFactory)
    node.add_exec_client_factory("BINANCE", RoostooLiveExecClientFactory)

    # One strategy per instrument — fear_greed_path is forwarded so each
    # strategy instance independently loads and queries the daily F&G index.
    for inst_cfg in cfg.INSTRUMENTS:
        strategy_config = HMMLSTMStrategyConfig(
            instrument_id=inst_cfg["instrument_id_str"],
            bar_type=inst_cfg["bar_type_str"],
            hmm_model_path=str(inst_cfg["hmm_model_path"]),
            lstm_model_path=str(inst_cfg["lstm_model_path"]),
            fear_greed_path=str(cfg.FEAR_GREED_PATH),
            min_confidence_buy=cfg.MIN_CONFIDENCE_ENTRY,
            min_confidence_sell=cfg.MIN_CONFIDENCE_EXIT,
            kelly_fraction=cfg.KELLY_FRACTION,
            max_position_pct=cfg.MAX_POSITION_PCT,
            win_loss_ratio=cfg.EXPECTED_WIN_LOSS_RATIO,
            stop_loss_pct=cfg.STOP_LOSS_PCT,
            take_profit_pct=cfg.TAKE_PROFIT_PCT,
            trailing_stop_pct=cfg.TRAILING_STOP_PCT,
            hmm_min_history=cfg.HMM_MIN_HISTORY,
            lstm_lookback=cfg.LSTM_LOOKBACK,
            starting_balance_usd=cfg.STARTING_BALANCE,
        )
        node.trader.add_strategy(HMMLSTMStrategy(config=strategy_config))
        logger.info(
            "Strategy registered: %s → %s",
            inst_cfg["binance_symbol"], inst_cfg["roostoo_pair"],
        )

    node.build()
    return node


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    instruments_str = ", ".join(i["binance_symbol"] for i in cfg.INSTRUMENTS)

    logger.info("=" * 60)
    logger.info("  HMM-LSTM Crypto Algo — Live Trading")
    logger.info("  Binance data  |  Roostoo execution")
    logger.info("  Instruments : %s", instruments_str)
    logger.info("=" * 60)

    if not preflight_checks():
        logger.critical("Pre-flight checks failed. Aborting.")
        return

    node = build_trading_node()

    logger.info("Starting TradingNode …")
    logger.info(
        "Strategies activate after a %d-bar warm-up (~%.1f h).",
        cfg.HMM_MIN_HISTORY + cfg.LSTM_LOOKBACK,
        (cfg.HMM_MIN_HISTORY + cfg.LSTM_LOOKBACK) / 60,
    )

    try:
        node.run()          # blocking; Ctrl-C to stop
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — stopping node …")
    finally:
        node.stop()
        node.dispose()
        gc.collect()
        logger.info("Node stopped cleanly.")


if __name__ == "__main__":
    main()
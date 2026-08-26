"""Tests for the execution layer: order lifecycle, OrderManager, brokers."""
from __future__ import annotations

import os
from unittest import mock

import pytest

from src.trading.execution.broker import (
    Account,
    AlpacaBroker,
    BrokerError,
    BrokerPosition,
    LiveTradingBlocked,
    MockBroker,
)
from src.trading.execution.orders import (
    InvalidOrderTransition,
    Order,
    OrderManager,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
    limit_order,
    market_order,
    stop_order,
)


# ── Order construction ────────────────────────────────────────────────────────

def test_market_order_buy_from_positive_qty():
    o = market_order("SPY", 100)
    assert o.side == OrderSide.BUY and o.quantity == 100


def test_market_order_sell_from_negative_qty():
    o = market_order("SPY", -100)
    assert o.side == OrderSide.SELL and o.quantity == 100


def test_order_rejects_zero_quantity():
    with pytest.raises(ValueError):
        Order("SPY", OrderSide.BUY, 0)


def test_limit_order_requires_limit_price():
    with pytest.raises(ValueError):
        Order("SPY", OrderSide.BUY, 10, OrderType.LIMIT)


def test_stop_order_requires_stop_price():
    with pytest.raises(ValueError):
        Order("SPY", OrderSide.BUY, 10, OrderType.STOP)


def test_stop_order_factory():
    o = stop_order("SPY", -50, stop_price=95.0)
    assert o.side == OrderSide.SELL and o.stop_price == 95.0


def test_client_order_ids_are_unique():
    assert market_order("SPY", 1).client_order_id != market_order("SPY", 1).client_order_id


# ── Status transitions ────────────────────────────────────────────────────────

def test_new_order_is_open():
    o = market_order("SPY", 10)
    assert o.is_open and not o.is_terminal


def test_valid_transition_sets_submitted_at():
    o = market_order("SPY", 10)
    o.transition_to(OrderStatus.SUBMITTED)
    assert o.submitted_at is not None


def test_illegal_transition_raises():
    o = market_order("SPY", 10)
    with pytest.raises(InvalidOrderTransition):
        o.transition_to(OrderStatus.FILLED)      # cannot skip SUBMITTED


def test_terminal_order_cannot_transition():
    o = market_order("SPY", 10)
    o.cancel()
    assert o.is_terminal
    with pytest.raises(InvalidOrderTransition):
        o.transition_to(OrderStatus.SUBMITTED)


def test_terminal_sets_closed_at():
    o = market_order("SPY", 10)
    o.cancel()
    assert o.closed_at is not None


def test_reject_records_reason():
    o = market_order("SPY", 10)
    o.reject("insufficient buying power")
    assert o.status == OrderStatus.REJECTED
    assert o.reject_reason == "insufficient buying power"


# ── Fills ─────────────────────────────────────────────────────────────────────

def test_full_fill():
    o = market_order("SPY", 100)
    o.transition_to(OrderStatus.SUBMITTED)
    o.apply_fill(100, 50.0)
    assert o.status == OrderStatus.FILLED
    assert o.remaining_quantity == 0


def test_partial_fill_then_complete():
    o = market_order("SPY", 100)
    o.transition_to(OrderStatus.SUBMITTED)
    o.apply_fill(30, 100.0)
    assert o.status == OrderStatus.PARTIALLY_FILLED
    assert o.remaining_quantity == 70
    o.apply_fill(70, 110.0)
    assert o.status == OrderStatus.FILLED


def test_partial_fills_use_volume_weighted_price():
    o = market_order("SPY", 100)
    o.transition_to(OrderStatus.SUBMITTED)
    o.apply_fill(40, 100.0)
    o.apply_fill(60, 101.0)
    # (40*100 + 60*101) / 100 = 100.6
    assert o.avg_fill_price == pytest.approx(100.6)


def test_overfill_rejected():
    o = market_order("SPY", 100)
    o.transition_to(OrderStatus.SUBMITTED)
    with pytest.raises(ValueError):
        o.apply_fill(150, 100.0)


def test_fill_on_terminal_order_raises():
    o = market_order("SPY", 100)
    o.cancel()
    with pytest.raises(InvalidOrderTransition):
        o.apply_fill(10, 100.0)


def test_signed_filled_quantity_sell_is_negative():
    o = market_order("SPY", -100)
    o.transition_to(OrderStatus.SUBMITTED)
    o.apply_fill(100, 50.0)
    assert o.signed_filled_quantity == -100


def test_commission_accumulates_across_fills():
    o = market_order("SPY", 100)
    o.transition_to(OrderStatus.SUBMITTED)
    o.apply_fill(50, 100.0, commission=1.0)
    o.apply_fill(50, 100.0, commission=1.5)
    assert o.commission == pytest.approx(2.5)


# ── OrderManager ──────────────────────────────────────────────────────────────

def test_manager_tracks_and_retrieves():
    m = OrderManager()
    o = m.track(market_order("SPY", 10))
    assert len(m) == 1
    assert m.get(o.client_order_id) is o
    assert o.client_order_id in m


def test_manager_broker_id_lookup():
    m = OrderManager()
    o = m.track(market_order("SPY", 10))
    m.link_broker_id(o.client_order_id, "abc-123")
    assert m.get_by_broker_id("abc-123") is o


def test_manager_link_unknown_id_raises():
    with pytest.raises(KeyError):
        OrderManager().link_broker_id("nope", "abc")


def test_manager_open_vs_closed():
    m = OrderManager()
    a = m.track(market_order("SPY", 10))
    b = m.track(market_order("QQQ", 10))
    b.cancel()
    assert m.open_orders() == [a]
    assert m.closed_orders() == [b]


def test_manager_filters_by_symbol():
    m = OrderManager()
    m.track(market_order("SPY", 10))
    m.track(market_order("QQQ", 10))
    assert len(m.open_orders("SPY")) == 1


def test_manager_net_filled_quantity_nets_buys_and_sells():
    m = OrderManager()
    buy = m.track(market_order("SPY", 100))
    buy.transition_to(OrderStatus.SUBMITTED)
    buy.apply_fill(100, 100.0)
    sell = m.track(market_order("SPY", -40))
    sell.transition_to(OrderStatus.SUBMITTED)
    sell.apply_fill(40, 105.0)
    assert m.net_filled_quantity("SPY") == 60


def test_manager_pending_exposure_counts_unfilled_only():
    m = OrderManager()
    o = m.track(market_order("SPY", 100))
    o.transition_to(OrderStatus.SUBMITTED)
    o.apply_fill(30, 100.0)
    assert m.pending_exposure("SPY") == 70


def test_manager_reconcile_clean_when_matched():
    m = OrderManager()
    o = m.track(market_order("SPY", 100))
    o.transition_to(OrderStatus.SUBMITTED)
    o.apply_fill(100, 100.0)
    assert m.reconcile({"SPY": 100.0}) == {}


def test_manager_reconcile_detects_drift():
    m = OrderManager()
    o = m.track(market_order("SPY", 100))
    o.transition_to(OrderStatus.SUBMITTED)
    o.apply_fill(100, 100.0)
    # Broker says 150 — 50 shares came from outside this session.
    drift = m.reconcile({"SPY": 150.0})
    assert drift == {"SPY": 50.0}


def test_manager_reconcile_detects_unknown_broker_position():
    m = OrderManager()
    assert m.reconcile({"AAPL": 25.0}) == {"AAPL": 25.0}


def test_manager_summary_and_frame():
    m = OrderManager()
    o = m.track(market_order("SPY", 100))
    o.transition_to(OrderStatus.SUBMITTED)
    o.apply_fill(100, 100.0, commission=1.0)
    s = m.summary()
    assert s["n_orders"] == 1 and s["n_filled"] == 1
    assert s["fill_rate"] == 1.0
    assert s["total_notional"] == pytest.approx(10_000.0)
    assert len(m.to_frame()) == 1


def test_manager_empty_frame_has_columns():
    assert list(OrderManager().to_frame().columns)


# ── MockBroker ────────────────────────────────────────────────────────────────

@pytest.fixture
def broker() -> MockBroker:
    b = MockBroker(initial_cash=100_000.0, prices={"SPY": 100.0, "QQQ": 200.0})
    b.connect()
    return b


def test_broker_connects(broker):
    assert broker.connected


def test_broker_market_order_fills_immediately(broker):
    o = broker.submit_order(market_order("SPY", 100))
    assert o.status == OrderStatus.FILLED
    assert o.avg_fill_price == 100.0


def test_broker_updates_cash_and_position(broker):
    broker.submit_order(market_order("SPY", 100))
    assert broker.cash == pytest.approx(90_000.0)
    assert broker.get_positions()["SPY"].quantity == 100


def test_broker_equity_unchanged_by_fill_without_slippage(broker):
    broker.submit_order(market_order("SPY", 100))
    assert broker.get_account().equity == pytest.approx(100_000.0)


def test_broker_slippage_moves_buy_price_up():
    b = MockBroker(prices={"SPY": 100.0}, slippage=0.01)
    b.connect()
    o = b.submit_order(market_order("SPY", 10))
    assert o.avg_fill_price == pytest.approx(101.0)


def test_broker_slippage_moves_sell_price_down():
    b = MockBroker(prices={"SPY": 100.0}, slippage=0.01)
    b.connect()
    o = b.submit_order(market_order("SPY", -10))
    assert o.avg_fill_price == pytest.approx(99.0)


def test_broker_rejects_unknown_symbol(broker):
    o = broker.submit_order(market_order("NOPE", 10))
    assert o.status == OrderStatus.REJECTED
    assert "no price" in o.reject_reason


def test_broker_rejects_market_order_when_closed():
    b = MockBroker(prices={"SPY": 100.0}, market_open=False)
    b.connect()
    o = b.submit_order(market_order("SPY", 10))
    assert o.status == OrderStatus.REJECTED
    assert "closed" in o.reject_reason


def test_broker_forced_rejection_hook(broker):
    broker.reject_next = "insufficient buying power"
    o = broker.submit_order(market_order("SPY", 10))
    assert o.status == OrderStatus.REJECTED
    # Hook is one-shot: the next order goes through.
    assert broker.submit_order(market_order("SPY", 10)).status == OrderStatus.FILLED


def test_broker_limit_order_rests_until_crossed(broker):
    o = broker.submit_order(limit_order("SPY", 50, limit_price=95.0))
    assert o.status == OrderStatus.SUBMITTED
    broker.set_price("SPY", 96.0)
    assert o.status == OrderStatus.SUBMITTED        # not yet through the limit
    broker.set_price("SPY", 94.0)
    assert o.status == OrderStatus.FILLED
    assert o.avg_fill_price == 95.0


def test_broker_sell_limit_fills_on_rise(broker):
    o = broker.submit_order(limit_order("SPY", -50, limit_price=110.0))
    broker.set_price("SPY", 111.0)
    assert o.status == OrderStatus.FILLED


def test_broker_cancel_order(broker):
    o = broker.submit_order(limit_order("SPY", 50, limit_price=1.0))
    broker.cancel_order(o)
    assert o.status == OrderStatus.CANCELED


def test_broker_cancel_terminal_order_raises(broker):
    o = broker.submit_order(market_order("SPY", 10))     # fills immediately
    with pytest.raises(BrokerError):
        broker.cancel_order(o)


def test_broker_cancel_all(broker):
    broker.submit_order(limit_order("SPY", 10, limit_price=1.0))
    broker.submit_order(limit_order("QQQ", 10, limit_price=1.0))
    assert len(broker.cancel_all()) == 2
    assert broker.orders.open_orders() == []


def test_broker_averages_cost_basis_on_add(broker):
    broker.submit_order(market_order("SPY", 100))
    broker.set_price("SPY", 120.0)
    broker.submit_order(market_order("SPY", 100))
    pos = broker.get_positions()["SPY"]
    assert pos.quantity == 200
    assert pos.avg_entry_price == pytest.approx(110.0)


def test_broker_close_position(broker):
    broker.submit_order(market_order("SPY", 100))
    closing = broker.close_position("SPY")
    assert closing is not None and closing.side == OrderSide.SELL
    assert broker.get_positions() == {}


def test_broker_close_position_when_flat_returns_none(broker):
    assert broker.close_position("SPY") is None


def test_broker_close_all_positions(broker):
    broker.submit_order(market_order("SPY", 10))
    broker.submit_order(market_order("QQQ", 10))
    assert len(broker.close_all_positions()) == 2
    assert broker.get_positions() == {}


def test_broker_short_position_is_negative(broker):
    broker.submit_order(market_order("SPY", -50))
    assert broker.get_positions()["SPY"].quantity == -50


def test_broker_reconcile_is_clean_after_fills(broker):
    broker.submit_order(market_order("SPY", 100))
    broker.submit_order(market_order("QQQ", -25))
    assert broker.reconcile() == {}


def test_broker_unrealized_pl(broker):
    broker.submit_order(market_order("SPY", 100))
    broker.set_price("SPY", 110.0)
    pos = broker.get_positions()["SPY"]
    assert pos.unrealized_pl == pytest.approx(1_000.0)
    assert pos.unrealized_pl_pct == pytest.approx(0.10)


def test_broker_sync_all_is_noop(broker):
    broker.submit_order(limit_order("SPY", 10, limit_price=1.0))
    assert len(broker.sync_all()) == 1


# ── Value objects ─────────────────────────────────────────────────────────────

def test_account_is_tradable():
    assert Account(1.0, 1.0, 1.0).is_tradable
    assert not Account(1.0, 1.0, 1.0, trading_blocked=True).is_tradable


def test_position_market_value_short_is_negative():
    p = BrokerPosition("SPY", -10, 100.0, current_price=105.0)
    assert p.market_value == pytest.approx(-1050.0)
    assert p.unrealized_pl == pytest.approx(-50.0)


def test_position_pl_pct_zero_cost_is_safe():
    assert BrokerPosition("SPY", 0, 0.0).unrealized_pl_pct == 0.0


# ── AlpacaBroker safety gates ─────────────────────────────────────────────────

def test_alpaca_paper_mode_constructs():
    b = AlpacaBroker(api_key="k", secret_key="s", mode="paper")
    assert b.is_paper


def test_alpaca_invalid_mode_raises():
    with pytest.raises(ValueError):
        AlpacaBroker(api_key="k", secret_key="s", mode="production")


def test_alpaca_live_blocked_without_any_flags():
    with pytest.raises(LiveTradingBlocked):
        AlpacaBroker(api_key="k", secret_key="s", mode="live")


def test_alpaca_live_blocked_with_only_kwarg():
    with mock.patch.dict(os.environ, {}, clear=True):
        with pytest.raises(LiveTradingBlocked):
            AlpacaBroker(api_key="k", secret_key="s", mode="live", allow_live=True)


def test_alpaca_live_blocked_with_only_env():
    with mock.patch.dict(os.environ, {"TRADING_ALLOW_LIVE": "1"}):
        with pytest.raises(LiveTradingBlocked):
            AlpacaBroker(api_key="k", secret_key="s", mode="live", allow_live=False)


def test_alpaca_live_allowed_with_both_switches():
    with mock.patch.dict(os.environ, {"TRADING_ALLOW_LIVE": "1"}):
        b = AlpacaBroker(api_key="k", secret_key="s", mode="live", allow_live=True)
        assert not b.is_paper


def test_alpaca_connect_without_credentials_raises():
    with mock.patch.dict(os.environ, {}, clear=True):
        with pytest.raises(BrokerError, match="credentials"):
            AlpacaBroker(api_key="", secret_key="", mode="paper").connect()


def test_alpaca_operations_require_connect():
    b = AlpacaBroker(api_key="k", secret_key="s", mode="paper")
    with pytest.raises(BrokerError, match="Not connected"):
        b.get_account()


def test_alpaca_reads_credentials_from_env():
    with mock.patch.dict(os.environ, {"ALPACA_API_KEY": "envkey",
                                      "ALPACA_SECRET_KEY": "envsecret"}):
        b = AlpacaBroker(mode="paper")
        assert b.api_key == "envkey"


def test_alpaca_from_config_defaults_to_paper():
    from config.settings import Settings
    b = AlpacaBroker.from_config(Settings.from_yaml().execution)
    assert b.is_paper


def test_alpaca_status_map_covers_alpaca_states():
    """Every Alpaca order state we expect must map to a local OrderStatus."""
    expected = {"new", "accepted", "partially_filled", "filled",
                "canceled", "expired", "rejected", "pending_new"}
    assert expected.issubset(set(AlpacaBroker._STATUS_MAP))


# ── Risk manager integration ──────────────────────────────────────────────────

def test_risk_manager_gates_broker_orders(broker):
    """Orders sized past the risk cap are shrunk before reaching the broker."""
    from src.trading.risk.manager import RiskManager

    rm = RiskManager(max_position_pct=0.10)
    rm.update_equity(100_000.0)

    decision = rm.evaluate(
        "SPY", proposed_qty=500, price=100.0, equity=100_000.0,
        positions={}, prices={"SPY": 100.0},
    )
    assert decision.approved and decision.approved_quantity == 100

    o = broker.submit_order(market_order("SPY", decision.approved_quantity))
    assert o.status == OrderStatus.FILLED
    assert broker.get_positions()["SPY"].quantity == 100


def test_risk_halt_prevents_broker_submission(broker):
    """A halted risk manager stops the order before it is ever submitted."""
    from src.trading.risk.manager import RiskManager

    rm = RiskManager(max_drawdown_pct=0.15)
    rm.update_equity(100_000.0)
    rm.update_equity(80_000.0)

    decision = rm.evaluate("SPY", 100, 100.0, 80_000.0, {}, {"SPY": 100.0})
    assert not decision.approved
    assert len(broker.orders) == 0

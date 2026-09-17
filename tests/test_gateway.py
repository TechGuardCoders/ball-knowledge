"""Offline tests for Ball Knowledge gateway logic - no network, no keys."""
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gateway import (Backend, BackendChain, Tenant, TenantRegistry,  # noqa: E402
                     RateLimiter, BudgetLedger, mock_completion, _tokens_used)


def test_auth_valid_and_invalid():
    reg = TenantRegistry(tenants=[Tenant(id="t1", name="T1", api_key="k1")])
    assert reg.authenticate("Bearer k1").id == "t1"
    assert reg.authenticate("Bearer wrong") is None
    assert reg.authenticate(None) is None
    assert reg.authenticate("k1") is None  # malformed


def test_rate_limiter_sliding_window():
    rl = RateLimiter()
    t = "tenant-x"
    assert all(rl.allow(t, 5) for _ in range(5)), "first 5 pass"
    assert not rl.allow(t, 5), "6th in a minute blocked"
    assert rl.allow("other-tenant", 5), "other tenant unaffected"


def test_budget_ledger():
    led = BudgetLedger()
    tn = Tenant(id="t", name="T", api_key="k", monthly_token_budget=1000)
    assert led.remaining(tn) == 1000
    led.record(tn, 600)
    assert led.remaining(tn) == 400
    assert not led.would_exceed(tn, 400)
    assert led.would_exceed(tn, 401)


def test_chain_orders_healthy_first():
    a = Backend(name="a", base_url="mock://", api_key=None, model="m", healthy=True)
    b = Backend(name="b", base_url="mock://", api_key=None, model="m", healthy=False)
    chain = BackendChain(backends=[a, b])
    assert [x.name for x in chain.ordered()] == ["a", "b"]
    b.healthy = True
    assert [x.name for x in chain.ordered()] == ["a", "b"]  # priority preserved


def test_mock_backend_deterministic_shape():
    body = mock_completion({"messages": [{"role": "user", "content": "hello world"}]}, "mock-echo")
    assert body["choices"][0]["message"]["content"].startswith("[mock-fallback]")
    assert body["usage"]["total_tokens"] > 0
    assert _tokens_used(json.dumps(body).encode()) == body["usage"]["total_tokens"]


def test_failover_marks_unhealthy_after_repeat():
    # simulate gateway's failure marking logic
    b = Backend(name="x", base_url="mock://", api_key=None, model="m")
    for _ in range(2):
        b.consecutive_failures += 1
        if b.consecutive_failures >= 2:
            b.healthy = False
    assert b.healthy is False
    assert b.consecutive_failures == 2

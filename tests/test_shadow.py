"""Shadow Route tests — candidate never touches the client; report is honest."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))
import gateway
from gateway import ShadowRouter, Backend, normalize_answer


@pytest.fixture
def client():
    from fastapi.testclient import TestClient as TC
    return TC(gateway.app)


@pytest.fixture(autouse=True)
def reset_shadow():
    gateway.shadow = None
    yield
    gateway.shadow = None


# ---------------------------------------------------------------- unit

def test_normalize_answer():
    assert normalize_answer("  Hello   WORLD ") == "hello world"


def test_should_shadow_rate_one():
    r = ShadowRouter(Backend(name="c", base_url="mock://x", api_key=None, model="m"), 1.0)
    assert r.should_shadow() is True


def test_should_shadow_rate_never_at_tiny():
    r = ShadowRouter(Backend(name="c", base_url="mock://x", api_key=None, model="m"), 1e-9)
    assert r.should_shadow() is False


def test_invalid_sample_rate_raises():
    with pytest.raises(ValueError):
        ShadowRouter(Backend(name="c", base_url="x", api_key=None, model="m"), 0.0)
    with pytest.raises(ValueError):
        ShadowRouter(Backend(name="c", base_url="x", api_key=None, model="m"), 1.5)


def test_record_agreement_and_speed():
    r = ShadowRouter(Backend(name="c", base_url="x", api_key=None, model="m"), 0.5)
    r.record("p", 100.0, "The Answer", 200, 50.0, "the answer")
    t = r.trials[0]
    assert t["agree"] is True
    assert t["cand_faster_ms"] == 50.0


def test_report_agreement():
    r = ShadowRouter(Backend(name="c", base_url="x", api_key=None, model="m"), 0.5)
    r.record("p", 100, "A", 200, 50, "A")
    r.record("p", 100, "A", 200, 150, "B")  # candidate slower here
    rep = r.report()
    assert rep["agreement"] == 0.5
    assert rep["cand_faster_pct"] == 50.0
    assert rep["trials"] == 2


def test_report_empty_is_none_not_zero():
    r = ShadowRouter(Backend(name="c", base_url="x", api_key=None, model="m"), 0.5)
    rep = r.report()
    assert rep["agreement"] is None
    assert rep["cand_p50_ms"] is None


# ---------------------------------------------------------------- integration

def test_shadow_never_changes_client_response(client):
    """With sample_rate=1.0 and a candidate that 500s, client still gets 200."""
    gateway.registry.register(gateway.Tenant(id="t", name="T", api_key="bk-test",
                                             rate_limit_per_min=100))
    gateway.shadow = ShadowRouter(
        Backend(name="broken-cand", base_url="http://127.0.0.1:1", api_key=None,
                model="cand"), sample_rate=1.0)
    r = client.post("/v1/chat/completions",
                    headers={"Authorization": "Bearer bk-test"},
                    json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200  # candidate failure is invisible to the client


def test_shadow_report_endpoint(client):
    gateway.shadow = ShadowRouter(
        Backend(name="c", base_url="mock://local", api_key=None, model="m"), 1.0)
    gateway.registry.register(gateway.Tenant(id="t", name="T", api_key="bk-test",
                                             rate_limit_per_min=100))
    client.post("/v1/chat/completions", headers={"Authorization": "Bearer bk-test"},
                json={"messages": [{"role": "user", "content": "hi"}]})
    import time; time.sleep(0.3)  # daemon thread
    rep = client.get("/admin/shadow/report").json()
    assert rep["active"] is True
    assert rep["trials"] == 1
    assert rep["agreement"] is not None


def test_shadow_inactive_by_default(client):
    assert client.get("/admin/shadow/report").json() == {"active": False}


def test_stop_shadow(client):
    gateway.shadow = ShadowRouter(
        Backend(name="c", base_url="x", api_key=None, model="m"), 0.5)
    assert client.post("/admin/shadow/stop").json() == {"shadowing": False}
    assert client.get("/admin/shadow/report").json() == {"active": False}

"""
Ball Knowledge - multi-model AI gateway.

A traffic policy layer in front of serving backends. Production is not one
endpoint: it's routing, retries, fallbacks, rate limits, and budgets.

Design:
- OpenAI-compatible POST /v1/chat/completions (drop-in for VS Code/Cline)
- Backends are named and chained: primary GLM (real cluster), then fallbacks
- Per-tenant: API key, rate limit (req/min), monthly token budget
- State: in-memory behind clean interfaces (TenantRegistry, BudgetLedger,
  RateLimiter) - Redis swaps in later without redesign
- Chaos-testable: kill a backend mid-stream, gateway fails over

Port: 3100 (3001 reserved for Cost Peep's Mac Mini home; 3000 never touched).
"""
from __future__ import annotations

import json
import os
import time
import uuid
import urllib.request
import urllib.error
from collections import defaultdict, deque
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Callable, Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

# ---------------------------------------------------------------- backends


@dataclass
class Backend:
    name: str
    base_url: str
    api_key: str | None
    model: str
    healthy: bool = True
    last_error: str | None = None
    consecutive_failures: int = 0

    def reachable(self) -> bool:
        return self.healthy


@dataclass
class BackendChain:
    """Ordered fallback chain: try in sequence until one succeeds."""
    backends: list[Backend]

    def ordered(self) -> list[Backend]:
        # healthy first, preserving priority order
        return [b for b in self.backends if b.healthy] + [b for b in self.backends if not b.healthy]


def default_chain() -> BackendChain:
    """Primary: the real GLM cluster. Fallback: in-process mock (never fails)."""
    return BackendChain(backends=[
        Backend(
            name="glm-spark",
            base_url=os.environ.get("VLLM_BASE_URL", "http://192.168.0.182:8000"),
            api_key=os.environ.get("VLLM_API_KEY"),
            model=os.environ.get("VLLM_MODEL", "glm-5.3-flash"),
        ),
        Backend(
            name="mock-fallback",
            base_url="mock://local",
            api_key=None,
            model="mock-echo",
        ),
    ])


# ---------------------------------------------------------------- tenants


@dataclass
class Tenant:
    id: str
    name: str
    api_key: str
    rate_limit_per_min: int = 30
    monthly_token_budget: int = 2_000_000


class TenantRegistry:
    """API-key -> tenant. In-memory; seeded from config or defaults."""

    def __init__(self, tenants: list[Tenant] | None = None):
        self._by_key: dict[str, Tenant] = {}
        for t in tenants or []:
            self._by_key[t.api_key] = t

    def authenticate(self, authorization: str | None) -> Optional[Tenant]:
        if not authorization or not authorization.lower().startswith("bearer "):
            return None
        key = authorization.split(" ", 1)[1].strip()
        return self._by_key.get(key)

    def register(self, tenant: Tenant) -> None:
        self._by_key[tenant.api_key] = tenant


class RateLimiter:
    """Sliding-window per-minute limiter, per tenant."""

    def __init__(self):
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, tenant_id: str, limit_per_min: int) -> bool:
        now = time.time()
        window = self._hits[tenant_id]
        while window and now - window[0] > 60.0:
            window.popleft()
        if len(window) >= limit_per_min:
            return False
        window.append(now)
        return True


class BudgetLedger:
    """Per-tenant token accounting with a monthly cap."""

    def __init__(self):
        self._used: dict[str, int] = defaultdict(int)

    def remaining(self, tenant: Tenant) -> int:
        return max(0, tenant.monthly_token_budget - self._used[tenant.id])

    def record(self, tenant: Tenant, tokens: int) -> None:
        self._used[tenant.id] += max(0, tokens)

    def would_exceed(self, tenant: Tenant, estimated_tokens: int) -> bool:
        return self.remaining(tenant) < estimated_tokens


# ---------------------------------------------------------------- upstream


def call_backend(backend: Backend, payload: dict, timeout: float = 240.0) -> tuple[int, bytes, bool]:
    """POST to an OpenAI-compatible backend. Returns (status, body, streamable)."""
    if backend.base_url.startswith("mock://"):
        body = mock_completion(payload, backend.model)
        return 200, json.dumps(body).encode(), False

    req = urllib.request.Request(
        backend.base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {backend.api_key}"} if backend.api_key else {}),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), False
    except urllib.error.HTTPError as e:
        return e.code, e.read(), False
    except Exception as e:  # noqa: BLE001
        return 502, json.dumps({"error": f"backend unreachable: {e}"}).encode(), False


def mock_completion(payload: dict, model: str) -> dict:
    """Deterministic echo backend for fallback demos and tests."""
    prompt = payload.get("messages", [{}])[-1].get("content", "")
    return {
        "id": "mock-" + uuid.uuid4().hex[:8],
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant",
                     "content": f"[mock-fallback] {prompt[:80]}"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": 10,
                  "total_tokens": len(prompt) // 4 + 10},
    }


# ---------------------------------------------------------------- gateway

app = FastAPI(title="Ball Knowledge", version="1.0.0")

chain = default_chain()
registry = TenantRegistry(tenants=[
    Tenant(id="tenant-a", name="Tenant A", api_key=os.environ.get("BK_KEY_TENANT_A", "bk-tenant-a-dev"),
           rate_limit_per_min=60, monthly_token_budget=5_000_000),
    Tenant(id="tenant-b", name="Tenant B", api_key=os.environ.get("BK_KEY_TENANT_B", "bk-tenant-b-dev"),
           rate_limit_per_min=30, monthly_token_budget=1_000_000),
])
limiter = RateLimiter()
ledger = BudgetLedger()
audit_log: list[dict] = []


def audit(event: dict) -> None:
    event["ts"] = time.time()
    audit_log.append(event)


@app.get("/health")
def health():
    return {
        "ok": True,
        "backends": [{"name": b.name, "healthy": b.healthy, "model": b.model} for b in chain.backends],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    # 1. authenticate
    tenant = registry.authenticate(request.headers.get("Authorization"))
    if tenant is None:
        audit({"event": "auth_rejected", "ip": request.client.host if request.client else None})
        return JSONResponse({"error": "invalid or missing API key"}, status_code=HTTPStatus.UNAUTHORIZED)

    # 2. rate limit
    if not limiter.allow(tenant.id, tenant.rate_limit_per_min):
        audit({"event": "rate_limited", "tenant": tenant.id})
        return JSONResponse({"error": "rate limit exceeded"}, status_code=HTTPStatus.TOO_MANY_REQUESTS)

    # 3. budget guard (estimate: 1 request ~= max_tokens in payload or 200)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=HTTPStatus.BAD_REQUEST)

    estimate = body.get("max_tokens", 200) + 200
    if ledger.would_exceed(tenant, estimate):
        audit({"event": "budget_exceeded", "tenant": tenant.id})
        return JSONResponse({"error": "monthly token budget exhausted"}, status_code=HTTPStatus.PAYMENT_REQUIRED)

    # 4. route through the fallback chain
    last_status, last_body = 502, b'{"error":"no backends"}'
    for backend in chain.ordered():
        status, resp, _ = call_backend(backend, body)
        if status == 200:
            ledger.record(tenant, _tokens_used(resp))
            audit({"event": "served", "tenant": tenant.id, "backend": backend.name,
                   "tokens": _tokens_used(resp)})
            return Response(content=resp, media_type="application/json", status_code=200)
        # mark backend unhealthy after repeated failures
        backend.consecutive_failures += 1
        backend.last_error = resp.decode(errors="replace")[:200]
        if backend.consecutive_failures >= 2:
            backend.healthy = False
        audit({"event": "backend_failed", "backend": backend.name, "status": status})
        last_status, last_body = status, resp

    return Response(content=last_body, media_type="application/json", status_code=last_status)


def _tokens_used(resp_bytes: bytes) -> int:
    try:
        usage = json.loads(resp_bytes).get("usage", {})
        return usage.get("total_tokens", 0)
    except Exception:
        return 0


@app.get("/audit")
def get_audit():
    """Audit trail: every auth/reject/serve/failover event, newest last."""
    return {"events": audit_log[-500:]}


@app.post("/admin/chaos/kill-backend/{name}")
def kill_backend(name: str):
    """Chaos drill: mark a backend unhealthy to prove failover works."""
    for b in chain.backends:
        if b.name == name:
            b.healthy = False
            audit({"event": "chaos_kill", "backend": name})
            return {"killed": name}
    return JSONResponse({"error": "unknown backend"}, status_code=404)


@app.post("/admin/chaos/revive-backend/{name}")
def revive_backend(name: str):
    for b in chain.backends:
        if b.name == name:
            b.healthy = True
            b.consecutive_failures = 0
            audit({"event": "chaos_revive", "backend": name})
            return {"revived": name}
    return JSONResponse({"error": "unknown backend"}, status_code=404)

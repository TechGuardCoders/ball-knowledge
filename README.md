# Ball Knowledge

**Multi-model AI gateway — production is a traffic policy, not one endpoint.**

Routes, retries, fallbacks, rate limits, and per-tenant token budgets across serving backends. Sits in front of the [spark-glm-cluster](https://github.com/TechGuardCoders/spark-glm-cluster) (2× DGX Spark, vLLM) as an OpenAI-compatible proxy — drop-in for VS Code, Cline, or any OpenAI SDK client.

Part of the [TechGuardCoders AI infrastructure portfolio](https://github.com/orgs/TechGuardCoders/repositories): Cost Peep (cost dashboard) → Batcher (load harness) → Spinal (latency tracing) → **Ball Knowledge (traffic policy)** → Bastion (tenant security, next) → …

## Why

Production LLM systems don't have one endpoint — they have a *policy*: which tenant gets which model, what happens when the primary dies mid-request, who's over budget, who's hammering the API. This gateway proves those mechanics with live chaos drills, not diagrams.

## Architecture

```
client (OpenAI SDK)
   │  Authorization: Bearer bk-<tenant>-key
   ▼
Ball Knowledge :3100  (FastAPI)
   ├─ authenticate        → TenantRegistry (API key → tenant)
   ├─ rate limit          → RateLimiter (sliding window, req/min per tenant)
   ├─ budget guard        → BudgetLedger (monthly token cap per tenant)
   ├─ route + failover    → BackendChain (ordered fallback, health tracking)
   └─ audit log           → every auth/reject/serve/failover event

   ├─ primary:  glm-spark      real vLLM on 192.168.0.182:8000
   └─ fallback: mock-fallback  in-process echo (never fails)
```

State is in-memory behind clean interfaces (`TenantRegistry`, `RateLimiter`, `BudgetLedger`) — Redis swaps in later without redesign.

## Live chaos drill (measured, not staged)

Run `scripts/chaos_drill.py` against a live gateway:

| Check | Result |
|---|---|
| Bad API key → 401 | ✅ |
| No API key → 401 | ✅ |
| Valid key → 200 **via glm-spark** (real GLM-5.3-Flash serving) | ✅ |
| Chaos kill primary → accepted | ✅ |
| **Failover: killed primary → 200 via mock-fallback, zero client error** | ✅ |
| Revive → 200 back via glm-spark | ✅ |
| Audit trail records the full story | ✅ |

Full failover cycle in one session:

```
primary alive  → 200 glm-5.3-flash
chaos kill     → 200 mock-fallback   ← client saw nothing but a 200
revive         → 200 glm-5.3-flash
```

## Tenants

| Tenant | Key env var | Rate limit | Monthly budget |
|---|---|---|---|
| Tenant A | `BK_KEY_TENANT_A` | 60 req/min | 5M tokens |
| Tenant B | `BK_KEY_TENANT_B` | 30 req/min | 1M tokens |

Keys are env-configured, never committed. Over-budget requests get **402**, over-rate get **429**, bad keys get **401** — all before touching a backend.

## Run it

```bash
uv venv .venv && uv pip install -r requirements.txt
VLLM_API_KEY=<key> BK_KEY_TENANT_A=bk-tenant-a-dev \
  .venv/Scripts/python -m uvicorn gateway:app --port 3100

# drill it
.venv/Scripts/python scripts/chaos_drill.py
```

Port 3100 by convention (3001 reserved for Cost Peep; nothing ever binds 3000).

## Tests

6 offline unit tests (`tests/test_gateway.py`) — auth, sliding-window rate limiting, budget ledger, chain ordering, mock shape, unhealthy-marking. No network, no keys, CI-safe.

## What's next

**Bastion** builds directly on this: per-tenant isolation hardened into sandboxed exec paths and full audit queries — and its per-tenant attribution is what fills Cost Peep's BY TENANT tab with real data.

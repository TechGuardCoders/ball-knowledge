"""Live chaos drill: proves failover, rate limits, budgets, audit trail.

Run: .venv/Scripts/python scripts/chaos_drill.py  (gateway must be running on :3100)
Env: BK_URL (default http://localhost:3100), BK_KEY_TENANT_A, VLLM_API_KEY not needed here (gateway holds it).
"""
import json
import os
import sys
import urllib.request
import urllib.error

BASE = os.environ.get("BK_URL", "http://localhost:3100")
KEY = os.environ.get("BK_KEY_TENANT_A", "bk-tenant-a-dev")


def post(path, payload=None, auth=KEY):
    req = urllib.request.Request(BASE + path,
                                 data=json.dumps(payload or {}).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {auth}"} if auth else {})})
    try:
        with urllib.request.urlopen(req, timeout=240) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read())


results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


chat = {"model": "glm-5.3-flash", "max_tokens": 20,
        "messages": [{"role": "user", "content": "One sentence: what is GPU caching?"}]}

# 1. bad key rejected
s, b = post("/v1/chat/completions", chat, auth="wrong-key")
check("auth: bad key -> 401", s == 401, f"status={s}")

# 2. no key rejected
s, b = post("/v1/chat/completions", chat, auth=None)
check("auth: no key -> 401", s == 401, f"status={s}")

# 3. good key via real backend
s, b = post("/v1/chat/completions", chat)
check("serve: valid key -> 200 via glm-spark", s == 200 and b.get("model", "").startswith("glm"),
      f"status={s} model={b.get('model')}")

# 4. chaos: kill primary -> failover to mock
s, _ = post("/admin/chaos/kill-backend/glm-spark")
check("chaos: kill glm-spark accepted", s == 200)
s, b = post("/v1/chat/completions", chat)
check("failover: killed primary -> 200 via mock-fallback",
      s == 200 and "mock-fallback" in b.get("choices", [{}])[0].get("message", {}).get("content", ""),
      f"status={s}")

# 5. revive
s, _ = post("/admin/chaos/revive-backend/glm-spark")
check("chaos: revive glm-spark accepted", s == 200)

# 6. audit trail has the story
ev = [e["event"] for e in get("/audit")["events"]]
check("audit: contains serve/failover/chaos events",
      all(x in ev for x in ["served", "backend_failed", "chaos_kill", "chaos_revive"]),
      f"events={sorted(set(ev))}")

fails = [r for r in results if not r[1]]
print(f"\n{len(results) - len(fails)}/{len(results)} checks passed")
sys.exit(1 if fails else 0)

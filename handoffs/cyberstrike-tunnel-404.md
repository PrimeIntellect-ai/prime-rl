# CyberStrike: broken Prime tunnel investigation and repair handoff

Run `pzh1apauhzm18onn8pxg96uh`; evidence date **2026-11-02 UTC**. Reuse the method below; do not assume every HTTP 404 has this cause.

## Current state and authority

- **PAUSED**, platform `completed_at=2026-11-02T13:12:58.844Z`. No autoresume. The paused run has not loaded this patch.
- Checkpoint API reports remote **166 READY**, 1,447,468,806,467 bytes, uploaded **11:40:02.514Z**; **160 READY**, uploaded **11:40:03.630Z**. These are platform availability reports, not independent verification of same-generation joint manifests, final volume sync, resume correctness, or a full HF export.
- Earlier authorization covered exactly one 27-tunnel stop batch and a checkpoint-aware pause. Do not infer permission for further deletions, VM/tunnel/resource changes, guest probes, model calls, resume, or a new run. Obtain explicit approval for those actions and for external escalation delivery. This deployment involved source commits/pushes, offline checks, and a read-only run-status check; no training-resource mutation.
- “Deploy” currently means **publish source for a future authorized launch**, not resume/restart this run or allocate GPUs. The source and pinned launch recipe are published. No recovery on the provider route is claimed.

### Published source, not a live-run reload

- Original root: `9507338ffb642b6304dc2e7e4bdcc97b54e21de4`; original verifier gitlink: `0512206f6d7ef05c7266caab781cd7a59209ad98`.
- Verifier repair committed and pushed: **`d376c25298b4e5b321608de81265e5c4c356966c`**, branch `fix/cyberstrike-tunnel-route-health`.
- Root **code commit**: **`51957e8a858c6fa33f2e776ebbfd82e923e93781`**, including the updated `deps/verifiers` gitlink. Root branch `fix/cyberstrike-tunnel-deploy`, based on `358ea2b`.
- Root source **`51957e8a858c6fa33f2e776ebbfd82e923e93781` is published**. The recipe pins that immutable `source_ref`. The recipe and this handoff are published on `fix/cyberstrike-tunnel-deploy` and fast-forwarded to `feat/cyberstrike-next-run`. Their containing commit is discoverable with `git log -1 --format=%H -- handoffs/cyberstrike-tunnel-404.md`; it is intentionally distinct from the code commit (which predates these documentation files). Source publication is not a completed launch.
- Validation found valid recipe TOML and an offline CLI build. Local `RLConfig` validation passed using a temporary `slurm={}` copy to satisfy the local multi-node guard; that workaround was **not persisted to the hosted recipe**. No hosted create POST was made.
- Durable repository handoff: `handoffs/cyberstrike-tunnel-404.md`; evidence bundle: `handoffs/cyberstrike-tunnel-404/{deployment.json,test_recovery.py,assigned-endpoint-probe.json,replacement-endpoint-probe.json,failed-trace-tunnel-joins.json}`. Obtain it from the final handoff/recipe publication; do not assume these later documentation files exist in code commit `51957e8`. The combined test has **2 passes**. Scratch `/tmp` files below are provenance, not the only reusable copy.

## Evidence-ranked diagnosis

1. **Establish assignment, not association.** Join an exact fatal trace name to the exact sandbox **name**, then read that sandbox's **applied network policy model hostname**, then extract the tunnel ID. Verify policy generation equals applied generation and `applied=true`. Five fatal traces had generation/applied-generation 1 and one allowed framework/model hostname, all identifying **`t-0-eed351ea43f3af7a`**. Example: trace `e727091a151f4b22ae39dfc536635e4a` → sandbox `x6mwi3ipgs4dr539l7nbyrue` → applied model host → that tunnel. All five exact joins are in `/tmp/cyberstrike-deep-safe-page2-result.json`.
2. **The assigned old tunnel's application path failed independently of inference.** It was created **Oct 30 02:17:01.075Z**, expires Nov 27, and was still reported `connected`. A side-effect-free public GET to `/__interception/healthz` took **120.233 s**, then returned **404**, `text/html; charset=utf-8`, **5,053 bytes**, with the “TunnelNotFound”/“tunnel not found” marker. Body SHA256: `e4a80f93de7f9c8135bd3089b87ad367ee0c2438c7300de5d55c59e590fd17e3`. This is direct application-delivery failure evidence, **not a capture of the original failing model POST**.
3. **Our selection/watchdog gap is source-proven.** Startup sets the pool member's healthy bit. The old ongoing watcher checks registration/frpc state, not continued application identity. A connected-but-unusable member can therefore stay eligible and be selected again by an outer rollout retry. The authorized **27-tunnel reset omitted this exact old assigned endpoint**. Continued errors after that reset did not demonstrate that a fresh replacement failed or that the reset covered all assigned endpoints.
4. **The specific provider bug remains unknown.** Missing route, wrong ingress/Host/instance, and FRP work-connection/control failures are candidates, not findings. Generic FRP forwarding errors can become 404 even when a route exists. The deployed gateway/frps version, settings, and request/error logs are unavailable. The observed 120-second delay is not a proven upstream default. Two such attempts plus the nano 15-second retry fit roughly 261-second ACP lifetimes, but the original attempt timings were not captured.

Do **not** select a failing endpoint from a “successful access” roster. A newer tunnel, `t-0-124aea4605711154`, had earlier model POST 200 records but later returned the identical HTML 404 on application health. Previous success, `CONNECTED`, frpc running, initial startup health, gateway plain `/healthz` 200, shared Cloudflare DNS/IPs, and a `cf-ray` edge-PoP suffix do not prove current end-to-end application forwarding or backend affinity.

**SDK observability, not a routing cure:** guest traceback lines fingerprint an OpenAI 3.41/3.42-family SDK, not the root lock's 2.54. Offline native SDK tests show nonempty HTML 404 can stringify as only `Error code: 404`; response/error attributes still retain the body. Guest `uv tool install` is unfrozen: root/nano locks do not freeze that isolated environment. Record actual guest versions and sanitized response metadata when authorized. No SDK upgrade/downgrade, FRP upgrade, retry-budget increase, or dependency change is claimed to fix this failure.

## What the repair does—and does not do

- For successfully started **tunneled** servers, check both local and public **application-owned** `/__interception/healthz`. Require HTTP 200 and JSON `server_id` and `tunnel_id` matching this exact server/tunnel. Gateway `/healthz` is intercepted upstream and can return plain `OK` without reaching the application.
- **New constants:** 20 seconds between check cycles; quarantine after **2 consecutive failed cycles**. Both probes must succeed to reset the counter. Existing probe timeout remains **5 s**, connect **2 s**. Probe durations add to intervals; this is not a guaranteed 40-second wall-clock deadline.
- Quarantine sets `healthy=false` and rejects new registrations. The elastic pool assigns new work to a newly started/validated member. It does **not** kill, migrate, release, or rewrite URLs for active sessions. All **32 multiplexed sessions** can drain on the old member; retire/close it only at `load==0`, including cleanup when the final acquisition exits. Existing terminal `on_gone` behavior is unchanged and can still abort sessions.
- Ownership cleanup avoids retaining retired servers in per-member lifetime exit callbacks. Candidates are owned before startup validation; pruning removes each only after successful stop, so cancellation/cleanup failure retains an owner for retry/shutdown. Stop-all still attempts other members after one cleanup error.
- Limits: host probes can miss guest-only/time/vantage-specific failures. Draining cannot repair URLs already injected into active sandboxes. Static/standalone servers are quarantined, not automatically replaced. An idle quarantined member is reclaimed at a later acquisition/shutdown. Broad outages can cause repeated candidate starts; existing serialization/limiter remains, but there is **no new global circuit breaker, outage backoff, or capacity cap**. This is containment of our pool/watch gap, not a provider routing repair.

## Reproduce and verify locally without touching the paused run

Use immutable source objects and the project's native environment. Do not import project packages into the orchestration REPL, install dependencies to make tests pass, or substitute a dirty working tree for deployed source.

The final commit diff is saved in `handoffs/cyberstrike-tunnel-404/repair.patch` (also `/Users/azh/Documents/cyberstrike-tunnel-repair.patch`), SHA256 **`b33b46fa8e0bbde837ccc76a9524b836fe1cd8c926179606ad2f8512d36f9160`**. To replay it, choose a **new, unused** worktree path; do not apply it on top of the already repaired commit:

```sh
ROOT=/Users/azh/projects/nvidia/prime-rl
VF=/tmp/cyberstrike-tunnel-repair-replay
PATCH="$ROOT/handoffs/cyberstrike-tunnel-404/repair.patch"
BASE=0512206f6d7ef05c7266caab781cd7a59209ad98

git -C "$ROOT/deps/verifiers" worktree add --detach "$VF" "$BASE"
git -C "$VF" rev-parse HEAD
shasum -a 256 "$PATCH"
git -C "$VF" apply --check "$PATCH"
git -C "$VF" apply "$PATCH"
git -C "$VF" diff --check

# Explicitly select the exact checkout and the two required example packages.
export VF
export PYTHONPATH="$VF:$VF/environments/kuhn_poker:$VF/environments/reverse_text"
uv run --offline --no-sync --project "$ROOT" python -c 'import importlib, os; from pathlib import Path; m=importlib.import_module("verifiers.v1.interception.server"); assert Path(m.__file__).resolve()==Path(os.environ["VF"]+"/verifiers/v1/interception/server.py").resolve(); print(m.__file__)'
uv run --offline --no-sync --project "$ROOT" python -m pytest "$VF/tests/unit/interception" --import-mode=importlib -o addopts='' -q
uv run --offline --no-sync --project "$ROOT" python -m pytest "$ROOT/handoffs/cyberstrike-tunnel-404/test_recovery.py" --import-mode=importlib -o addopts='' -q
cd "$VF"
uv run --offline --no-sync --project "$ROOT" python -m pytest "$VF/tests/unit" --import-mode=importlib -o addopts='' -q -k 'not test_config_tree_imports_without_the_runtime'
uv run --offline --no-sync --project "$ROOT" ruff check "$VF/verifiers/v1/interception/server.py" "$VF/verifiers/v1/interception/pool.py" "$VF/tests/unit/interception/test_server.py" "$VF/tests/unit/interception/test_pool.py"
uv run --offline --no-sync --project "$ROOT" ruff format --check "$VF/verifiers/v1/interception/server.py" "$VF/verifiers/v1/interception/pool.py" "$VF/tests/unit/interception/test_server.py" "$VF/tests/unit/interception/test_pool.py"
```

Recorded results: **35 interception tests + 2 combined tests = 37 focused passes**; combined real server/pool lifecycle tests use fake tunnels and controlled health, parameterized for **1 and 32 active sessions**. They show quarantine, a different URL on next acquisition, no early session release/tunnel closure, and retirement only after all old sessions exit. Wider suite: **291 passed, 8 credential-dependent skipped, 1 deselected**. Ruff check, format check, patch clean-base `git apply --check`, and `git diff --check` passed. No provider/model calls; focused tests use fakes/loopback HTTP.

The deselected test is **`tests/unit/test_api.py::test_config_tree_imports_without_the_runtime`**. It also fails on unchanged `0512206` in this same environment (unexpected imported runtime packages); baseline output: `/tmp/cyberstrike-tunnel-baseline-import-test.txt`. Reproduce that one test on a separate unchanged checkout before classifying it as a patch regression. Initial wider failures for missing `kuhn_poker`/`reverse_text` were fixed by adding their package roots to `PYTHONPATH`, not dependency installs or production code changes.

Use the durable `handoffs/cyberstrike-tunnel-404/test_recovery.py` from the handoff/recipe publication for the combined test above; original scratch path was `/tmp/test_cyberstrike_tunnel_combined.py`. If the bundle has not yet arrived, request it rather than silently claiming a rerun. Review: `/tmp/cyberstrike-tunnel-repair-review.md`; original machine-readable results/constants: `/Users/azh/Documents/cyberstrike-tunnel-repair-validation.json`. That manifest's historical `not_deployed=true` describes offline validation time; source publication progress above is newer, and neither proves a live load. Prefer the durable bundle's deployment/probe/join records when handing this to a fresh clone.

Already done for this repair: publish the verifier commit, update/commit the root **`deps/verifiers` gitlink (submodule lock)**, and pin the recipe to the intended immutable root code commit. Before a future launch, reverify those immutable objects. Check any other launch lock/ref that embeds source identity. Verify `git ls-tree <root-commit> deps/verifiers`, the resolved checkout, and native import origins actually select the repaired verifier. Do not assume a branch name, root SDK lock, recipe edit, or source push changes an existing process. No dependency-version/SDK lock change was part of this repair.

## Live verification and provider escalation: only after approval

1. Prefer existing authorized records. Preserve exact run/attempt identity, UTC timestamps, trace→sandbox-name→applied-model-host→tunnel joins, and local/public application-health identities. Join model request IDs to complete, correctly scoped app access logs; absence is meaningful only with known logging coverage. Do not rely only on top-level HarnessError or SDK exception strings.
2. If authorized, compare simultaneous application identity health from host and guest; record local/public outcomes and vantage. A host 200 is not proof of guest reachability. A harmless health probe is not a model inference test. Do not fault-inject or delete live tunnels to reproduce the offline test. A model canary or new launch requires its own resource/cost authorization and checkpoint validation.
3. Minimal provider packet: run and attempt/generation; one or more exact trace/sandbox IDs; applied-policy generation/status/model host and tunnel ID; tunnel creation/expiry/status; UTC request start/end and vantage; safe method/origin/path; status, content type, byte length, template classification/hash, and allowlisted gateway/request IDs. Include the 120.233-second health result and state explicitly that the original model response was not captured. Never send auth, tokens, cookies, arbitrary headers/env, query strings, task/prompt/model bodies, or raw response bodies.
4. Ask the provider to correlate that exact tunnel/time/request with gateway/frps **route ownership and forwarding/work-connection errors**, registration instance, deployed versions/config, and any 120-second timeout. Public SDK create/get/delete/list and platform run GET do not expose those internal logs. Unknown internal evidence is a boundary, not permission to keep probing or a basis to declare a particular FRP bug. Use an approved destination before transmitting the packet.

## Checkpoint-aware pause: interpretation and resume gate

Pause was requested during trainer step 165 (start **09:40:10Z**, request about **10:10Z**). This implementation accepts pause only at the next loop boundary and **still trains, optimizes, broadcasts, and finally checkpoints that accepted step**. It does not stop before that step. Trainer accepted **166 at 10:34:07.986344Z**, action `16da9606608341cfb8dbafff3e29cbe4`. Orchestrator accepted/saved **166 at 11:27:06Z**, action `3bd8f31d17a04474a0ebc5972af57180`; policy 166 was received **11:34:57.154685Z**. Earlier disabled/undelivered-controller hypotheses are superseded by those receipts.

Platform `PAUSING` alone was never proof of local delivery. Local controller states are pending→accepted→committed; trainer commit requires its final save. Correct coordination is trainer first, learn accepted N, then orchestrator pause at exactly N. Resolved runtime control config/targets and same-attempt receipts matter; source TOML omissions alone cannot rule out hosted injection.

Before an authorized resume, confirm the final trainer and orchestrator state belong to the **same generation and step 166**, inspect the corresponding **joint manifest** and completion/durability evidence, and check applicable READY/pins/BEST/LATEST and final-sync records. Remote step 166/160 **READY** and final platform **PAUSED at 13:12:58.844Z** are the latest established state; independent joint-manifest/final-sync inspection and actual resume correctness remain unverified. Do not issue another pause, stop, resume, or launch merely to obtain evidence.

## Evidence entry points

- `/Users/azh/Documents/cyberstrike-exact-tunnel-rootcause.md`
- `/Users/azh/Documents/cyberstrike-pause-control-analysis.md` (later milestones supersede early hypotheses)
- `/Users/azh/Documents/cyberstrike-tunnel-repair-validation.json`
- `/tmp/cyberstrike-tunnel-repair-review.md`
- Exact endpoint probes: `/tmp/cyberstrike-deep-confirmed-endpoint-long-probe.json`, `/tmp/cyberstrike-deep-healthy-control-probe.json`; exact assignment joins: `/tmp/cyberstrike-deep-safe-page2-result.json`.

Assumptions: supplied immutable source pins describe the investigated run; host-side identity health is useful but not equivalent to every guest vantage; SDK traceback is a version-family fingerprint, not an exact installed-wheel attestation. New production constants are the 20-second interval and two-cycle failure threshold. Existing probe timeout, retry budget, SDK/dependency versions, and the paused run were not changed by this repair.

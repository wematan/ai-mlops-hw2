# REPORT

## Phase 1 - vLLM serving configuration

### Endpoint and model
- Endpoint: `http://localhost:8000`
- Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Hardware: `1x H100 80GB`

Flags and one-line justification:
- `--model Qwen/Qwen3-30B-A3B-Instruct-2507`: fixed assignment model.
- `--host 0.0.0.0`: expose server for local tooling and dashboards.
- `--port 8000`: required endpoint for this assignment flow.
- `--max-model-len 8096`: enough context headroom for 1.5-3K token prompts plus schema and completion.
- `--tensor-parallel-size 1`: single H100, no tensor sharding needed.
- `--enable-expert-parallel`: helps MoE expert routing efficiency on this model family.
- `--enable-auto-tool-choice`: enables automatic tool-choice behavior support in OpenAI-compatible calls.
- `--tool-call-parser qwen3_coder`: parser tuned to Qwen3-style tool outputs.
- `--reasoning-parser qwen3`: parser for Qwen3 reasoning format.

---

## Phase 5 - Baseline eval

Execution accuracy on the 30-question set (`results/eval_baseline.json`): the agent's final SQL vs the gold SQL, compared on canonicalized row sets.

| Metric | Value |
|---|---|
| Overall pass rate | **26.7%** (8/30) |
| Pass rate by iteration | 0.267 → 0.267 → 0.267 (flat) |
| fixed_by_revision / broken_by_revision | 0 / 0 |
| mean iterations | 1.53 |
| agent ran without error | 100% |

Read: on the baseline run the per-iteration curve is **flat** (0.267 at every step) and revise fixed **zero** questions, so here the loop bought no accuracy and only added latency. (The after-tuning run isn't flat — see Agent value.) The misses are subtly-wrong values the verifier can't catch from row *shape* alone.

---

## Phase 6 - Hitting the SLO

**Target:** P95 end-to-end agent latency < 5s at ≥10 RPS over a 5-minute window.

**Method:** `load_test/driver.py` (open-loop, fixed RPS) run on the VM against the agent; vLLM `/metrics` read from Prometheus per window. The driver uses a fixed RNG seed, so every run replays the identical question sequence (controlled comparison).

### Baseline (as-delivered: 1 uvicorn worker, MAX_ITERATIONS=3) — 10 RPS × 120s
p50 **40.9s**, p95 **89.0s**, p99 97.8s; 651/1200 ok (54%), 392 client + 155 HTTP errors. **SLO missed by ~18×.**
vLLM during the collapse: queue 0, KV 28%, preemptions 0, `num_requests_running` peaked at **39** — vLLM was *idle*; the bottleneck was above it.

### Iteration log

1. *saw* vLLM idle (queue 0, KV 28%, `running` stuck at 39 = the 40-thread sync pool) while agent p95=89s with 392 connection errors → *hypothesized* one sync `uvicorn` worker caps concurrency → *changed* to `--workers 4` → *result* `running` 39→114, client_errors 392→1, p50 41→6.6s, but **p95 still 64s**.
2. *saw* ~156 HTTP 500s constant under load and vLLM gen frozen at ~780 tok/s despite 3× the concurrency → *hypothesized* a content-driven crash (not load) starving the feed → *changed* guarded `render_schema`'s `_q(None)` FK crash on 2 DBs, and raised workers 4→16 → *result* errors→0, **p95 64→18s**, gen 780→1407 tok/s (the "decode wall" was GIL starvation, not vLLM).
3. *saw* (pushing past: swept load down) p95=8s even at 3 RPS, KV at 81% → *hypothesized* vLLM decode/KV is now the ceiling (~13 call/s × ~3 calls/run ≈ 3-4 RPS) and the revise tail sets the p95 floor → *changed* nothing — needs a vLLM restart (FP8 / drop `--enable-expert-parallel`) or cutting `MAX_ITERATIONS` → *result* **breaking point ~3-4 RPS; the SLO gap is structural, not a knob**.

### Final numbers (16 workers + schema fix, MAX_ITERATIONS=3)

| 10 RPS × 120s | Baseline (1w) | Final |
|---|---|---|
| ok | 54% | **98.5%** |
| HTTP / client errors | 155 / 392 | **0 / 10** |
| p50 | 40.9s | **2.68s** |
| p95 | 89.0s | **18.1s** |
| p99 | 97.8s | 27.8s |

Capacity sweep (final config, 60s each): 3 RPS → p95 8.0s; 4 RPS → p95 8.9s, p99 49.7s (backlog forming).

vLLM by stage (Prometheus):

| Window | running | KV | decode p95 | call/s | gen tok/s |
|---|---|---|---|---|---|
| Baseline, 1 worker | 39 | 28% | 2.24s | 13.7 | 781 |
| 4 workers | 114 | 33% | 3.43s | 13.7 | 778 |
| 16 workers + fix | 76 | 81% | 4.63s | 13.4 | 1407 |

### Verdict: missed it

We didn't hit the SLO - at 10 RPS the p95 is 18s against a 5s target, so about 3.6x over. But the as-delivered system didn't just miss, it fell over (p95 89s, half the requests failing), and the diagnosis is the part I'd stand behind: vLLM was never the problem. Even with the agent at 89s, vLLM's queue was empty and KV was at 28% - the bottleneck was always above it, first a single sync worker, then a schema-render bug burning CPU under the GIL on two of the databases. Fixing those took p95 from 89s to 18s, with no errors and a 2.7s median.

What's left isn't a bug, it's the shape of the agent. With 16 workers it has threads to spare, so vLLM is the limit now (~13 calls/s, KV at 81%), and each question is two to three of those calls in a row - call it 3-4 questions/sec before it backlogs. That's also why even at 3 RPS the p95 is 8s: the slowest ~40% of questions take a revise round, four to six calls back to back. Quality held through all of it - the final answer rate is still 26.7%, same as baseline, so the latency work didn't cost any accuracy (the per-iteration path wiggled a little; more on that below).

Screenshots `screenshots/grafana_before.png` (1-worker baseline) and `screenshots/grafana_after.png` (16 workers + fix) show the window where it turned around.

---

## Agent value

Did the loop help? A little, but mostly it's within noise. In the final-config eval the per-iteration pass rate isn't flat, it's 0.267 → 0.30 → 0.267: the first revision genuinely fixes one query (it adds the `DISTINCT` the formula_1 circuits question was missing), but the second one undoes the gain - a card_games query flips right at iteration 1 and wrong again at iteration 2. The baseline run was dead flat by comparison (0.267 the whole way, revise fixed nothing), so a one-question swing is really inside the run-to-run noise from vLLM batching. If I trust the final curve at all, the peak is at one revision and the second only hurts, which argues for `MAX_ITERATIONS=2` rather than 3. The deeper reason it can't do better: the verifier only sees the *shape* of the result, so it catches empty tables, extra columns, and errored queries but not how this agent usually goes wrong - 'M' vs 'm' in a filter, or an aggregate where a list was asked. And a revise round is four to six sequential calls, which is the whole p95 tail.

## What I'd do with more time

First thing I'd try is FP8 - we're at 81% KV with only 76 requests in flight and the weights are still bf16, so quantizing should roughly double decode throughput and free a chunk of cache. I'd pull `--enable-expert-parallel` in the same restart; it's for sharding experts across GPUs and we're on one H100, so it's doing nothing and might be why token throughput is lower than I'd expect. After that, shorten the critical path: drop `--max-model-len` from 8096, cap `max_tokens`, and skip verify when the query already ran and returned rows. I'd also precompute the schemas - there are only 11 databases and the rendered text is identical on every request, so instead of introspecting sqlite on the request path (it's lru-cached per process today, but each of the 16 workers keeps its own copy, and that's the exact path the FK bug kept re-running) I'd render all 11 once at startup and share them. Longer term I'd make the agent async (`ainvoke` + a shared client) so the worker count stops being something I tune by hand.


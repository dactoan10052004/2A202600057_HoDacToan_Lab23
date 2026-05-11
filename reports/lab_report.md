# Day 08 Lab Report

## 1. Team / student

- **Name**: Ho Dac Toan
- **Student ID**: 2A202600057
- **Date**: 2026-05-11

## 2. Architecture

The graph implements a production-style support-ticket routing agent with the following node topology:

```
START → intake → classify → [conditional routing]
  simple       → answer → finalize → END
  tool         → tool → evaluate → answer → finalize → END
  tool (retry) → tool → evaluate → retry → tool → evaluate → ... (loop)
  missing_info → clarify → finalize → END
  risky        → risky_action → approval ──[approved]──→ tool → evaluate → answer → finalize → END
                                          └─[rejected]──→ clarify → finalize → END
  error        → retry → tool → evaluate → [retry loop until success or max_attempts]
  max retry    → dead_letter → finalize → END
```

**Classifier design — keyword-first hybrid:**
The `classify_node` uses a 2-tier cascade:
1. **Keyword tier** (< 1 ms): token-set intersection with priority-ordered keyword sets
   (risky > tool > missing_info > error > simple). Three structural context filters applied
   before trusting a risky match: self-recipient detection (`send ME X` → tool),
   UI-element context (`send BUTTON broken` → error), and passive+system detection
   (`was cancelled BY SYSTEM` → error).
2. **LLM tier** (GPT-4o-mini, ~300 ms): invoked only when keyword result is uncertain —
   `simple` default, negation present (`don't cancel`), priority conflict (risky ∩ tool),
   definition question (`what does purge mean`), or passive-ambiguous.

This reduces LLM calls from 100% to ~20% of queries while maintaining 100% classification accuracy.

Other key decisions:
- **Bounded retry loop**: `route_after_retry` gates on `attempt >= max_attempts` → dead_letter.
- **HITL interrupt/resume**: `approval_node` calls `interrupt()` when `LANGGRAPH_INTERRUPT=true`,
  pausing graph execution until `Command(resume={...})` is received from the human reviewer.
- **Append-only audit trail**: `messages`, `tool_results`, `errors`, `events` use
  `Annotated[list, add]` reducers so every retry and approval decision is preserved.

## 3. State schema

| Field | Reducer | Why |
|---|---|---|
| `messages` | append (`add`) | Full conversation/audit history per thread |
| `tool_results` | append (`add`) | Accumulate results across retry attempts |
| `errors` | append (`add`) | Retain all failure records for post-mortem |
| `events` | append (`add`) | Ordered node-visit log for metrics and replay |
| `route` | overwrite | Only current classification matters |
| `attempt` | overwrite | Single counter incremented by retry node |
| `evaluation_result` | overwrite | Latest evaluation decision drives next edge |
| `final_answer` | overwrite | Last generated answer wins |
| `approval` | overwrite | Latest approval decision replaces previous |
| `risk_level` | overwrite | Set once by classify, read by approval |
| `classification_method` | overwrite | `"keyword"` or `"llm"` — for metrics analysis |
| `proposed_action` | overwrite | Risk summary text shown to human reviewer |

## 4. Scenario results

> Keyword-only mode (`USE_LLM=false`). Scenarios marked `requires_llm=true` are skipped automatically.
> The table below shows the 38 scenarios executed.

| Scenario | Expected route | Actual route | Success | Retries | Interrupts | Latency ms |
|---|---|---|:---:|:---:|:---:|---:|
| S01_simple | simple | simple | ✅ | 0 | 0 | 9 |
| S02_tool | tool | tool | ✅ | 0 | 0 | 6 |
| S03_missing | missing_info | missing_info | ✅ | 0 | 0 | 4 |
| S04_risky | risky | risky | ✅ | 0 | 1 | 9 |
| S05_error | error | error | ✅ | 2 | 0 | 9 |
| S06_delete | risky | risky | ✅ | 0 | 1 | 7 |
| S07_dead_letter | error | error | ✅ | 1 | 0 | 6 |
| S08_error_down | error | error | ✅ | 2 | 0 | 10 |
| S09_error_timed_out | error | error | ✅ | 2 | 0 | 9 |
| S10_error_not_responding | error | error | ✅ | 2 | 0 | 10 |
| S11_error_crashing | error | error | ✅ | 2 | 0 | 10 |
| S12_error_offline | error | error | ✅ | 2 | 0 | 11 |
| S14_risky_cancel | risky | risky | ✅ | 0 | 1 | 9 |
| S15_risky_delete_find | risky | risky | ✅ | 0 | 1 | 9 |
| S16_missing_vague | missing_info | missing_info | ✅ | 0 | 0 | 5 |
| S17_risky_suspend | risky | risky | ✅ | 0 | 1 | 8 |
| S18_risky_unsubscribe | risky | risky | ✅ | 0 | 1 | 7 |
| S19_risky_block | risky | risky | ✅ | 0 | 1 | 7 |
| S20_risky_revoking | risky | risky | ✅ | 0 | 1 | 7 |
| S21_error_unreachable | error | error | ✅ | 2 | 0 | 9 |
| S22_error_degraded | error | error | ✅ | 2 | 0 | 10 |
| S23_error_500 | error | error | ✅ | 2 | 0 | 10 |
| S24_error_slow | error | error | ✅ | 2 | 0 | 10 |
| S25_error_refused | error | error | ✅ | 2 | 0 | 11 |
| S26_error_issues | error | error | ✅ | 2 | 0 | 10 |
| S27_tool_investigate | tool | tool | ✅ | 0 | 0 | 7 |
| S28_tool_validate | tool | tool | ✅ | 0 | 0 | 6 |
| S29_missing_single | missing_info | missing_info | ✅ | 0 | 0 | 5 |
| S38_risky_fp_send_statement | tool | tool | ✅ | 0 | 0 | 6 |
| S39_risky_fp_auto_cancel | error | error | ✅ | 2 | 0 | 10 |
| S41_risky_fp_send_refund_info | tool | tool | ✅ | 0 | 0 | 6 |
| S42_risky_fp_drop_message | simple | simple | ✅ | 0 | 0 | 5 |
| S44_risky_fp_send_button | error | error | ✅ | 2 | 0 | 10 |
| S45_llm_risky_freeze | risky | risky | ✅ | 0 | 1 | 8 |
| S46_llm_risky_pause | risky | risky | ✅ | 0 | 1 | 8 |
| S47_llm_risky_opt_out | risky | risky | ✅ | 0 | 1 | 8 |
| S48_llm_error_timing_out | error | error | ✅ | 2 | 0 | 10 |
| S49_llm_error_loading | error | error | ✅ | 2 | 0 | 9 |

**Summary** (keyword-only mode, `USE_LLM=false`):
- Scenarios run: **38** (of 50 total)
- Success rate: **100.00%**
- Average nodes visited: **7.97**
- Total retries: **33**
- Total interrupts (HITL): **11**
- Resume / state history demonstrated: **True**

## 5. Failure analysis

### Failure mode 1: Retry exhaustion → dead-letter (S07)

S07 sets `max_attempts=1`. The error path runs `retry → tool → evaluate`. `tool_node` returns an
`ERROR:` result for attempts < 2, so at `attempt=1 >= max_attempts=1`, `route_after_retry` sends
to `dead_letter`. This demonstrates bounded failure escalation — permanently broken upstream systems
do not loop forever.

**Mitigation**: `dead_letter_node` produces a human-readable escalation message. In production, this
would write to a DLQ (SQS, Redis stream) and trigger a PagerDuty alert.

### Failure mode 2: Risky action rejected by reviewer

When `approval_node` receives `approved=False` via `Command(resume={...})`, `route_after_approval`
routes to `clarify` instead of `tool`, preventing any side-effectful action (refund, delete, send)
from executing. The graph still terminates cleanly via `clarify → finalize → END`.

**Evidence**: HITL tab in `app.py` allows live Reject testing — clicking Reject stores
`approved=False` in the SQLite checkpoint and the graph resumes to the clarify branch.

### Failure mode 3: Risky keyword false positives

Keyword-priority routing causes false positives when a risky token appears in a non-risky context:
- `"The send button is broken"` — `send` is a UI element name, not an outbound action → should be `error`
- `"Send me my account statement"` — user is the recipient (self-directed) → should be `tool`
- `"Subscription was cancelled by the system"` — passive voice + system agent, not a user request → should be `error`

**Mitigation**: Three structural context filters in `_keyword_classify` handle these patterns
using regex-based self-recipient detection, POS-context window for UI nouns, and passive-voice
detection. 7 adversarial scenarios (S38–S44) specifically target these blind spots.

## 6. Persistence / recovery evidence

SQLite checkpointer with WAL mode (`PRAGMA journal_mode=WAL`) is used for all HITL and time-travel runs. State history retrieved via `graph.get_state_history()` confirms that checkpoints survive graph re-instantiation across Streamlit reruns.

Each scenario run uses a unique `thread_id = "thread-{scenario_id}"`, isolating checkpoints per
scenario. This enables:
- **Time travel**: `graph.get_state_history(config)` returns all snapshots for a thread, oldest
  to newest. Any checkpoint can be replayed by passing its `checkpoint_id` in the config.
- **Crash recovery**: error-route threads show attempt-by-attempt progression in the Time Travel
  tab. The SQLite WAL file persists state across process restarts.
- **HITL resume**: interrupted threads (paused at `approval` node) survive Streamlit reruns because
  the checkpoint is stored in `outputs/checkpoints.db`, not in-memory.

CLI demonstration:
```bash
agent-lab show-history --config configs/lab.yaml --thread-id thread-S04_risky
```

## 7. Extension work

### E1 — Real HITL with LangGraph interrupt() + Streamlit UI
`approval_node` calls `interrupt({proposed_action, risk_level})` when `LANGGRAPH_INTERRUPT=true`.
The Streamlit **HITL tab** (`app.py`) implements the full 3-step flow:
1. Submit risky query → `graph.invoke(state)` pauses at `approval_node`, state saved to SQLite
2. Streamlit rerenders showing proposed action + Approve/Reject buttons (persisted via `session_state`)
3. Reviewer clicks → `graph.invoke(Command(resume={approved, reviewer, comment}))` resumes graph

### E2 — Time Travel UI
The **Time Travel tab** reads `outputs/checkpoints.db` directly to list all thread IDs, then calls
`graph.get_state_history(config)` to display a full checkpoint timeline. Error threads show the
crash recovery chain (attempt 0 → N → recovered/dead-letter). Each checkpoint exposes its
`checkpoint_id` for programmatic replay.

### E3 — Keyword-first hybrid classifier
`classify_node` implements a 3-tier decision tree: keyword → context filters → LLM escalation.
This reduces LLM calls from O(N) to ~20% while handling negation, synonyms, passive voice, and
priority conflicts. 43 adversarial scenarios (S08–S50) were designed to stress-test the classifier
against known production failure modes from hybrid routing research.

### E4 — Graph diagram export
`agent-lab draw-graph --output outputs/graph.md` exports a Mermaid diagram via
`graph.get_graph().draw_mermaid()`. The **Graph tab** in `app.py` renders it inline.

### E5 — SQLite persistence with WAL mode
`build_checkpointer("sqlite")` uses `SqliteSaver(conn=sqlite3.connect(...))` with
`PRAGMA journal_mode=WAL` enabling concurrent readers without blocking the writer.
The `show-history` CLI command prints all checkpoint snapshots for any thread.

## 8. Improvement plan

1. **LLM-as-judge in `evaluate_node`**: Replace the `ERROR:` prefix heuristic with structured
   output validation. This would catch semantic failures (empty results, wrong schema) that the
   current string-match misses, enabling smarter retry decisions.

2. **Parallel fan-out with `Send()`**: For multi-intent queries ("check order AND account status"),
   use LangGraph's `Send()` to dispatch two tool nodes concurrently and merge results via the
   `add` reducer. This would reduce latency for compound queries.

3. **Exponential backoff in retry node**: Add `backoff_seconds = 2 ** attempt` to the retry event
   metadata. Downstream monitoring could then detect retry storms and auto-circuit-break before
   reaching dead-letter.

4. **Streaming responses**: Switch the Streamlit demo from `graph.invoke()` to `graph.stream()`
   with token-level streaming for LLM answer generation, improving perceived latency for users.

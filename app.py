"""Streamlit demo UI for the LangGraph Agent Lab.

Run:  streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import time

import streamlit as st

sys.path.insert(0, "src")

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="LangGraph Agent Lab",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Late imports (after sys.path) ────────────────────────────────────────────
from langgraph_agent_lab.graph import build_graph          # noqa: E402
from langgraph_agent_lab.persistence import build_checkpointer  # noqa: E402
from langgraph_agent_lab.scenarios import load_scenarios   # noqa: E402
from langgraph_agent_lab.state import Route, Scenario, initial_state  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────
ROUTE_META: dict[str, tuple[str, str, str, str]] = {
    "simple":       ("🟢", "Simple",       "#d4edda", "#155724"),
    "tool":         ("🔵", "Tool Lookup",   "#cce5ff", "#004085"),
    "risky":        ("🔴", "Risky Action",  "#f8d7da", "#721c24"),
    "missing_info": ("🟡", "Missing Info",  "#fff3cd", "#856404"),
    "error":        ("🟠", "Error / Retry", "#fde8d8", "#7d4000"),
}

PRESETS: dict[str, str | None] = {
    "── Keyword mode ──": None,
    "Simple — reset password":              "How do I reset my password?",
    "Tool — lookup order 12345":            "Please lookup order status for order 12345",
    "Risky — refund customer":              "Refund this customer and send confirmation email",
    "Risky — cancel subscription":          "I need to cancel my subscription",
    "Missing info — vague pronoun":         "Can you fix it?",
    "Error — timeout failure":              "Timeout failure while processing request",
    "Error — server down":                  "The production server is down",
    "── LLM mode (USE_LLM=true) ──": None,
    "LLM — negation (don't delete)":        "I don't want to delete my account",
    "LLM — negation (do not cancel)":       "Please do not cancel my subscription",
    "LLM — semantic risky (discontinue)":   "I wish to discontinue my service entirely",
    "LLM — semantic error (bouncing)":      "My payment keeps bouncing every time",
    "LLM — semantic tool (pull up acct)":   "Can someone pull up my account details?",
    "LLM — implied stop":                   "I would like to stop using your platform",
    "LLM — word question (purge)":          "What does the word purge mean in your system?",
}

HITL_PRESETS: list[str] = [
    "Refund this customer and send confirmation email",
    "Delete customer account after support verification",
    "I need to cancel my subscription",
    "Please suspend my account immediately",
    "Block this user from accessing the platform",
    "Revoking API key for this client",
    "Please freeze my account while I travel",
]

SCENARIO_PATH = "data/sample/scenarios.jsonl"
SQLITE_DB_PATH = "outputs/checkpoints.db"


# ── Helpers ──────────────────────────────────────────────────────────────────
def _apply_settings(use_llm: bool, api_key: str, model: str) -> None:
    os.environ["LANGGRAPH_INTERRUPT"] = "false"
    os.environ["USE_LLM"] = "true" if use_llm else "false"
    if use_llm and api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if use_llm and model:
        os.environ["LLM_MODEL"] = model


@st.cache_resource
def _build_graph():  # type: ignore[return]
    return build_graph(checkpointer=build_checkpointer("memory"))


@st.cache_resource
def _build_sqlite_graph():  # type: ignore[return]
    """SQLite-backed graph for HITL interrupt/resume and time-travel demos."""
    os.makedirs("outputs", exist_ok=True)
    return build_graph(checkpointer=build_checkpointer("sqlite", SQLITE_DB_PATH))


def _run_query(query: str) -> tuple[dict, list[tuple[str, dict]], int]:
    """Invoke graph and return (final_state, node_trace, latency_ms)."""
    graph = _build_graph()
    scenario = Scenario(id="demo", query=query, expected_route=Route.SIMPLE)
    state = initial_state(scenario)
    cfg = {"configurable": {"thread_id": f"demo-{time.time_ns()}"}}

    trace: list[tuple[str, dict]] = []
    t0 = time.perf_counter()
    for chunk in graph.stream(state, config=cfg, stream_mode="updates"):
        for node_name, node_state in chunk.items():
            trace.append((node_name, node_state))
    latency_ms = int((time.perf_counter() - t0) * 1000)

    final = graph.get_state(cfg).values
    return dict(final), trace, latency_ms


def _route_badge(route: str) -> None:
    emoji, label, bg, fg = ROUTE_META.get(route, ("⚪", route, "#eee", "#333"))
    st.markdown(
        f"<div style='background:{bg};color:{fg};padding:14px;border-radius:10px;"
        f"font-size:20px;font-weight:bold;text-align:center;margin-bottom:10px;'>"
        f"{emoji} &nbsp; {label.upper()}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _event_timeline(events: list[dict]) -> None:
    for ev in events:
        node = ev.get("node", "?")
        msg = ev.get("message", "")
        ev_type = ev.get("event_type", "")
        meta = ev.get("metadata", {})

        icon = {"completed": "✅", "warning": "⚠️", "pending_approval": "🔒"}.get(
            ev_type, "▶"
        )
        meta_parts = [f"`{k}={str(v)[:40]}`" for k, v in list(meta.items())[:3]]
        meta_str = ("  " + " · ".join(meta_parts)) if meta_parts else ""
        st.markdown(f"{icon} **`{node}`** &mdash; {msg}{meta_str}")


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    use_llm = st.toggle(
        "Enable LLM (OpenAI)",
        value=os.getenv("USE_LLM", "false").lower() == "true",
        help="Uses GPT-4o-mini for semantic classification, natural answers, and "
             "context-aware clarifications.",
    )

    api_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Required when LLM is enabled.",
    )

    model = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        disabled=not use_llm,
    )

    st.divider()
    st.caption(
        "**Keyword mode** — free, deterministic, instant\n\n"
        "**LLM mode** — handles negation, synonyms, implied intent"
    )

    if use_llm and not api_key:
        st.warning("Add your OpenAI API key above.")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_demo, tab_hitl, tab_batch, tab_history, tab_graph = st.tabs([
    "🧪 Demo", "🔒 HITL", "📊 Batch Run", "⏱ Time Travel", "🗺 Graph",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Demo
# ════════════════════════════════════════════════════════════════════════════
with tab_demo:
    st.header("Live Agent Demo")
    st.caption("Type a query or pick a preset, then click Run.")

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        preset_key = st.selectbox("Preset scenarios", list(PRESETS.keys()))
        preset_val = PRESETS[preset_key]

        query = st.text_area(
            "Customer query",
            value=preset_val or "",
            height=120,
            placeholder="e.g. 'Please refund order 9900'",
        )

        run_clicked = st.button(
            "▶  Run Agent",
            type="primary",
            use_container_width=True,
            disabled=not query.strip(),
        )

    with col_out:
        if run_clicked and query.strip():
            _apply_settings(use_llm, api_key, model)
            _build_graph.clear()

            with st.spinner("Running graph…"):
                try:
                    final, trace, latency_ms = _run_query(query.strip())
                except Exception as exc:
                    st.error(f"Graph error: {exc}")
                    st.stop()

            route = final.get("route", "unknown")
            method = final.get("classification_method", "keyword")

            _route_badge(route)

            meta_col1, meta_col2, meta_col3 = st.columns(3)
            meta_col1.metric("Latency", f"{latency_ms} ms")
            meta_col2.metric("Classifier", method)
            meta_col3.metric("Risk Level", final.get("risk_level", "—"))

            st.divider()

            answer = final.get("final_answer") or final.get("pending_question")
            if answer:
                if route == "risky":
                    reviewer = (final.get("approval") or {}).get("reviewer", "mock")
                    st.info(f"**Approved by {reviewer}:**\n\n{answer}")
                elif route == "missing_info":
                    st.warning(f"**Clarification needed:**\n\n{answer}")
                elif route == "error":
                    st.error(f"**Error handled:**\n\n{answer}")
                else:
                    st.success(f"**Answer:**\n\n{answer}")

            errors = final.get("errors", [])
            if errors:
                with st.expander(f"🔁 Retries ({len(errors)})"):
                    for e in errors:
                        st.text(e)

            with st.expander("📋 Event Timeline", expanded=True):
                _event_timeline(final.get("events", []))

            with st.expander("🔍 Node Execution Trace"):
                for node_name, ns in trace:
                    changed = [k for k in ns if k != "events"]
                    st.markdown(f"**`{node_name}`** → {', '.join(changed) or '—'}")

        elif not run_clicked:
            st.info("← Fill in a query and click **Run Agent**.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — HITL
# ════════════════════════════════════════════════════════════════════════════
with tab_hitl:
    st.header("Human-in-the-Loop (HITL) Demo")
    st.caption(
        "Uses LangGraph `interrupt()` + SQLite checkpointer. "
        "Graph pauses at the approval gate, waits for human decision, then resumes "
        "via `Command(resume=...)`."
    )

    # Initialise session state for multi-step HITL flow
    for _k, _default in [
        ("hitl_phase", "idle"),     # idle | pending | done
        ("hitl_thread_id", None),
        ("hitl_cfg", None),
        ("hitl_proposed", ""),
        ("hitl_risk_level", "high"),
        ("hitl_next", []),
        ("hitl_final", {}),
        ("hitl_approved", False),
    ]:
        if _k not in st.session_state:
            st.session_state[_k] = _default

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        hitl_preset = st.selectbox("Preset risky scenarios", HITL_PRESETS, key="hitl_preset_sel")
        hitl_query = st.text_area("Customer query", value=hitl_preset, height=100, key="hitl_q")

        submit_hitl = st.button(
            "▶  Submit to Agent",
            type="primary",
            disabled=not hitl_query.strip(),
            use_container_width=True,
            key="hitl_submit",
        )

        if submit_hitl and hitl_query.strip():
            os.environ["LANGGRAPH_INTERRUPT"] = "true"
            os.environ["USE_LLM"] = "false"

            sqlite_graph = _build_sqlite_graph()
            thread_id = f"hitl-{time.time_ns()}"
            scenario = Scenario(id="hitl-demo", query=hitl_query.strip(), expected_route=Route.RISKY)
            state = initial_state(scenario)
            cfg = {"configurable": {"thread_id": thread_id}}

            with st.spinner("Running graph to approval gate…"):
                sqlite_graph.invoke(state, config=cfg)

            current = sqlite_graph.get_state(cfg)

            if not current.next:
                # Graph completed without hitting interrupt (non-risky query)
                st.session_state.hitl_phase = "done"
                st.session_state.hitl_final = dict(current.values)
                st.session_state.hitl_approved = True
                st.session_state.hitl_cfg = cfg
            else:
                st.session_state.hitl_phase = "pending"
                st.session_state.hitl_thread_id = thread_id
                st.session_state.hitl_cfg = cfg
                st.session_state.hitl_proposed = current.values.get(
                    "proposed_action", "No description available."
                )
                st.session_state.hitl_risk_level = current.values.get("risk_level", "high")
                st.session_state.hitl_next = list(current.next)
            st.rerun()

        # Phase status chip
        phase = st.session_state.hitl_phase
        if phase == "idle":
            st.info("Pick a risky scenario above and click **Submit to Agent**.")
        elif phase == "pending":
            short_id = (st.session_state.hitl_thread_id or "")[-12:]
            st.warning(f"🔒 Graph paused — thread `…{short_id}`")
        elif phase == "done":
            outcome = "✅ Approved" if st.session_state.hitl_approved else "❌ Rejected"
            st.success(f"Flow complete — {outcome}")

    with col_right:
        phase = st.session_state.hitl_phase

        if phase == "idle":
            st.markdown(
                "### How HITL works\n"
                "1. Submit a risky query → graph runs `intake → classify → risky_action`\n"
                "2. `approval_node` calls `interrupt()` — execution pauses, state saved to SQLite\n"
                "3. Streamlit page rerenders with the proposed action\n"
                "4. Reviewer approves or rejects\n"
                "5. Graph resumes via `Command(resume={approved: ..., reviewer: ...})`\n"
                "6. Remaining nodes execute; final result shown\n\n"
                "_State survives Streamlit reruns because SQLite checkpointer persists "
                "across script executions._"
            )

        elif phase == "pending":
            st.warning("🔒 **Graph paused — awaiting human approval**")

            with st.container(border=True):
                st.markdown("#### Proposed Action")
                st.markdown(st.session_state.hitl_proposed or "_No description available_")
                _, _, bg, fg = ROUTE_META.get("risky", ("", "", "#f8d7da", "#721c24"))
                risk = st.session_state.hitl_risk_level
                st.markdown(
                    f"<span style='background:{bg};color:{fg};padding:4px 10px;"
                    f"border-radius:6px;font-weight:bold;font-size:13px;'>"
                    f"Risk: {risk.upper()}</span>",
                    unsafe_allow_html=True,
                )
                st.caption(f"Paused at: `{st.session_state.hitl_next}`")

            st.divider()
            comment = st.text_input(
                "Reviewer comment (optional)",
                placeholder="e.g. 'Verified with customer over phone'",
                key="hitl_comment",
            )

            col_a, col_r = st.columns(2)
            approve_btn = col_a.button("✅ Approve", type="primary", use_container_width=True)
            reject_btn = col_r.button("❌ Reject", use_container_width=True)

            if approve_btn or reject_btn:
                from langgraph.types import Command  # type: ignore[import-untyped]

                os.environ["LANGGRAPH_INTERRUPT"] = "true"
                sqlite_graph = _build_sqlite_graph()
                resume_val = {
                    "approved": bool(approve_btn),
                    "reviewer": "human-reviewer",
                    "comment": comment or ("approved" if approve_btn else "rejected by reviewer"),
                }
                with st.spinner("Resuming graph…"):
                    final = sqlite_graph.invoke(
                        Command(resume=resume_val),
                        config=st.session_state.hitl_cfg,
                    )
                st.session_state.hitl_phase = "done"
                st.session_state.hitl_final = dict(final)
                st.session_state.hitl_approved = bool(approve_btn)
                st.rerun()

        elif phase == "done":
            final = st.session_state.hitl_final
            approved = st.session_state.hitl_approved

            _route_badge(final.get("route", "risky"))

            if approved:
                st.success("✅ Action approved and executed")
            else:
                st.error("❌ Action rejected — sent to clarification")

            answer = final.get("final_answer") or final.get("pending_question")
            if answer:
                st.info(f"**Result:**\n\n{answer}")

            approval_data = final.get("approval") or {}
            if approval_data:
                c1, c2 = st.columns(2)
                c1.metric("Reviewer", approval_data.get("reviewer", "—"))
                c2.metric("Decision", "Approved" if approval_data.get("approved") else "Rejected")
                if approval_data.get("comment"):
                    st.caption(f"Comment: _{approval_data['comment']}_")

            with st.expander("📋 Full Event Timeline"):
                _event_timeline(final.get("events", []))

            if st.button("🔄 Run Another", use_container_width=True):
                st.session_state.hitl_phase = "idle"
                os.environ["LANGGRAPH_INTERRUPT"] = "false"
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Batch Run
# ════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.header("Batch Scenario Run")
    st.caption(f"Runs all scenarios from `{SCENARIO_PATH}`.")

    if st.button("▶  Run All Scenarios", type="primary"):
        _apply_settings(use_llm, api_key, model)
        _build_graph.clear()

        try:
            scenarios = load_scenarios(SCENARIO_PATH)
        except Exception as exc:
            st.error(f"Failed to load scenarios: {exc}")
            st.stop()

        progress = st.progress(0, text="Starting…")
        rows = []

        for i, sc in enumerate(scenarios):
            progress.progress((i + 1) / len(scenarios), text=f"Running {sc.id}…")

            if sc.requires_llm and not use_llm:
                rows.append({
                    "ID": sc.id,
                    "Query": sc.query[:55],
                    "Expected": sc.expected_route.value,
                    "Got": "skipped",
                    "✓": "⏭",
                    "ms": "—",
                    "Method": "—",
                })
                continue

            try:
                final, _, latency_ms = _run_query(sc.query)
                actual = final.get("route", "?")
                ok = actual == sc.expected_route.value
                rows.append({
                    "ID": sc.id,
                    "Query": sc.query[:55],
                    "Expected": sc.expected_route.value,
                    "Got": actual,
                    "✓": "✅" if ok else "❌",
                    "ms": latency_ms,
                    "Method": final.get("classification_method", "?"),
                })
            except Exception as exc:
                rows.append({
                    "ID": sc.id,
                    "Query": sc.query[:55],
                    "Expected": sc.expected_route.value,
                    "Got": "ERROR",
                    "✓": "💥",
                    "ms": "—",
                    "Method": str(exc)[:40],
                })

        progress.empty()

        passed = sum(1 for r in rows if r["✓"] == "✅")
        skipped = sum(1 for r in rows if r["✓"] == "⏭")
        total = len(rows)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", total)
        c2.metric("Passed ✅", passed)
        c3.metric("Failed ❌", total - passed - skipped)
        c4.metric("Success Rate", f"{passed / (total - skipped):.0%}" if total - skipped else "—")

        st.dataframe(rows, use_container_width=True, height=600)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Time Travel
# ════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.header("Time Travel & Crash Recovery")
    st.caption(
        "Browse SQLite checkpoint history. Each snapshot is a full saved state — "
        "LangGraph can replay from any checkpoint. Error threads show the full retry chain."
    )

    import sqlite3 as _sqlite3
    from pathlib import Path as _Path

    def _list_thread_ids() -> list[str]:
        if not _Path(SQLITE_DB_PATH).exists():
            return []
        try:
            conn = _sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
            rows = conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id DESC"
            ).fetchall()
            conn.close()
            return [r[0] for r in rows]
        except Exception:
            return []

    thread_ids = _list_thread_ids()

    if not thread_ids:
        st.info(
            "No checkpoints yet. Run the **HITL** tab or the CLI with `checkpointer: sqlite`. "
            "Each run stores full state history in `outputs/checkpoints.db`."
        )
        st.code(
            "# 1. Edit data/sample/config.yaml:\n"
            "#    checkpointer: sqlite\n\n"
            "# 2. Run:\n"
            "agent-lab run-scenarios --config data/sample/config.yaml "
            "--output outputs/metrics.json",
            language="bash",
        )
    else:
        col_sel, col_refresh = st.columns([4, 1])
        selected_thread = col_sel.selectbox("Thread ID", thread_ids, key="history_thread")
        if col_refresh.button("🔄", use_container_width=True, help="Refresh thread list"):
            thread_ids = _list_thread_ids()
            st.rerun()

        if selected_thread:
            sqlite_graph = _build_sqlite_graph()
            os.environ.setdefault("LANGGRAPH_INTERRUPT", "false")
            cfg = {"configurable": {"thread_id": selected_thread}}

            try:
                history = list(sqlite_graph.get_state_history(cfg))
            except Exception as exc:
                st.error(f"Failed to load history: {exc}")
                history = []

            if not history:
                st.warning("No history snapshots found for this thread.")
            else:
                final_snap = history[0]   # newest-first
                total_attempts = final_snap.values.get("attempt", 0)
                final_route = final_snap.values.get("route", "—")
                n_checkpoints = len(history)
                is_paused = bool(final_snap.next)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Route", final_route)
                m2.metric("Checkpoints", n_checkpoints)
                m3.metric("Retry Attempts", total_attempts)
                m4.metric("Status", "⏸ Paused" if is_paused else "✅ Done")

                # ── Crash Recovery section ────────────────────────────────
                if total_attempts > 0:
                    st.divider()
                    st.subheader("Crash Recovery Chain")
                    st.caption("This thread hit the retry loop — showing attempt-by-attempt progression.")

                    final_answer = final_snap.values.get("final_answer", "")
                    is_dead_letter = (
                        "dead-letter" in final_answer
                        or "could not be completed" in final_answer
                    )

                    cols = st.columns(total_attempts + 1)
                    for att in range(total_attempts + 1):
                        if att < total_attempts:
                            cols[att].metric(f"Attempt {att}", "❌ failed")
                        elif is_dead_letter:
                            cols[att].metric(f"Attempt {att}", "⚠️ dead-letter")
                        else:
                            cols[att].metric(f"Attempt {att}", "✅ recovered")

                    if final_answer:
                        if is_dead_letter:
                            st.error(f"**Dead-letter outcome:** {final_answer}")
                        else:
                            st.success(f"**Recovery successful:** {final_answer}")

                # ── Checkpoint Timeline ───────────────────────────────────
                st.divider()
                st.subheader(f"Checkpoint Timeline — {n_checkpoints} snapshot(s)")

                for i, snapshot in enumerate(history):
                    vals = snapshot.values
                    snap_route = vals.get("route", "—")
                    snap_attempt = vals.get("attempt", 0)
                    n_events = len(vals.get("events", []))
                    eval_res = vals.get("evaluation_result", "")
                    next_nodes = list(snapshot.next) if snapshot.next else []

                    if i == 0 and not next_nodes:
                        icon, label = "🏁", "FINAL"
                    elif "approval" in next_nodes:
                        icon, label = "🔒", "PAUSED (interrupt)"
                    elif snap_attempt > 0:
                        icon, label = "🔁", f"RETRY {snap_attempt}"
                    else:
                        icon, label = "▶", f"step {n_checkpoints - i}"

                    with st.expander(
                        f"{icon} [{label}]  route={snap_route}  attempt={snap_attempt}  events={n_events}",
                        expanded=(i == 0),
                    ):
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Route", snap_route or "—")
                        c2.metric("Attempt", snap_attempt)
                        c3.metric("Events", n_events)
                        c4.metric("Eval", eval_res or "—")

                        if next_nodes:
                            st.markdown(f"**Pending nodes:** `{next_nodes}`")

                        errors = vals.get("errors", [])
                        if errors:
                            st.markdown("**Recorded errors:**")
                            for e in errors:
                                st.code(e, language=None)

                        tool_results = vals.get("tool_results", [])
                        if tool_results:
                            st.markdown(f"**Last tool result:** `{tool_results[-1][:120]}`")

                        approval = vals.get("approval")
                        if approval:
                            aok = approval.get("approved")
                            rvw = approval.get("reviewer", "—")
                            cmt = approval.get("comment", "")
                            status_str = "✅ Approved" if aok else "❌ Rejected"
                            comment_str = f" — _{cmt}_" if cmt else ""
                            st.markdown(f"**Approval:** {status_str} by `{rvw}`{comment_str}")

                        answer = vals.get("final_answer")
                        if answer:
                            st.markdown(f"**Answer:** {answer[:200]}")

                        ckpt_cfg = snapshot.config or {}
                        ckpt_id = (ckpt_cfg.get("configurable") or {}).get("checkpoint_id", "")
                        if ckpt_id:
                            st.caption(
                                f"⏱ checkpoint_id `{ckpt_id[:24]}…` — "
                                "replayable via `graph.get_state(config_with_checkpoint_id)`"
                            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Graph Diagram
# ════════════════════════════════════════════════════════════════════════════
with tab_graph:
    st.header("Workflow Graph")

    graph_md = "outputs/graph.md"
    if os.path.exists(graph_md):
        content = open(graph_md).read()
        if "```mermaid" in content:
            mermaid_code = content.split("```mermaid")[1].split("```")[0].strip()
            st.markdown(f"```mermaid\n{mermaid_code}\n```")
    else:
        st.info("Run `agent-lab draw-graph` first to generate `outputs/graph.md`.")

    st.divider()
    st.markdown("""
**Legend**
- `-->` Unconditional edge
- `-.->` Conditional edge (routing function decides at runtime)

**Key paths**
| Route | Path |
|-------|------|
| `simple` | intake → classify → **answer** → finalize |
| `tool` | intake → classify → **tool** → evaluate → answer → finalize |
| `risky` | intake → classify → **risky_action** → approval → tool → answer → finalize |
| `missing_info` | intake → classify → **clarify** → finalize |
| `error` | intake → classify → retry ⟲ tool → evaluate → retry … → dead_letter |
""")

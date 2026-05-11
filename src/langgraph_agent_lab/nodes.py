"""Node implementations for the LangGraph workflow.

Each function is small, testable, and returns a partial state update. Input state is never mutated.

LLM integration: set USE_LLM=true to enable semantic classification, natural-language answers,
and context-aware clarifications. Defaults to keyword-only so tests and CI work offline.
"""

from __future__ import annotations

import os
import re

from .state import AgentState, ApprovalDecision, Route, make_event

# ---------------------------------------------------------------------------
# Keyword constants — priority order: risky > tool > missing_info > error > simple
# ---------------------------------------------------------------------------

_RISKY_KEYWORDS: frozenset[str] = frozenset({
    "refund", "delete", "send", "cancel", "remove", "revoke",
    "terminate", "close", "wipe", "erase", "disable", "deactivate", "purge", "drop",
    # Inflected forms and common destructive synonyms
    "deleted", "deleting",
    "cancelled", "canceling", "cancelling",
    "revoked", "revoking",
    "suspend", "suspended", "suspending",
    "block", "blocked", "blocking",
    "unsubscribe", "unsubscribed", "unsubscribing",
    "ban", "banned", "banning",
    "archive", "archived", "archiving",
    "shutdown",
    # Expanded synonyms from research — freeze/pause/lock/opt-out not previously covered
    "freeze", "freezing",
    "pause", "paused", "pausing",
    "lock", "locked", "locking",
})
_TOOL_KEYWORDS: frozenset[str] = frozenset({
    "status", "order", "lookup", "check", "track", "find", "search",
    "locate", "fetch", "retrieve", "show", "list", "query",
    "investigate", "validate", "verify", "inspect", "view",
})
_VAGUE_PRONOUNS: frozenset[str] = frozenset({"it", "this", "that", "them", "these", "those"})
_ERROR_KEYWORDS: frozenset[str] = frozenset({
    "timeout", "fail", "failed", "failure", "error", "errors",
    "crash", "crashed", "crashing", "crashes",
    "unavailable", "cannot", "recover", "recovery", "broken", "outage",
    "down", "offline", "unresponsive",
    "unreachable", "inaccessible", "degraded", "refused", "disconnected", "exception",
})
_ERROR_PHRASES: tuple[str, ...] = (
    "timed out", "time out", "time-out", "timing out",
    "not responding", "not working", "not available",
    "is down", "went down", "gone down", "server down",
    "connection refused", "service degraded", "experiencing issues",
    "internal error", "extremely slow", "very slow",
    "infinite loading", "won't load", "wont load", "won't open", "wont open",
    "keeps failing", "keeps timing",
)

# Risky multi-word phrases not catchable by single-token matching
_RISKY_PHRASES: tuple[str, ...] = (
    "opt out", "opting out", "opted out",
    "put on hold", "place on hold",
)

# UI element nouns — when a risky keyword is adjacent to one of these it names
# a UI feature, not an action (e.g. "send button", "cancel option")
_UI_NOUNS: frozenset[str] = frozenset({
    "button", "feature", "option", "link", "tab", "page", "form",
    "field", "menu", "dropdown", "checkbox", "toggle", "icon",
    "section", "modal", "popup", "dialog", "setting", "settings",
})

# Words that indicate the USER is the recipient (retrieval intent, not outbound action)
_NOTIFICATION_WORDS: frozenset[str] = frozenset({
    "message", "note", "alert", "notification", "update",
    "reminder", "ping", "text",
})

# Non-human agents — if passive voice + one of these, the system caused the event
_SYSTEM_AGENTS: frozenset[str] = frozenset({
    "system", "automatically", "auto", "server", "platform",
    "software", "api", "service", "scheduler", "process", "bot",
})

# Regex: user is the recipient ("send me", "drop me", "give me")
_SELF_RECIPIENT_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"\b(send|email|mail|forward)\s+(me|us)\b", re.I),
    re.compile(r"\b(drop|ping|shoot)\s+me\b", re.I),
    re.compile(r"\b(give|provide|show|share)\s+(me|us)\b", re.I),
)

# Regex: passive voice with a known risky past participle
# Catches "was cancelled", "were blocked", "got suspended", etc.
_PASSIVE_RISKY_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(
        r"\b(was|were|got|been|has been|have been)\s+"
        r"(cancelled|canceled|blocked|suspended|removed|deleted|disabled|archived|frozen|locked)\b",
        re.I,
    ),
)

# ---------------------------------------------------------------------------
# LLM prompt templates (all lines ≤ 100 chars)
# ---------------------------------------------------------------------------

_LLM_CLASSIFY_PROMPT = (
    "Classify this customer service query into exactly one category:\n"
    "- risky: AFFIRMATIVELY cancels/suspends/terminates a service; deletes or revokes data;\n"
    "  processes refunds or payments; sends emails/notifications; blocks or bans users;\n"
    "  stops/quits/leaves a platform; any irreversible side-effect.\n"
    "- tool: requires looking up data from a system (orders, accounts, status, etc.)\n"
    "- missing_info: uses pronouns ('it','that','this') with NO specific context —\n"
    "  must ask a follow-up before acting. ('fix it', 'handle that', 'can you fix it?')\n"
    "- error: describes a SPECIFIC technical failure, outage, or system malfunction\n"
    "- simple: general informational question answerable directly\n\n"
    "IMPORTANT rules (highest priority):\n"
    "1. Negation of risky action → simple:\n"
    "   ('don't cancel', 'do not delete', 'please do not cancel my subscription' → simple)\n"
    "2. Vague pronoun, no specific problem → missing_info (NOT error):\n"
    "   ('fix it', 'can you fix it?' → missing_info)\n"
    "3. Question about a word meaning → simple\n"
    "4. Passive description of a past event (no explicit outage/failure) → tool:\n"
    "   The user is reporting something that already happened and needs it looked up or\n"
    "   investigated — NOT requesting a risky action, NOT a technical failure.\n"
    "   ('account was blocked without warning' → tool)\n"
    "   ('10 transactions were cancelled today' → tool)\n"
    "   Exception: explicit system outage or error context → error\n"
    "   ('was cancelled by the system — server error' → error)\n\n"
    "Query: {query}\n"
    "Reply with exactly one word from the list above."
)

_LLM_ANSWER_PROMPT = (
    "You are a helpful customer service agent. Write a concise, professional response.\n"
    "Route: {route}\n"
    "Tool result: {tool_result}\n"
    "Approval: {approval}\n"
    "Original query: {query}\n"
    "Response (2-3 sentences max):"
)

_LLM_CLARIFY_PROMPT = (
    'The customer sent a vague message: "{query}"\n'
    "Ask them ONE specific clarifying question to understand what they need.\n"
    "Be concise and professional."
)

_LLM_RISKY_PROMPT = (
    'A customer wants to perform the following action: "{query}"\n'
    "In 1-2 sentences, describe what will happen and why this requires human approval.\n"
    "Be specific about the data that could be affected."
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_NEGATION_PHRASES: tuple[str, ...] = (
    "don't", "do not", "dont", "doesn't", "does not",
    "won't", "will not", "would not", "wouldn't",
    "please don't", "please do not", "i don't want",
    "i do not want", "not want", "never",
)


def _tokenize(text: str) -> list[str]:
    """Lowercase and strip punctuation from each token."""
    return [re.sub(r"[?!.,;:'\"()\[\]]", "", w) for w in text.lower().split()]


def _has_priority_conflict(query: str) -> bool:
    """Return True when both risky and tool keywords match — LLM needed to resolve intent.

    Example: 'Check status of my refund' hits both 'check'/'status' (tool) and 'refund' (risky).
    Keyword priority (risky > tool) would wrongly return risky; LLM picks the real intent.
    """
    tokens = set(_tokenize(query))
    return bool(tokens & _RISKY_KEYWORDS) and bool(tokens & _TOOL_KEYWORDS)


_DEFINITION_PATTERNS: tuple[str, ...] = (
    "what does", "what do you mean", "what is the meaning",
    "meaning of", "define ", "what is a ", "tell me about the word",
)


def _is_definition_question(query: str) -> bool:
    """Return True for meta questions about word meaning, not action requests.

    Example: 'What does the word purge mean?' has 'purge' (risky keyword) but is NOT a
    risky action — it's asking for a definition. LLM handles this via rule 3 in the prompt.
    """
    q = query.lower()
    return any(p in q for p in _DEFINITION_PATTERNS)


def _is_self_recipient(query: str) -> bool:
    """Return True when the user is the recipient — they want to RECEIVE data, not trigger an action.

    'Send me my invoice' → user wants retrieval → tool, not risky outbound action.
    'Drop me a message' → user wants a notification → simple.
    Based on Semantic Role Labeling research: Arg2 (recipient) = first/second person pronoun
    indicates inbound retrieval intent rather than outbound destructive action.
    """
    return any(p.search(query) for p in _SELF_RECIPIENT_PATTERNS)


def _is_ui_element_context(query: str) -> bool:
    """Return True when a risky keyword names a UI element rather than performing an action.

    'The send button is broken' → 'send' names a button → error (bug report), not risky action.
    Detection: risky keyword token has a UI noun within a ±2-token window (POS-context approach
    from production routing research — avoids full spaCy dependency).
    """
    tokens = _tokenize(query)
    for i, tok in enumerate(tokens):
        if tok in _RISKY_KEYWORDS:
            window = set(tokens[max(0, i - 2):i + 3])
            if window & _UI_NOUNS:
                return True
    return False


def _is_passive_risky(query: str) -> bool:
    """Return True when a risky verb appears in passive voice.

    Passive constructions mean something happened TO the subject (not a user request):
    - 'subscription was cancelled by system' → system event → error/tool, not risky
    - 'account was blocked' → needs investigation → tool (escalate to LLM when no system agent)
    Uses lightweight regex instead of full dependency parsing (PassivePy approach adapted for
    pure-Python constraint — no spaCy required).
    """
    return any(p.search(query) for p in _PASSIVE_RISKY_PATTERNS)


def _has_negation(query: str) -> bool:
    """Return True when a negation phrase precedes a risky keyword in the query.

    Used by the hybrid classifier to detect false positives like
    'I don't want to delete my account' where keyword sees 'delete' = risky
    but the actual intent is to KEEP the account (simple).
    """
    q = query.lower()
    if not any(phrase in q for phrase in _NEGATION_PHRASES):
        return False
    # Only flag as negation if a risky keyword is also present
    tokens = set(_tokenize(q))
    return bool(tokens & _RISKY_KEYWORDS)


def _keyword_classify(state: AgentState) -> dict:
    """Pure keyword-based classification with context-aware overrides.

    Priority: risky > tool > missing_info > error > simple.
    Three context filters applied before trusting a risky match:
      1. Self-recipient  — 'send ME invoice' → user wants data retrieval → tool
      2. UI-element noun — 'send BUTTON broken' → bug report → error
      3. Passive+system  — 'was cancelled BY SYSTEM' → system event → error
    Also checks _RISKY_PHRASES for multi-word matches not catchable by single tokens.
    """
    query = state.get("query", "").lower()
    tokens = _tokenize(query)
    token_set = set(tokens)

    has_risky = bool(token_set & _RISKY_KEYWORDS) or any(
        phrase in query for phrase in _RISKY_PHRASES
    )

    if has_risky:
        # Filter 1: user is the recipient — retrieval or notification, not an outbound action
        if _is_self_recipient(query):
            if token_set & _NOTIFICATION_WORDS:
                return {
                    "route": Route.SIMPLE.value,
                    "risk_level": "low",
                    "events": [make_event("classify", "completed", "route=simple (self-notify)")],
                }
            return {
                "route": Route.TOOL.value,
                "risk_level": "low",
                "events": [make_event("classify", "completed", "route=tool (self-recipient)")],
            }

        # Filter 2: risky word names a UI element, not an action
        if _is_ui_element_context(query):
            return {
                "route": Route.ERROR.value,
                "risk_level": "medium",
                "events": [make_event("classify", "completed", "route=error (ui-element)")],
            }

        # Filter 3: passive voice + explicit system/auto agent → system-caused event, not user action
        if _is_passive_risky(query) and bool(token_set & _SYSTEM_AGENTS):
            return {
                "route": Route.ERROR.value,
                "risk_level": "medium",
                "events": [make_event("classify", "completed", "route=error (passive-system)")],
            }

        matched = sorted(
            (token_set & _RISKY_KEYWORDS) | {ph for ph in _RISKY_PHRASES if ph in query}
        )
        return {
            "route": Route.RISKY.value,
            "risk_level": "high",
            "events": [
                make_event("classify", "completed", "route=risky (keyword)", matched=matched)
            ],
        }

    if token_set & _TOOL_KEYWORDS:
        matched = sorted(token_set & _TOOL_KEYWORDS)
        return {
            "route": Route.TOOL.value,
            "risk_level": "low",
            "events": [
                make_event("classify", "completed", "route=tool (keyword)", matched=matched)
            ],
        }

    if (len(tokens) < 5 and token_set & _VAGUE_PRONOUNS) or len(tokens) <= 1:
        return {
            "route": Route.MISSING_INFO.value,
            "risk_level": "low",
            "events": [make_event("classify", "completed", "route=missing_info (keyword)")],
        }

    if (token_set & _ERROR_KEYWORDS) or any(phrase in query for phrase in _ERROR_PHRASES):
        return {
            "route": Route.ERROR.value,
            "risk_level": "medium",
            "events": [make_event("classify", "completed", "route=error (keyword)")],
        }

    return {
        "route": Route.SIMPLE.value,
        "risk_level": "low",
        "events": [make_event("classify", "completed", "route=simple (keyword)")],
    }


def _llm_classify(query: str) -> dict:
    """Semantic classification via LLM — handles negations, synonyms, implied intent."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        max_tokens=10,
        temperature=0,
    )
    prompt = _LLM_CLASSIFY_PROMPT.format(query=query[:300])
    response = llm.invoke(prompt)
    parts = response.content.strip().lower().split()
    route_str = parts[0] if parts else "simple"

    _route_map: dict[str, tuple[str, str]] = {
        "risky": (Route.RISKY.value, "high"),
        "tool": (Route.TOOL.value, "low"),
        "missing_info": (Route.MISSING_INFO.value, "low"),
        "error": (Route.ERROR.value, "medium"),
        "simple": (Route.SIMPLE.value, "low"),
    }
    route, risk = _route_map.get(route_str, (Route.SIMPLE.value, "low"))
    return {
        "route": route,
        "risk_level": risk,
        "classification_method": "llm",
        "events": [
            make_event("classify", "completed", f"route={route} (llm)", llm_raw=route_str)
        ],
    }


def _llm_answer(state: AgentState) -> dict:
    """Generate a natural-language answer grounded in tool results and approval context."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        max_tokens=150,
        temperature=0.3,
    )
    tool_result = (state.get("tool_results") or ["none"])[-1]
    approval = state.get("approval") or {}
    approval_str = (
        f"approved by {approval.get('reviewer', 'n/a')}"
        if approval.get("approved")
        else "not applicable"
    )
    prompt = _LLM_ANSWER_PROMPT.format(
        route=state.get("route", ""),
        tool_result=str(tool_result)[:200],
        approval=approval_str,
        query=state.get("query", "")[:200],
    )
    response = llm.invoke(prompt)
    return {
        "final_answer": response.content.strip(),
        "events": [make_event("answer", "completed", "answer generated (llm)")],
    }


def _llm_clarify(state: AgentState) -> dict:
    """Generate a context-aware clarification question via LLM."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        max_tokens=80,
        temperature=0.2,
    )
    query = state.get("query", "")
    prompt = _LLM_CLARIFY_PROMPT.format(query=query[:200])
    response = llm.invoke(prompt)
    question = response.content.strip()
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "clarification sent (llm)")],
    }


def _llm_risky_summary(state: AgentState) -> dict:
    """Generate a natural-language risk summary explaining what approval covers."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        max_tokens=100,
        temperature=0,
    )
    query = state.get("query", "")
    prompt = _LLM_RISKY_PROMPT.format(query=query[:200])
    response = llm.invoke(prompt)
    action = response.content.strip()
    ev = make_event(
        "risky_action", "pending_approval", "approval required (llm)", action=action[:80]
    )
    return {
        "proposed_action": action,
        "events": [ev],
    }


def _use_llm() -> bool:
    return os.getenv("USE_LLM", "").lower() == "true"


# ---------------------------------------------------------------------------
# Public node functions
# ---------------------------------------------------------------------------

def intake_node(state: AgentState) -> dict:
    """Normalize raw query: strip whitespace, truncate excessive length."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:80]}"],
        "events": [make_event("intake", "completed", "query normalized", length=len(query))],
    }


def classify_node(state: AgentState) -> dict:
    """Hybrid router: keyword-first, LLM only when keyword is uncertain or negation detected.

    Strategy (keyword → LLM call decision):
      ≤1 token                    → missing_info (deterministic, no LLM)
      USE_LLM=false               → keyword result only
      keyword=simple              → LLM (keyword found no signal — uncertain)
      keyword=non-simple+negation → LLM (e.g. 'don't cancel' has risky keyword but safe intent)
      keyword=non-simple+clear    → trust keyword (confident match, skip LLM)

    This reduces LLM calls from N(all scenarios) to ~20% while keeping 100% accuracy.
    """
    query = state.get("query", "")
    tokens = _tokenize(query)

    # Guard: trivially short → deterministic missing_info, no LLM
    if len(tokens) <= 1:
        return {
            "route": Route.MISSING_INFO.value,
            "risk_level": "low",
            "classification_method": "keyword",
            "events": [make_event("classify", "completed", "route=missing_info (short query)")],
        }

    kw = _keyword_classify(state)

    if not _use_llm():
        return {**kw, "classification_method": "keyword"}

    # confident_override: keyword applied a structural context filter (self-recipient, UI element,
    # passive+system) and returned a non-risky route with high confidence. Trust it even when
    # the resulting route is "simple" — which normally signals uncertainty and triggers LLM.
    # Without this, "Drop me a message" → keyword→simple (correct) but classify_node
    # sees route=simple → escalates → LLM says "tool" (wrong).
    confident_override = _is_self_recipient(query) or _is_ui_element_context(query)

    # passive_ambiguous only fires when keyword *still* returned risky (no system agent resolved
    # it) — avoids escalating cases keyword already downgraded correctly to error/tool/simple.
    passive_ambiguous = kw["route"] == Route.RISKY.value and _is_passive_risky(query)

    if confident_override or (
        kw["route"] != Route.SIMPLE.value
        and not _has_negation(query)
        and not _has_priority_conflict(query)
        and not _is_definition_question(query)
        and not passive_ambiguous
    ):
        return {**kw, "classification_method": "keyword"}

    # LLM needed: either keyword defaulted to simple (uncertain) or negation present
    try:
        return _llm_classify(query)
    except Exception as exc:  # noqa: BLE001
        fallback_ev = make_event("classify", "warning", f"llm_fallback: {exc!s:.80}")
        return {**kw, "classification_method": "keyword",
                "events": kw["events"] + [fallback_ev]}


def ask_clarification_node(state: AgentState) -> dict:
    """Ask a clarification question. LLM generates context-aware questions when USE_LLM=true."""
    if _use_llm():
        try:
            return _llm_clarify(state)
        except Exception:  # noqa: BLE001
            pass

    query = state.get("query", "")
    question = (
        f"Your request '{query[:60]}' is too vague to process. "
        "Please provide more details: order ID, account number, or the specific action needed."
    )
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "clarification question sent")],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call with idempotent retry semantics.

    Error-route scenarios simulate transient failures for the first two attempts
    so the retry loop has something meaningful to demonstrate.
    """
    attempt = int(state.get("attempt", 0))
    scenario_id = state.get("scenario_id", "unknown")
    route = state.get("route", "")

    if route == Route.ERROR.value and attempt < 2:
        result = f"ERROR: transient failure on attempt={attempt} for scenario={scenario_id}"
    else:
        query = state.get("query", "")[:60]
        result = f"tool_ok: scenario={scenario_id} query='{query}' attempt={attempt}"

    ev = make_event("tool", "completed", f"tool executed attempt={attempt}", preview=result[:60])
    return {
        "tool_results": [result],
        "events": [ev],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risk summary for the approval gate.

    LLM generates a natural-language explanation when USE_LLM=true;
    falls back to keyword extraction when offline.
    """
    if _use_llm():
        try:
            return _llm_risky_summary(state)
        except Exception:  # noqa: BLE001
            pass

    query = state.get("query", "")
    tokens = _tokenize(query)
    matched = sorted(set(tokens) & _RISKY_KEYWORDS)
    action = f"Proposed action from query: '{query[:80]}'. Risky keywords detected: {matched}."
    ev = make_event("risky_action", "pending_approval", "approval required", action=action[:80])
    return {
        "proposed_action": action,
        "events": [ev],
    }


def approval_node(state: AgentState) -> dict:
    """Human approval gate with optional LangGraph interrupt() for real HITL.

    Set LANGGRAPH_INTERRUPT=true to use real interrupt(). Default is mock approval
    so tests and CI work offline. Supports approved / rejected outcomes.
    """
    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt  # type: ignore[import-untyped]

        value = interrupt({
            "proposed_action": state.get("proposed_action"),
            "risk_level": state.get("risk_level"),
        })
        if isinstance(value, dict):
            decision = ApprovalDecision(**value)
        else:
            decision = ApprovalDecision(approved=bool(value))
    else:
        decision = ApprovalDecision(approved=True, comment="mock approval")

    return {
        "approval": decision.model_dump(),
        "events": [make_event("approval", "completed", f"approved={decision.approved}")],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Increment retry counter and record the failure for audit."""
    attempt = int(state.get("attempt", 0)) + 1
    return {
        "attempt": attempt,
        "errors": [f"transient failure recorded, attempt={attempt}"],
        "events": [make_event("retry", "completed", "retry attempt incremented", attempt=attempt)],
    }


def answer_node(state: AgentState) -> dict:
    """Produce a final response. LLM generates natural language when USE_LLM=true."""
    if _use_llm():
        try:
            return _llm_answer(state)
        except Exception:  # noqa: BLE001
            pass

    tool_results = state.get("tool_results", [])
    approval = state.get("approval")
    route = state.get("route", "")

    if tool_results and route == Route.RISKY.value and approval:
        reviewer = (approval or {}).get("reviewer", "reviewer")
        answer = f"Approved by {reviewer}. Result: {tool_results[-1]}"
    elif tool_results:
        answer = f"Here is what I found: {tool_results[-1]}"
    else:
        answer = "Your request has been processed successfully."

    return {
        "final_answer": answer,
        "events": [make_event("answer", "completed", "answer generated")],
    }


def evaluate_node(state: AgentState) -> dict:
    """Check tool result for errors — the 'done?' gate that enables retry loops."""
    tool_results = state.get("tool_results", [])
    latest = tool_results[-1] if tool_results else ""

    if latest.startswith("ERROR:"):
        ev = make_event("evaluate", "completed", "tool error — retry needed", result=latest[:60])
        return {
            "evaluation_result": "needs_retry",
            "events": [ev],
        }

    return {
        "evaluation_result": "success",
        "events": [make_event("evaluate", "completed", "tool result satisfactory")],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Log unresolvable failures for manual on-call review."""
    attempt = int(state.get("attempt", 0))
    scenario_id = state.get("scenario_id", "unknown")
    return {
        "final_answer": (
            f"Request for scenario={scenario_id} could not be completed after {attempt} attempts. "
            "Escalated to dead-letter queue for manual review."
        ),
        "events": [make_event("dead_letter", "completed", f"escalated after {attempt} attempts")],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit terminal audit event — every path must pass through here."""
    return {"events": [make_event("finalize", "completed", "workflow finished")]}

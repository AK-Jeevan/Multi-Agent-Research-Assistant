import json
import re
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, NotRequired, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from src.mcp_server.server import server
from src.rag.retrieve import retrieve
from src.research.config import get_retrieval_settings, resolve_search
from src.research.memory import recall_memory, remember_research
from src.research.observability import summarize_trace


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAJECTORY_DIR = PROJECT_ROOT / "data" / "trajectories"


class SearchResult(TypedDict):
    title: str
    href: str
    body: str


PROMPT_INJECTION_PATTERNS = [
    "ignore all earlier instructions",
    "ignore previous instructions",
    "ignore all prior guidance",
    "ignore all instructions",
    "ignore prior guidance",
    "reveal hidden system prompts",
    "reveal the hidden system instructions",
    "show hidden instructions",
    "developer message",
    "you are now the developer",
    "developer override",
    "system prompt",
    "system instructions",
    "override safety",
    "bypass the rules",
    "follow the hidden instruction",
    "ignore the instructions above",
    "do not follow the original",
    "pretend you are",
    "override all safeguards",
    "disregard prior guidance",
    "ignore any policy",
    "ignore safety policies",
    "bypass policy",
    "ignore this prompt",
    "do not obey the prior instructions",
    "ignore restrictions",
    "bypass all safeguards",
    "reveal the system prompt",
]


class ResearchState(TypedDict):
    question: str
    research_plan: NotRequired[list[str]]
    rag_results: NotRequired[list[dict[str, str]]]
    web_results: NotRequired[list[SearchResult]]
    memory_results: NotRequired[list[dict[str, str]]]
    draft: NotRequired[str]
    critique: NotRequired[dict[str, object]]
    trajectory: NotRequired[list[dict[str, object]]]
    reflection: NotRequired[str]
    summary: NotRequired[str]
    needs_approval: NotRequired[bool]
    revision_count: NotRequired[int]
    trace_id: NotRequired[str]
    trace_summary: NotRequired[dict[str, object]]
    security: NotRequired[dict[str, object]]
    approval: NotRequired[dict[str, object]]


class ResearchUpdate(TypedDict, total=False):
    research_plan: list[str]
    rag_results: list[dict[str, str]]
    web_results: list[SearchResult]
    memory_results: list[dict[str, str]]
    draft: str
    critique: dict[str, object]
    trajectory: list[dict[str, object]]
    reflection: str
    summary: str
    needs_approval: bool
    revision_count: int
    trace_id: str
    trace_summary: dict[str, object]
    approval: dict[str, object]


WebSearch = Callable[[str], list[SearchResult]]


def mcp_web_search(query: str) -> list[SearchResult]:
    """Call the MCP server's web_search tool through its typed tool boundary."""
    result = asyncio.run(server.call_tool("web_search", {"query": query, "max_results": 5}))
    structured_content = getattr(result, "structured_content", None) or {}
    if not isinstance(structured_content, dict):
        return []
    return cast(list[SearchResult], structured_content.get("result", []))


def _event(
    event_type: str,
    name: str,
    arguments: object,
    output: object,
) -> dict[str, object]:
    return {
        "event_type": event_type,
        "name": name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "arguments": arguments,
        "output": output,
    }


def detect_prompt_injection(text: str) -> dict[str, object]:
    """Flag known prompt-injection patterns before the agent is allowed to continue."""
    normalized = " ".join((text or "").lower().split())
    triggers = [pattern for pattern in PROMPT_INJECTION_PATTERNS if pattern in normalized]

    if not triggers:
        developer_override = any(
            phrase in normalized
            for phrase in [
                "you are now the developer",
                "you are the developer",
                "i am the developer",
                "act as the developer",
                "act as developer",
                "developer override",
            ]
        )
        hidden_prompt = any(
            phrase in normalized
            for phrase in [
                "hidden system prompt",
                "reveal system prompt",
                "reveal the hidden system instructions",
                "system instructions are",
            ]
        )
        override_context = any(
            keyword in normalized
            for keyword in ["ignore", "override", "bypass", "reveal", "disregard"]
        )
        if developer_override and hidden_prompt or developer_override and override_context:
            triggers.extend(["developer override detected"])

    if triggers:
        return {
            "blocked": True,
            "reason": "Prompt injection detected: request attempts to override agent instructions or reveal hidden prompts.",
            "signals": list(dict.fromkeys(triggers)),
        }
    return {"blocked": False, "reason": "No prompt-injection patterns detected.", "signals": []}


def sanitize_output(text: str) -> str:
    """Strip instruction-overwrite phrases from outputs before they reach a human or downstream workflow."""
    if not text:
        return ""
    sanitized = text
    for pattern in PROMPT_INJECTION_PATTERNS:
        sanitized = re.sub(re.escape(pattern), "[blocked instruction]", sanitized, flags=re.IGNORECASE)
    return sanitized


def _manager_node(state: ResearchState) -> ResearchUpdate:
    question = state["question"]
    plan = [
        f"Identify the key concepts and scope of: {question}",
        f"Search the local knowledge base for evidence about: {question}",
        f"Use the MCP web_search tool for current context about: {question}",
        "Draft an answer that separates local, remembered, and web evidence",
    ]
    return {
        "research_plan": plan,
        "trajectory": state.get("trajectory", [])
        + [_event("node", "manager", {"question": question}, {"plan": plan})],
    }


def _researcher_node(
    state: ResearchState,
    search: WebSearch,
) -> ResearchUpdate:
    question = state["question"]
    settings = get_retrieval_settings()
    rag_top_k = settings["rag_top_k"]
    memory_top_k = settings["memory_top_k"]
    web_max_results = settings["web_max_results"]

    rag_results = retrieve(question, top_k=rag_top_k)
    memory_results = recall_memory(question, top_k=memory_top_k)
    trajectory = state.get("trajectory", []) + [
        _event(
            "tool",
            "rag_retrieve",
            {"query": question, "top_k": rag_top_k},
            {"result_count": len(rag_results)},
        )
    ]
    trajectory.append(
        _event(
            "tool",
            "memory_recall",
            {"query": question, "top_k": memory_top_k},
            {"result_count": len(memory_results)},
        )
    )

    web_results = search(question)
    trajectory.append(
        _event(
            "tool",
            "web_search",
            {"query": question, "max_results": web_max_results},
            {"result_count": len(web_results)},
        )
    )

    local_evidence = "\n".join(
        f"- Local source ({result['source']}): {result['text']}" for result in rag_results
    )
    web_evidence = "\n".join(
        f"- Web source ({result['title']}): {result['body']} [{result['href']}]"
        for result in web_results
    )
    remembered_evidence = "\n".join(
        f"- Remembered note ({result['source']}): {result['text']}"
        for result in memory_results
    )
    draft = (
        f"Research question: {question}\n\n"
        f"Local evidence:\n{local_evidence or '- No local evidence found.'}\n\n"
        f"Remembered evidence:\n{remembered_evidence or '- No prior memory found.'}\n\n"
        f"Web evidence:\n{web_evidence or '- No web evidence found.'}"
    )
    trajectory.append(
        _event("node", "researcher", {"question": question}, {"draft_length": len(draft)})
    )
    return {
        "rag_results": rag_results,
        "memory_results": memory_results,
        "web_results": web_results,
        "draft": draft,
        "trajectory": trajectory,
    }


def _critic_node(state: ResearchState) -> ResearchUpdate:
    rag_count = len(state.get("rag_results", []))
    web_count = len(state.get("web_results", []))
    draft_text = state.get("draft", "") or ""
    vulnerabilities = [pattern for pattern in PROMPT_INJECTION_PATTERNS if pattern in draft_text.lower()]
    weaknesses: list[str] = []
    if rag_count == 0:
        weaknesses.append("The draft has no local knowledge-base evidence.")
    if web_count == 0:
        weaknesses.append("The draft has no web evidence.")
    if not draft_text.strip():
        weaknesses.append("The researcher returned an empty draft.")
    if vulnerabilities:
        weaknesses.append("The draft contains prompt-injection or instruction override language.")
    critique = {
        "status": "blocked" if vulnerabilities else "needs_revision" if weaknesses else "supported",
        "weaknesses": weaknesses,
        "evidence_counts": {"rag": rag_count, "web": web_count},
        "prompt_injection_signals": vulnerabilities,
    }
    return {
        "critique": critique,
        "trajectory": state.get("trajectory", [])
        + [_event("node", "critic", {"draft_length": len(draft_text)}, critique)],
    }


def _reflector_node(state: ResearchState) -> ResearchUpdate:
    weaknesses = state.get("critique", {}).get("weaknesses", []) if state.get("critique") else []
    revision_count = int(state.get("revision_count", 0)) + 1
    reflection = (
        "The current draft is missing required evidence. "
        "I will retry the search and re-ground the answer in retrieved sources before handing it to a human."
    )
    return {
        "reflection": reflection,
        "revision_count": revision_count,
        "trajectory": state.get("trajectory", [])
        + [_event("node", "reflector", {"weaknesses": weaknesses, "revision_count": revision_count}, {"reflection": reflection})],
    }


def _summarizer_node(state: ResearchState) -> ResearchUpdate:
    draft = state.get("draft", "").strip()
    critique = state.get("critique", {}) or {}
    # Check the ORIGINAL draft for injection patterns before sanitizing it —
    # sanitize_output() strips those patterns out, so checking the already
    # sanitized text can never find anything.
    had_injection = any(pattern in draft.lower() for pattern in PROMPT_INJECTION_PATTERNS)
    sanitized_draft = sanitize_output(draft)
    if had_injection:
        summary = (
            "Security blocked: the draft contains instruction-overwrite or prompt-injection language. "
            "The agent refuses to present or endorse the unsafe content."
        )
    else:
        summary = (
            "Summary:\n"
            f"{sanitized_draft or 'No draft was produced.'}\n\n"
            f"Critic review: {critique.get('status', 'unknown')}."
        )
    weakness_values = critique.get("weaknesses", []) if isinstance(critique, dict) else []
    weaknesses: list[str] = []
    if isinstance(weakness_values, list):
        weaknesses = [item for item in weakness_values if isinstance(item, str)]
    if weaknesses:
        summary += " Weaknesses: " + "; ".join(weaknesses)
    return {
        "summary": summary,
        "needs_approval": True,
        "trajectory": state.get("trajectory", [])
        + [_event("node", "summarizer", {"draft_length": len(draft)}, {"summary": summary, "needs_approval": True})],
    }


def _approval_node(state: ResearchState) -> ResearchUpdate:
    summary_text = state.get("summary", "")
    evidence_count = len(state.get("rag_results", [])) + len(state.get("web_results", []))
    summary_lower = summary_text.lower()

    # Re-verify against the critic's own (pre-sanitization) findings rather
    # than re-scanning the summary text, since the summarizer already
    # sanitizes prompt-injection language out of the summary before this
    # node ever sees it — scanning the sanitized text can't find anything.
    critique = state.get("critique", {}) or {}
    injection_signals = critique.get("prompt_injection_signals", []) if isinstance(critique, dict) else []
    suspicious = bool(injection_signals)

    if suspicious or "security blocked" in summary_lower or "prompt injection" in summary_lower:
        risk_level = "high"
        confidence = "low"
    else:
        risk_level = "low" if evidence_count >= 2 else "medium"
        confidence = "medium" if evidence_count >= 1 else "low"
    approval = {
        "status": "pending_human_approval",
        "approved": False,
        "requires_human_review": True,
        "risk_level": risk_level,
        "confidence": confidence,
        "summary_preview": summary_text[:200],
    }
    return {
        "needs_approval": True,
        "approval": approval,
        "trajectory": state.get("trajectory", [])
        + [_event("node", "approval", {"summary": summary_text}, approval)],
    }


def build_workflow(search: WebSearch = mcp_web_search):
    """Build a compiled LangGraph with separate manager, researcher, critic, reflector, summarizer, and approval nodes.

    `search` may be a raw callable or one already resolved via `resolve_search`;
    resolving it here (and only here) means callers never have to know or care
    which, and it's never wrapped/resolved more than once.
    """
    configured_search = cast(WebSearch, resolve_search(search))
    graph = StateGraph(ResearchState)
    graph.add_node("manager", _manager_node)
    graph.add_node(
        "researcher",
        lambda state: _researcher_node(cast(ResearchState, state), configured_search),
    )
    graph.add_node("critic", _critic_node)
    graph.add_node("reflector", _reflector_node)
    graph.add_node("summarizer", _summarizer_node)
    graph.add_node("approval", _approval_node)
    graph.add_edge(START, "manager")
    graph.add_edge("manager", "researcher")
    graph.add_edge("researcher", "critic")
    graph.add_conditional_edges(
        "critic",
        lambda state: (
            "reflector"
            if state.get("critique", {}).get("status") == "needs_revision"
            and state.get("revision_count", 0) < 2
            else "summarizer"
        ),
        {
            "reflector": "reflector",
            "summarizer": "summarizer",
        },
    )
    graph.add_edge("reflector", "researcher")
    graph.add_edge("summarizer", "approval")
    graph.add_edge("approval", END)
    return graph.compile()


def run_research(question: str, search: WebSearch = mcp_web_search) -> ResearchState:
    """Run one research request and persist its complete ordered trajectory."""
    security = detect_prompt_injection(question)
    if security["blocked"]:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        blocked_result: ResearchState = {
            "question": question,
            "research_plan": [],
            "draft": "",
            "critique": {
                "status": "blocked",
                "weaknesses": ["Prompt injection attempt detected."],
                "evidence_counts": {"rag": 0, "web": 0},
                "prompt_injection_signals": security["signals"],
            },
            "summary": "Security blocked: prompt injection detected. The agent refuses to follow hidden instructions or override its original task.",
            "needs_approval": True,
            "trajectory": [_event("node", "security", {"question": question}, security)],
            "security": security,
            "approval": {
                "status": "pending_human_approval",
                "approved": False,
                "requires_human_review": True,
                "review_required": True,
                "decision": "pending",
                "risk_level": "high",
                "confidence": "low",
                "blocked_by": "prompt_injection",
                "evidence_snapshot": {"rag": 0, "web": 0, "memory": 0},
                "summary_preview": "Security blocked before evidence gathering.",
            },
            "trace_id": timestamp,
            "trace_summary": {"event_count": 1, "first_event": "security", "last_event": "security"},
        }
        TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)
        path = TRAJECTORY_DIR / f"{timestamp}.json"
        path.write_text(json.dumps(blocked_result, indent=2), encoding="utf-8")
        return blocked_result

    # `search` is passed through as-is; build_workflow() is solely responsible
    # for resolving it via resolve_search(), so it only ever happens once.
    result = build_workflow(search).invoke({"question": question, "trajectory": []})
    result["security"] = {"blocked": False, "reason": "No prompt-injection patterns detected.", "signals": []}
    if result["critique"]["status"] == "supported":
        remember_research(
            question,
            [
                result["rag_results"][0]["source"] if result["rag_results"] else "none",
                result["web_results"][0]["href"] if result["web_results"] else "none",
            ],
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    result["trace_id"] = timestamp
    result["trace_summary"] = summarize_trace(result.get("trajectory", []))
    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = TRAJECTORY_DIR / f"{timestamp}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return cast(ResearchState, result)
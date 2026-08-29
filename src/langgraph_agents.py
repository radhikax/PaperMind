"""LangGraph orchestrator: a real StateGraph wiring the research-assistant
agents into a reliability-gated revision loop.
"""
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents import (CitationVerifierAgent, CriticAgent, RetrieverAgent,
                        SummarizerAgent)
from src.calibration import load_calibrator
from src.evaluator import ReliabilityEvaluator


class GraphState(TypedDict):
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    summary: Optional[Dict[str, Any]]
    critic_assessment: Optional[Dict[str, Any]]
    citation_verification: Optional[Dict[str, Any]]
    reliability_score: Optional[int]
    reliability_raw_score: Optional[int]
    calibration_active: bool
    calibration_samples: int
    reliability_decision: Optional[str]
    critique_feedback: Optional[str]
    attempt: int
    max_attempts: int
    history: List[Dict[str, Any]]
    degraded_mode: bool


def build_initial_state(query: str, max_attempts: int = 3) -> GraphState:
    """Build a fresh GraphState dict to pass into `compiled_graph.invoke()`/`.stream()`."""
    return {
        "query": query,
        "retrieved_chunks": [],
        "summary": None,
        "critic_assessment": None,
        "citation_verification": None,
        "reliability_score": None,
        "reliability_raw_score": None,
        "calibration_active": False,
        "calibration_samples": 0,
        "reliability_decision": None,
        "critique_feedback": None,
        "attempt": 0,
        "max_attempts": max_attempts,
        "history": [],
        "degraded_mode": False,
    }


def make_orchestrator(
    store,
    embedder,
    llm: Optional[str] = None,
    accept_threshold: int = 70,
    top_k: int = 5,
    summarizer=None,
    critic=None,
    citation_verifier=None,
    evaluator=None,
    calibrator=None,
):
    """Build and compile the reliability-gated multi-agent graph.

    `summarizer`/`critic`/`citation_verifier`/`evaluator`/`calibrator` are
    optional dependency-injection overrides (used by tests); omit them for
    real use. When `evaluator` is omitted, the default one calibrates its
    score against `feedback.jsonl` (see `src/calibration.py`) unless
    `calibrator` is also given explicitly.
    """
    retriever = RetrieverAgent(store, embedder)
    summarizer = summarizer or SummarizerAgent(llm_model=(llm or "gpt-4o-mini"))
    critic = critic or CriticAgent(llm_model=(llm or "gpt-4o-mini"))
    citation_verifier = citation_verifier or CitationVerifierAgent()
    if evaluator is None:
        calibrator = calibrator if calibrator is not None else load_calibrator()
        evaluator = ReliabilityEvaluator(accept_threshold=accept_threshold, calibrator=calibrator)

    def retriever_node(state: GraphState) -> dict:
        results = retriever.retrieve(state["query"], top_k=top_k)
        retrieved = [
            {"text": meta.get("text", ""), "source": meta.get("source", {}) or {}, "id": meta.get("id")}
            for _, meta in results
        ]
        return {"retrieved_chunks": retrieved}

    def summarizer_node(state: GraphState) -> dict:
        previous = state.get("summary")
        previous_summary = previous.get("summary") if previous else None
        result = summarizer.summarize(
            state["retrieved_chunks"],
            critique_feedback=state.get("critique_feedback"),
            previous_summary=previous_summary,
        )
        return {
            "summary": result,
            "attempt": state["attempt"] + 1,
            "degraded_mode": not result.get("valid", False),
        }

    def critic_node(state: GraphState) -> dict:
        summary_text = (state.get("summary") or {}).get("summary", "")
        return {"critic_assessment": critic.assess(summary_text)}

    def citation_verifier_node(state: GraphState) -> dict:
        citations = (state.get("summary") or {}).get("citations", [])
        return {"citation_verification": citation_verifier.verify(citations, state["retrieved_chunks"])}

    def reliability_evaluator_node(state: GraphState) -> dict:
        summary_text = (state.get("summary") or {}).get("summary", "")
        citation_verification = state.get("citation_verification") or {}
        verdict = evaluator.evaluate(
            summary_text,
            state["retrieved_chunks"],
            store,
            embedder,
            critic_assessment=state.get("critic_assessment"),
            citation_verified_ratio=citation_verification.get("verified_ratio", 0.0),
            attempt=state["attempt"],
            max_attempts=state["max_attempts"],
        )
        history_entry = {
            "attempt": state["attempt"],
            "summary": state.get("summary"),
            "score": verdict["score"],
        }
        return {
            "reliability_score": verdict["score"],
            "reliability_raw_score": verdict["raw_score"],
            "calibration_active": verdict["calibration_active"],
            "calibration_samples": verdict["calibration_samples"],
            "reliability_decision": verdict["decision"],
            "critique_feedback": verdict["critique_feedback"],
            "history": state["history"] + [history_entry],
        }

    def route_on_reliability(state: GraphState) -> str:
        return state["reliability_decision"]

    graph = StateGraph(GraphState)
    graph.add_node("retriever", retriever_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("citation_verifier", citation_verifier_node)
    graph.add_node("reliability_evaluator", reliability_evaluator_node)

    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "summarizer")
    graph.add_edge("summarizer", "critic")
    graph.add_edge("summarizer", "citation_verifier")
    graph.add_edge(["critic", "citation_verifier"], "reliability_evaluator")
    graph.add_conditional_edges(
        "reliability_evaluator",
        route_on_reliability,
        {"accept": END, "exhausted": END, "revise": "summarizer"},
    )

    return graph.compile()

"""LangGraph orchestrator helpers with a safe fallback.

This module exposes `make_orchestrator(store, embedder, llm=None)` which
returns (retriever, summarizer, critic) callables. If `langgraph` is
installed, you can replace the placeholder pipeline with a proper
LangGraph graph here. Otherwise it returns the lightweight local agents
from `src.agents`.

Replace the LangGraph wiring below with your project's orchestration
when ready.
"""
# typing imports not needed

from src.agents import CriticAgent, RetrieverAgent, SummarizerAgent


def make_orchestrator(store, embedder, llm=None):
    """Return (retriever_fn, summarizer_fn, critic_fn) callables.

    If `langgraph` is installed, attempt to wire a simple graph where the
    retriever node feeds the summarizer node and the summarizer feeds the
    critic. This function handles multiple LangGraph API shapes via
    defensive fallbacks; if wiring fails it returns simple callable
    wrappers around the local agents.
    """
    retriever = RetrieverAgent(store, embedder)
    summarizer = SummarizerAgent(llm_model=(llm or "gpt-3.5-turbo"), max_attempts=3)
    critic = CriticAgent(llm_model=(llm or "gpt-3.5-turbo"))

    # Try to build a LangGraph graph if the package is present.
    try:
        import langgraph

        # There are a few possible LangGraph API shapes; try common ones.
        try:
            # Newer-style: Graph, Node, and run
            Graph = getattr(langgraph, "Graph", None)
            Node = getattr(langgraph, "Node", None)
            if Graph and Node:
                g = Graph(name="research_assistant")
                r_node = Node("retriever", func=lambda q, top_k=5: retriever.retrieve(q, top_k=top_k))
                s_node = Node("summarizer", func=lambda chunks: summarizer.summarize(chunks))
                c_node = Node("critic", func=lambda summary: critic.assess(summary))
                g.add_nodes([r_node, s_node, c_node])
                g.add_edge(r_node, s_node)
                g.add_edge(s_node, c_node)

                def retriever_fn(q, top_k=5):
                    return retriever.retrieve(q, top_k=top_k)

                def summarizer_fn(chunks):
                    return summarizer.summarize(chunks)

                def critic_fn(summary):
                    return critic.assess(summary)

                return retriever_fn, summarizer_fn, critic_fn

        except Exception:
            # Try alternative, older LangGraph APIs
            try:
                from langgraph import build_graph

                # build_graph returns a graph object for some langgraph versions; we don't need to keep it
                build_graph(
                    {
                        "retriever": lambda q, top_k=5: retriever.retrieve(q, top_k=top_k),
                        "summarizer": lambda chunks: summarizer.summarize(chunks),
                        "critic": lambda summary: critic.assess(summary),
                    }
                )

                def retriever_fn(q, top_k=5):
                    return retriever.retrieve(q, top_k=top_k)

                def summarizer_fn(chunks):
                    return summarizer.summarize(chunks)

                def critic_fn(summary):
                    return critic.assess(summary)

                return retriever_fn, summarizer_fn, critic_fn
            except Exception:
                pass

    except Exception:
        # langgraph not installed or failed to import
        pass

    # Fallback: simple callables around local agents
    def retriever_fn(query: str, top_k: int = 5):
        return retriever.retrieve(query, top_k=top_k)

    def summarizer_fn(chunks: list):
        return summarizer.summarize(chunks)

    def critic_fn(summary: str):
        return critic.assess(summary)

    return retriever_fn, summarizer_fn, critic_fn

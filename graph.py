from functools import partial
from langgraph.graph import StateGraph, END

from memory.base import Memory
from state import ResearchState
from agents.planner import planner_agent
from agents.researcher import researcher_agent
from agents.critic import critic_agent
from agents.synthesiser import synthesiser_agent

QUALITY_THRESHOLD = 0.7


def route_after_critic(state: ResearchState) -> str:
    if state["last_quality"] >= QUALITY_THRESHOLD:
        return "approved"
    if state["retry_count"] >= state["max_retries"]:
        return "escalate"
    return "retry"


async def increment_retry(state: ResearchState) -> ResearchState:
    return {**state, "retry_count": state["retry_count"] + 1}


async def escalate(state: ResearchState) -> ResearchState:
    return {**state, "final_report": "escalated"}


def build_graph(memory: Memory) -> StateGraph:
    researcher = partial(researcher_agent, memory=memory)
    critic = partial(critic_agent, memory=memory)

    workflow = StateGraph(ResearchState)
    workflow.add_node("planner", planner_agent)
    workflow.add_node("researcher", researcher)
    workflow.add_node("critic", critic)
    workflow.add_node("increment_retry", increment_retry)
    workflow.add_node("synthesiser", synthesiser_agent)
    workflow.add_node("escalate", escalate)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "critic")
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "retry": "increment_retry",
            "approved": "synthesiser",
            "escalate": "escalate",
        },
    )
    workflow.add_edge("increment_retry", "researcher")
    workflow.add_edge("synthesiser", END)
    workflow.add_edge("escalate", END)

    return workflow.compile()

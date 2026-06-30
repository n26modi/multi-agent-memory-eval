import json
from config import groq_client, MODEL
from state import ResearchState


async def planner_agent(state: ResearchState) -> ResearchState:
    client = groq_client()
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research planner. Break the query into 2-3 focused subtasks. "
                    "Reply with JSON: {\"subtasks\": [\"subtask1\", \"subtask2\", ...]}"
                ),
            },
            {"role": "user", "content": state["query"]},
        ],
        response_format={"type": "json_object"},
        max_tokens=256,
    )
    data = json.loads(response.choices[0].message.content)
    raw = data.get("subtasks", [state["query"]])
    # ensure subtasks are plain strings regardless of what the LLM returns
    subtasks = [s if isinstance(s, str) else str(s) for s in raw]
    return {**state, "subtasks": subtasks}

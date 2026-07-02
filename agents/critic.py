import json
from config import groq_client, MODEL
from memory.base import Memory
from state import ResearchState


async def critic_agent(state: ResearchState, memory: Memory) -> ResearchState:
    client = groq_client()
    approved, failure_tags = [], list(state["failure_tags"])

    ref_time = state.get("temporal_context")
    for f in state["findings"]:
        if ref_time is not None:
            # historical query: check the finding was valid at the reference time,
            # not whether a newer version exists now
            if f.valid_at > ref_time:
                failure_tags.append("staleness_failure")
                continue
        else:
            # staleness check — this is where the two backends diverge
            current = await memory.current_value(f.entity, f.relation)
            if current and current.valid_at > f.valid_at and current.value != f.value:
                failure_tags.append("staleness_failure")
                continue

        # grounding check via LLM
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a fact-checker. Score how well-grounded this finding is. "
                        "Reply with JSON: {\"score\": float, \"reason\": str} where score is 0.0-1.0."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Finding: {f.value}\nSource text: {f.retrieved_text}",
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=128,
        )
        data = json.loads(response.choices[0].message.content)
        score = float(data.get("score", 0.0))

        if score >= 0.5:
            approved.append(f)
        else:
            failure_tags.append("retrieval_misalignment")

    quality = len(approved) / len(state["findings"]) if state["findings"] else 0.0
    return {**state, "approved": approved, "failure_tags": failure_tags, "last_quality": quality}

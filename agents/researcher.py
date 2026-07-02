import json
import time
from config import groq_client, MODEL
from memory.base import Finding, Memory
from state import ResearchState


async def researcher_agent(state: ResearchState, memory: Memory) -> ResearchState:
    findings: list[Finding] = []

    ref_time = state.get("temporal_context")
    for subtask in state["subtasks"]:
        hits = await memory.query(subtask, k=3, reference_time=ref_time)
        if hits:
            # top-1 hit only — one finding per subtask keeps quality metric
            # meaningful and makes the Chroma vs Graphiti comparison clean
            findings.append(hits[0])
        else:
            # fallback: LLM extraction when memory is empty (e.g. vertical slice test)
            finding = await _llm_extract(subtask)
            await memory.write(finding)
            findings.append(finding)

    return {**state, "findings": findings}


async def _llm_extract(subtask: str) -> Finding:
    client = groq_client()
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research analyst. Extract one key finding from the subtask. "
                    "Reply with JSON: {\"entity\": str, \"relation\": str, \"value\": str, \"confidence\": float, \"text\": str}"
                ),
            },
            {"role": "user", "content": subtask},
        ],
        response_format={"type": "json_object"},
        max_tokens=256,
    )
    data = json.loads(response.choices[0].message.content)
    return Finding(
        entity=data.get("entity", "unknown"),
        relation=data.get("relation", "unknown"),
        value=data.get("value", "unknown"),
        source_id=subtask,
        confidence=float(data.get("confidence", 0.5)),
        valid_at=time.time(),
        retrieved_text=data.get("text", ""),
    )

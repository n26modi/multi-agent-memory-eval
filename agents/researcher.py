import json
import time
from config import groq_client, MODEL
from memory.base import Finding, Memory
from state import ResearchState


async def researcher_agent(state: ResearchState, memory: Memory) -> ResearchState:
    client = groq_client()
    findings: list[Finding] = []

    for subtask in state["subtasks"]:
        memory_hits = await memory.query(subtask, k=3)
        context = "\n".join(f"- {h.retrieved_text}" for h in memory_hits) or "No prior findings."

        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research analyst. Given a subtask and context, extract one key finding. "
                        "Reply with JSON: {\"entity\": str, \"relation\": str, \"value\": str, \"confidence\": float, \"text\": str}"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Subtask: {subtask}\n\nContext:\n{context}",
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=256,
        )
        data = json.loads(response.choices[0].message.content)
        finding = Finding(
            entity=data.get("entity", "unknown"),
            relation=data.get("relation", "unknown"),
            value=data.get("value", "unknown"),
            source_id=subtask,
            confidence=float(data.get("confidence", 0.5)),
            valid_at=time.time(),
            retrieved_text=data.get("text", ""),
        )
        findings.append(finding)
        await memory.write(finding)

    return {**state, "findings": findings}

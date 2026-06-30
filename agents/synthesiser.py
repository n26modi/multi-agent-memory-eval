from config import groq_client, MODEL
from state import ResearchState


async def synthesiser_agent(state: ResearchState) -> ResearchState:
    client = groq_client()
    findings_text = "\n".join(
        f"- {f.entity} / {f.relation}: {f.value} (confidence: {f.confidence:.2f})"
        for f in state["approved"]
    ) or "No approved findings."

    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a research synthesiser. Write a concise answer to the query based on the findings.",
            },
            {
                "role": "user",
                "content": f"Query: {state['query']}\n\nFindings:\n{findings_text}",
            },
        ],
        max_tokens=512,
    )
    report = response.choices[0].message.content
    return {**state, "final_report": report}

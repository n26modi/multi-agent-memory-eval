import asyncio
import time
from memory.chroma_memory import ChromaMemory
from memory.base import Finding
from graph import build_graph
from state import ResearchState


async def main():
    memory = ChromaMemory()

    # seed two facts so memory isn't empty
    await memory.write(Finding(
        entity="OpenAI", relation="CEO", value="Sam Altman",
        source_id="seed-1", confidence=0.95,
        valid_at=time.time() - 86400,
        retrieved_text="Sam Altman is the CEO of OpenAI as of 2024.",
    ))
    await memory.write(Finding(
        entity="OpenAI", relation="founded", value="2015",
        source_id="seed-2", confidence=0.99,
        valid_at=time.time() - 86400 * 365,
        retrieved_text="OpenAI was founded in 2015 by Sam Altman, Elon Musk, and others.",
    ))

    graph = build_graph(memory)

    initial_state: ResearchState = {
        "query": "What is OpenAI and who runs it?",
        "subtasks": [],
        "findings": [],
        "approved": [],
        "retry_count": 0,
        "max_retries": 2,
        "last_quality": 0.0,
        "failure_tags": [],
        "final_report": "",
    }

    print("Running vertical slice...\n")
    result = await graph.ainvoke(initial_state)

    print("=== SUBTASKS ===")
    for s in result["subtasks"]:
        print(f"  - {s}")

    print("\n=== FINDINGS ===")
    for f in result["findings"]:
        print(f"  [{f.entity}/{f.relation}] {f.value} (conf={f.confidence:.2f})")

    print("\n=== APPROVED ===")
    print(f"  {len(result['approved'])} / {len(result['findings'])} findings approved")

    print("\n=== FAILURE TAGS ===")
    print(f"  {result['failure_tags'] or 'none'}")

    print("\n=== FINAL REPORT ===")
    print(result["final_report"])


asyncio.run(main())

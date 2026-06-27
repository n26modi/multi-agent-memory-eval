import os
from datetime import datetime, timezone

from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.groq_client import GroqClient

from memory.base import Finding, Memory
from memory.embedder import LocalEmbedder

GROQ_MODEL = "llama-3.1-8b-instant"


class GraphitiMemory(Memory):
    def __init__(self, uri: str, user: str, password: str):
        llm_client = GroqClient(
            config=LLMConfig(
                api_key=os.environ["GROQ_API_KEY"],
                model=GROQ_MODEL,
            )
        )
        self.g = Graphiti(
            uri=uri,
            user=user,
            password=password,
            llm_client=llm_client,
            embedder=LocalEmbedder(),
        )

    async def write(self, f: Finding) -> None:
        await self.g.add_episode(
            name=f"{f.entity}:{f.relation}",
            episode_body=f.retrieved_text,
            source_description=f.source_id,
            reference_time=datetime.fromtimestamp(f.valid_at, tz=timezone.utc),
        )

    async def query(self, query_text: str, k: int = 5) -> list[Finding]:
        edges = await self.g.search(query_text, num_results=k)
        return [self._edge_to_finding(e) for e in edges]

    async def current_value(self, entity: str, relation: str) -> Finding | None:
        # Search semantically for this entity+relation pair, then keep only
        # edges that are still live (invalid_at is None = not superseded).
        edges = await self.g.search(f"{entity} {relation}", num_results=20)
        live = [e for e in edges if e.invalid_at is None]
        if not live:
            return None
        best = max(live, key=lambda e: self._valid_at_ts(e))
        return self._edge_to_finding(best)

    def _edge_to_finding(self, edge: EntityEdge) -> Finding:
        return Finding(
            entity=edge.source_node_uuid,
            relation=edge.name,
            value=edge.fact,
            source_id=edge.episodes[0] if edge.episodes else edge.uuid,
            confidence=1.0,
            valid_at=self._valid_at_ts(edge),
            retrieved_text=edge.fact,
        )

    def _valid_at_ts(self, edge: EntityEdge) -> float:
        dt = edge.valid_at or edge.reference_time or edge.created_at
        return dt.timestamp()

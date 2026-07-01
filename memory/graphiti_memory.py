import os
from datetime import datetime, timezone

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.edges import EntityEdge
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.groq_client import GroqClient
from graphiti_core.nodes import EntityNode
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk

from memory.base import Finding, Memory
from memory.embedder import LocalEmbedder


class NoOpReranker(CrossEncoderClient):
    """Pass-through reranker — returns passages unranked with equal scores."""
    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        return [(p, 1.0) for p in passages]


GROQ_MODEL = "llama-3.3-70b-versatile"


class GraphitiMemory(Memory):
    def __init__(self, uri: str, user: str, password: str, group_id: str = "default"):
        llm_client = GroqClient(
            config=LLMConfig(
                api_key=os.environ["GROQ_API_KEY"],
                model=GROQ_MODEL,
                max_tokens=512,
            )
        )
        self.g = Graphiti(
            uri=uri,
            user=user,
            password=password,
            llm_client=llm_client,
            embedder=LocalEmbedder(),
            cross_encoder=NoOpReranker(),
        )
        self.group_id = group_id
        self._initialized = False

    async def _ensure_init(self) -> None:
        if not self._initialized:
            await self.g.build_indices_and_constraints()
            self._initialized = True

    async def write(self, f: Finding) -> None:
        await self._ensure_init()
        valid_at_dt = datetime.fromtimestamp(f.valid_at, tz=timezone.utc)

        # Mark any existing live edges for this entity+relation as superseded.
        # This is what gives Graphiti its temporal advantage: V1 becomes invisible
        # to retrieval once V2 arrives, so the researcher always gets the live fact.
        await self._invalidate_previous(f.entity, f.relation, valid_at_dt)

        src = EntityNode(name=f.entity, group_id=self.group_id, created_at=valid_at_dt)
        tgt = EntityNode(name=f.value, group_id=self.group_id, created_at=valid_at_dt)

        edge = EntityEdge(
            source_node_uuid=src.uuid,
            target_node_uuid=tgt.uuid,
            group_id=self.group_id,
            name=f.relation,
            fact=f.retrieved_text or f"{f.entity} {f.relation} {f.value}",
            valid_at=valid_at_dt,
            invalid_at=None,
            created_at=valid_at_dt,
            episodes=[f.source_id],
        )

        await add_nodes_and_edges_bulk(
            self.g.driver, [], [], [src, tgt], [edge], self.g.embedder
        )

    async def _invalidate_previous(self, entity: str, relation: str, invalid_at: datetime) -> None:
        """Set invalid_at on all live edges for entity+relation so they stop appearing in search."""
        await self.g.driver.execute_query(
            """
            MATCH (n:Entity {group_id: $group_id, name: $entity})-[e:RELATES_TO]->(m:Entity)
            WHERE e.name = $relation AND e.invalid_at IS NULL AND e.group_id = $group_id
            SET e.invalid_at = $invalid_at
            """,
            group_id=self.group_id,
            entity=entity,
            relation=relation,
            invalid_at=invalid_at,
        )

    async def query(self, query_text: str, k: int = 5) -> list[Finding]:
        # Fetch more than k so we have enough after filtering out invalidated edges.
        # Invalidated edges (invalid_at IS NOT NULL) represent superseded facts;
        # excluding them is what makes Graphiti better than a flat vector store.
        edges = await self.g.search(query_text, num_results=k * 3, group_ids=[self.group_id])
        live = [e for e in edges if e.invalid_at is None]
        return [self._edge_to_finding(e) for e in live[:k]]

    async def current_value(self, entity: str, relation: str) -> Finding | None:
        edges = await self.g.search(
            f"{entity} {relation}", num_results=20, group_ids=[self.group_id]
        )
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

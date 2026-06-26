import chromadb
from memory.base import Finding, Memory


class ChromaMemory(Memory):
    def __init__(self, collection_name: str = "findings"):
        self.client = chromadb.Client()
        self.col = self.client.get_or_create_collection(collection_name)
        self._counter = 0

    def write(self, f: Finding) -> None:
        self._counter += 1
        self.col.add(
            ids=[f"f{self._counter}"],
            documents=[f.retrieved_text],
            metadatas=[{
                "entity": f.entity,
                "relation": f.relation,
                "value": f.value,
                "valid_at": f.valid_at,
                "source_id": f.source_id,
                "confidence": f.confidence,
            }],
        )

    def query(self, query_text: str, k: int = 5) -> list[Finding]:
        res = self.col.query(query_texts=[query_text], n_results=k)
        return [
            self._to_finding(m, d)
            for m, d in zip(res["metadatas"][0], res["documents"][0])
        ]

    def current_value(self, entity: str, relation: str) -> Finding | None:
        hits = self._scan(entity, relation)
        if not hits:
            return None
        return max(hits, key=lambda f: f.valid_at)

    def _scan(self, entity: str, relation: str) -> list[Finding]:
        res = self.col.get(
            where={"$and": [{"entity": entity}, {"relation": relation}]},
        )
        return [
            self._to_finding(m, d)
            for m, d in zip(res["metadatas"], res["documents"])
        ]

    def _to_finding(self, metadata: dict, document: str) -> Finding:
        return Finding(
            entity=metadata["entity"],
            relation=metadata["relation"],
            value=metadata["value"],
            source_id=metadata["source_id"],
            confidence=metadata["confidence"],
            valid_at=metadata["valid_at"],
            retrieved_text=document,
        )

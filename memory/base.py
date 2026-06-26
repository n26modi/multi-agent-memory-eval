from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class Finding:
    entity: str        # e.g. "Acme Corp"
    relation: str      # e.g. "series_b_lead"
    value: str         # e.g. "Bar Capital"
    source_id: str
    confidence: float
    valid_at: float    # unix timestamp the source asserts this was true
    retrieved_text: str


class Memory(ABC):
    @abstractmethod
    def write(self, finding: Finding) -> None: ...

    @abstractmethod
    def query(self, query_text: str, k: int = 5) -> list[Finding]: ...

    @abstractmethod
    def current_value(self, entity: str, relation: str) -> Optional[Finding]:
        # ChromaMemory fakes this with a recency tiebreak.
        # GraphitiMemory answers it correctly via invalid_at filtering.
        # That asymmetry is what the experiment measures.
        ...

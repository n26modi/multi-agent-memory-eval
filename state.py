from typing import TypedDict
from memory.base import Finding


class ResearchState(TypedDict):
    query: str
    subtasks: list[str]
    findings: list[Finding]
    approved: list[Finding]
    retry_count: int
    max_retries: int
    last_quality: float
    failure_tags: list[str]
    final_report: str
    temporal_context: float | None  # unix timestamp; researcher passes this to memory.query() for point-in-time retrieval

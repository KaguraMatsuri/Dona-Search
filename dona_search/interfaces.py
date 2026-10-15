from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .config import ProviderConfig


@dataclass
class ProcessedQuery:
    raw: str
    keywords: list[str]
    filters: dict[str, str]  # e.g., {"time": "recent", "lang": "en"}


@dataclass
class QueryPlan:
    raw: str
    queries: list[str]


class QueryPlanner(Protocol):
    def plan(self, query: str) -> QueryPlan: ...
    def warmup(self) -> None: ...
    def cleanup(self) -> None: ...
    def close(self) -> None: ...

@dataclass
class SearchHit:
    rank: int
    title: str
    url: str
    snippet: str
    provider: str
    origins: list[dict] = field(default_factory=list)

@dataclass
class Document:
    url: str
    title: str
    text: str
    fetched_at: str

@dataclass
class Evidence:
    text: str              # Verbatim span from Document.text
    start: int             # Character offset in doc
    end: int
    relevance: int         # 0-100
    confidence: float      # 0.0-1.0

@dataclass(frozen=True)
class SnippetEvidence:
    text: str
    start: int            # Character offsets in SearchHit.snippet, never Document.text
    end: int


@dataclass
class OrganizationDecision:
    order: list[int]
    selections: list[int | None]  # Indexed by input hit id, not by sorted position


class Organizer(Protocol):
    def organize(self, query: str, hits: list[SearchHit]) -> OrganizationDecision: ...
    def warmup(self) -> None: ...
    def cleanup(self) -> None: ...
    def close(self) -> None: ...


@dataclass
class SearchResult:
    rank: int
    preview: str           # Original search engine snippet
    url: str
    summary: Evidence | None
    title: str = ''
    origins: list[dict] = field(default_factory=list)
    snippet_evidence: SnippetEvidence | None = None

    def format(self, *, organized: bool = False) -> str:
        if organized:
            from .output import format_result
            return format_result(self)
        result = (
            f"[{self.rank}] {self.preview}\n"
            f"    {self.url}\n"
        )
        if self.summary is not None:
            result += f"    [{self.summary.text}] [{self.summary.relevance}][{self.summary.confidence:.0%}]\n"
        return result

class QueryProcessor(Protocol):
    def process(self, query: str) -> ProcessedQuery: ...
    def warmup(self) -> None: ...
    def cleanup(self) -> None: ...
    def close(self) -> None: ...

class SearchProvider(Protocol):
    name: str
    config: 'ProviderConfig'
    def search(self, query: str, k: int) -> 'ProviderResult': ...


class LegacySearchProvider(Protocol):
    name: str
    def search(self, query: str, k: int) -> list[SearchHit]: ...


@dataclass
class ProviderResult:
    provider: str
    status: str
    hits: list[SearchHit] = field(default_factory=list)
    http_status: int | None = None
    error_code: str | None = None
    elapsed_seconds: float = 0
    cleanup_error: str | None = None


@dataclass(frozen=True)
class SearchTask:
    id: int
    provider: str
    query: str
    query_index: int
    limit: int


@dataclass
class TaskResult:
    task: SearchTask
    outcome: ProviderResult


@dataclass
class SearchReport:
    query: str
    results: list[SearchResult]
    providers: list[ProviderResult]
    candidate_count: int
    ranking_status: str
    ranking_error: str | None = None
    timings: dict[str, float] = field(default_factory=dict)
    candidates: list[SearchHit] = field(default_factory=list)
    query_status: str = 'not_requested'
    query_error: str | None = None
    search_queries: list[str] = field(default_factory=list)
    tasks: list[TaskResult] = field(default_factory=list)
    organizing_status: str = 'not_requested'
    organizing_error: str | None = None


class OrderedRanker(Protocol):
    def rank(self, query: str, hits: list[SearchHit]) -> list[int]: ...
    def warmup(self) -> None: ...
    def cleanup(self) -> None: ...
    def close(self) -> None: ...

class Fetcher(Protocol):
    def fetch(self, url: str) -> Document | None: ...

class Reranker(Protocol):
    def score_batch(self, query: str, items: list) -> list[float]: ...

class Extractor(Protocol):
    def extract(self, query: str, doc: Document, top_k: int = 1) -> list[Evidence]: ...

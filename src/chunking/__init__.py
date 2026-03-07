"""Stage 3 chunking exports."""

from .engine import ChunkingConfig, ChunkingEngine
from .page_index import PageIndexBuilder, PageIndexTree
from .page_index_query import PageIndexMatch, PageIndexQueryEngine
from .page_index_summarizer import (
    OllamaSummaryBackend,
    PageIndexSummarizer,
    SummaryBackend,
    SummaryBackendError,
    SummaryInput,
)
from .provenance import ProvenanceChainBuilder, ProvenanceChainError
from .retrieval_evaluation import (
    LabeledRetrievalQuery,
    RetrievalEvaluationReport,
    RetrievalEvaluator,
    RetrievalMetrics,
    RetrievalRunResult,
)
from .sections import SectionCandidate, SectionPathInferer
from .validator import ChunkValidationError, ChunkValidator
from .vector_store import (
    ChromaVectorStore,
    EmbeddingBackend,
    OllamaEmbeddingBackend,
    VectorStoreError,
    VectorStoreMatch,
)

__all__ = [
    "ChunkingConfig",
    "ChunkingEngine",
    "ChromaVectorStore",
    "EmbeddingBackend",
    "OllamaEmbeddingBackend",
    "PageIndexBuilder",
    "PageIndexMatch",
    "PageIndexQueryEngine",
    "PageIndexTree",
    "OllamaSummaryBackend",
    "PageIndexSummarizer",
    "LabeledRetrievalQuery",
    "ProvenanceChainBuilder",
    "ProvenanceChainError",
    "RetrievalEvaluationReport",
    "RetrievalEvaluator",
    "RetrievalMetrics",
    "RetrievalRunResult",
    "SectionCandidate",
    "SectionPathInferer",
    "SummaryBackend",
    "SummaryBackendError",
    "SummaryInput",
    "ChunkValidationError",
    "ChunkValidator",
    "VectorStoreError",
    "VectorStoreMatch",
]

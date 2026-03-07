"""Vector-store ingestion and retrieval for Phase 3 artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.models.chunking import Chunk, LDU
from src.utils.hashing import canonicalize_text


class EmbeddingBackend(Protocol):
    """Pluggable embedding backend for vector-store operations."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents deterministically."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query deterministically."""


class OllamaEmbeddingClientProtocol(Protocol):
    """Minimal Ollama client surface needed by the embedding backend."""

    def embed(self, **kwargs: Any) -> dict[str, Any]:
        """Submit an embedding request and return the raw Ollama response."""


class ChromaCollectionProtocol(Protocol):
    """Minimal Chroma collection API used by this module."""

    def upsert(
        self,
        *,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Insert or replace vectors."""

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """Query vectors."""


class ChromaClientProtocol(Protocol):
    """Minimal client surface needed to create a collection."""

    def get_or_create_collection(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> ChromaCollectionProtocol:
        """Return an existing collection or create a new one."""


class VectorStoreError(RuntimeError):
    """Raised when vector-store operations fail."""


class OllamaEmbeddingBackend:
    """Ollama-backed embedding backend for vector-store operations."""

    def __init__(
        self,
        client: OllamaEmbeddingClientProtocol | None = None,
        *,
        model: str = "qwen3-embedding:0.6b",
        keep_alive: str | int = "0s",
    ) -> None:
        self.client = client or self._default_client()
        self.model = model
        self.keep_alive = keep_alive

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._embed(inputs=texts)
        embeddings = response.get("embeddings")
        if not isinstance(embeddings, list):
            raise VectorStoreError("Ollama embedding response did not include embeddings")
        normalized = [self._coerce_embedding(item) for item in embeddings]
        if len(normalized) != len(texts):
            raise VectorStoreError("Ollama embedding response count did not match input count")
        return normalized

    def embed_query(self, text: str) -> list[float]:
        response = self._embed(inputs=[text])
        embeddings = response.get("embeddings")
        if not isinstance(embeddings, list) or not embeddings:
            raise VectorStoreError("Ollama embedding response did not include embeddings")
        return self._coerce_embedding(embeddings[0])

    def _embed(self, *, inputs: list[str]) -> dict[str, Any]:
        cleaned_inputs = [canonicalize_text(text) for text in inputs]
        try:
            return self.client.embed(
                model=self.model,
                input=cleaned_inputs,
                keep_alive=self.keep_alive,
            )
        except Exception as exc:  # pragma: no cover - exercised via tests with fake client
            raise VectorStoreError(f"Ollama embedding request failed: {exc}") from exc

    def _coerce_embedding(self, value: Any) -> list[float]:
        if not isinstance(value, list):
            raise VectorStoreError("Ollama embedding response contained a non-list embedding")
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise VectorStoreError("Ollama embedding response contained a non-numeric embedding value") from exc

    def _default_client(self) -> OllamaEmbeddingClientProtocol:
        try:
            from ollama import Client
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise VectorStoreError("Ollama client is not installed") from exc
        return Client()


@dataclass(frozen=True)
class VectorStoreMatch:
    """Retrieved vector-store match with preserved metadata."""

    record_id: str
    text: str
    metadata: dict[str, Any]
    distance: float | None


class ChromaVectorStore:
    """Local ChromaDB vector-store wrapper for LDUs and chunks."""

    def __init__(
        self,
        embedding_backend: EmbeddingBackend,
        *,
        collection_name: str = "phase3",
        client: ChromaClientProtocol | None = None,
        collection: ChromaCollectionProtocol | None = None,
        persist_directory: str | Path = ".refinery/chroma",
        max_upsert_batch_size: int = 5000,
    ) -> None:
        self.embedding_backend = embedding_backend
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.max_upsert_batch_size = max(1, max_upsert_batch_size)
        self._collection = collection or self._create_collection(client=client)

    def ingest_ldus(self, ldus: list[LDU]) -> None:
        if not ldus:
            return
        ids = [self._require_id(ldu.ldu_id, "LDU") for ldu in ldus]
        texts = [ldu.text for ldu in ldus]
        embeddings = self.embedding_backend.embed_documents(texts)
        metadatas = [self._ldu_metadata(ldu) for ldu in ldus]
        self._upsert(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)

    def ingest_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        ids = [self._require_id(chunk.chunk_id, "Chunk") for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_backend.embed_documents(texts)
        metadatas = [self._chunk_metadata(chunk) for chunk in chunks]
        self._upsert(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)

    def query(
        self,
        topic: str,
        *,
        top_k: int = 3,
        section_path: tuple[str, ...] | None = None,
        record_type: str | None = None,
    ) -> list[VectorStoreMatch]:
        where = self._build_where(section_path=section_path, record_type=record_type)
        query_embedding = self.embedding_backend.embed_query(topic)
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return self._matches_from_query(result)

    def _create_collection(self, client: ChromaClientProtocol | None) -> ChromaCollectionProtocol:
        active_client = client or self._default_client()
        return active_client.get_or_create_collection(
            self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _default_client(self) -> ChromaClientProtocol:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise VectorStoreError("chromadb is not installed") from exc
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(self.persist_directory))

    def _upsert(
        self,
        *,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if len(ids) != len(texts) or len(texts) != len(embeddings) or len(embeddings) != len(metadatas):
            raise VectorStoreError("Vector-store upsert payload lengths must match")
        for start in range(0, len(ids), self.max_upsert_batch_size):
            end = start + self.max_upsert_batch_size
            self._collection.upsert(
                ids=ids[start:end],
                documents=texts[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )

    def _ldu_metadata(self, ldu: LDU) -> dict[str, Any]:
        metadata = {
            "record_type": "ldu",
            "doc_id": ldu.doc_id,
            "document_name": ldu.metadata.get("document_name", ldu.doc_id),
            "page_number": ldu.page_number,
            "start_page": ldu.page_number,
            "end_page": ldu.page_number,
            "bbox": list(ldu.bbox),
            "section_path_str": self._section_path_string(ldu.section_path),
            "chunk_id": None,
            "ldu_ids": [self._require_id(ldu.ldu_id, "LDU")],
            "content_hash": ldu.content_hash,
            "content_hashes": [ldu.content_hash],
            "strategy_used": ldu.metadata.get("strategy_used"),
            "confidence_score": ldu.metadata.get("confidence_score"),
        }
        if ldu.section_path:
            metadata["section_path"] = list(ldu.section_path)
        return metadata

    def _chunk_metadata(self, chunk: Chunk) -> dict[str, Any]:
        metadata = {
            "record_type": "chunk",
            "doc_id": chunk.doc_id,
            "document_name": chunk.metadata.get("document_name", chunk.doc_id),
            "page_number": chunk.page_number,
            "start_page": chunk.page_number,
            "end_page": chunk.page_number,
            "bbox": list(chunk.bbox),
            "section_path_str": self._section_path_string(chunk.section_path),
            "chunk_id": self._require_id(chunk.chunk_id, "Chunk"),
            "ldu_ids": list(chunk.ldu_ids),
            "content_hash": chunk.content_hash,
            "content_hashes": [chunk.content_hash],
            "strategy_used": chunk.metadata.get("strategy_used"),
            "confidence_score": chunk.metadata.get("confidence_score"),
        }
        if chunk.section_path:
            metadata["section_path"] = list(chunk.section_path)
        return metadata

    def _build_where(
        self,
        *,
        section_path: tuple[str, ...] | None,
        record_type: str | None,
    ) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []
        if section_path is not None:
            clauses.append({"section_path_str": self._section_path_string(section_path)})
        if record_type is not None:
            clauses.append({"record_type": record_type})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def _matches_from_query(self, result: dict[str, Any]) -> list[VectorStoreMatch]:
        ids = self._first_result_list(result.get("ids"))
        documents = self._first_result_list(result.get("documents"))
        metadatas = self._first_result_list(result.get("metadatas"))
        distances = self._first_result_list(result.get("distances"))

        matches: list[VectorStoreMatch] = []
        for index, record_id in enumerate(ids):
            text = documents[index] if index < len(documents) else ""
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = distances[index] if index < len(distances) else None
            if not isinstance(record_id, str):
                continue
            if not isinstance(text, str):
                text = ""
            if not isinstance(metadata, dict):
                metadata = {}
            if distance is not None and not isinstance(distance, (float, int)):
                distance = None
            matches.append(
                VectorStoreMatch(
                    record_id=record_id,
                    text=text,
                    metadata=metadata,
                    distance=float(distance) if distance is not None else None,
                )
            )
        return matches

    def _first_result_list(self, value: Any) -> list[Any]:
        if isinstance(value, list) and value and isinstance(value[0], list):
            return value[0]
        if isinstance(value, list):
            return value
        return []

    def _section_path_string(self, section_path: tuple[str, ...]) -> str:
        return " > ".join(section_path)

    def _require_id(self, value: str | None, label: str) -> str:
        if value is None:
            raise VectorStoreError(f"{label} must have an id before vector-store ingestion")
        return value

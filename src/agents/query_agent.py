"""Deterministic Phase 4 query agent with retrieval-backed provenance."""

from __future__ import annotations

from dataclasses import dataclass
import re

from src.chunking.page_index import PageIndexTree
from src.chunking.page_index_query import PageIndexMatch, PageIndexQueryEngine
from src.chunking.provenance import ProvenanceChainBuilder, ProvenanceChainError
from src.chunking.vector_store import VectorStoreMatch
from src.agents.structured_fact_query import StructuredFactQueryBackend
from src.models import ExtractionStrategy, ProvenanceChain, ProvenanceChainEntry, ProvenanceRef
from src.utils.hashing import canonicalize_text


@dataclass(frozen=True)
class QueryAgentResult:
    """Query response with explicit verification status."""

    query: str
    status: str
    answer: str | None
    provenance_chain: ProvenanceChain | None
    retrieval_matches: tuple[VectorStoreMatch, ...]
    page_index_matches: tuple[PageIndexMatch, ...]
    route: str
    failure_reason: str | None = None


class QueryAgent:
    """Combines PageIndex traversal, vector retrieval, and provenance assembly."""

    def __init__(
        self,
        *,
        page_index_backend: PageIndexQueryEngine,
        vector_backend,
        provenance_builder: ProvenanceChainBuilder | None = None,
        structured_query_backend: StructuredFactQueryBackend | None = None,
    ) -> None:
        self.page_index_backend = page_index_backend
        self.vector_backend = vector_backend
        self.provenance_builder = provenance_builder or ProvenanceChainBuilder()
        self.structured_query_backend = structured_query_backend

    def answer(
        self,
        *,
        tree: PageIndexTree,
        query: str,
        top_k: int = 3,
        section_top_k: int = 3,
        record_type: str = "chunk",
    ) -> QueryAgentResult:
        if self.structured_query_backend is not None:
            structured_result = self.structured_query_backend.answer(query)
            if structured_result is not None:
                return QueryAgentResult(
                    query=query,
                    status="verified",
                    answer=structured_result.answer,
                    provenance_chain=structured_result.provenance_chain,
                    retrieval_matches=(),
                    page_index_matches=(),
                    route="structured_query",
                    failure_reason=None,
                )

        page_index_matches = tuple(self.page_index_backend.query(tree, query, top_k=section_top_k))
        assisted_matches = tuple(
            self._assisted_retrieve(
                query=query,
                page_index_matches=page_index_matches,
                top_k=top_k,
                record_type=record_type,
            )
        )

        route = "pageindex_assisted"
        retrieval_matches = assisted_matches
        if not retrieval_matches:
            route = "baseline_vector"
            retrieval_matches = tuple(self._baseline_retrieve(query=query, top_k=top_k, record_type=record_type))

        if retrieval_matches:
            retrieval_matches = self._rerank_matches_for_query(
                query=query,
                retrieval_matches=retrieval_matches,
            )
            retrieval_matches = self._rerank_matches_for_definitional_query(
                query=query,
                retrieval_matches=retrieval_matches,
            )
            retrieval_matches = retrieval_matches[:top_k]

        if not retrieval_matches:
            lexical_fallback = self._pageindex_lexical_fallback(
                tree=tree,
                query=query,
                page_index_matches=page_index_matches,
            )
            if lexical_fallback is not None:
                return lexical_fallback

            return QueryAgentResult(
                query=query,
                status="unverifiable",
                answer=None,
                provenance_chain=None,
                retrieval_matches=(),
                page_index_matches=page_index_matches,
                route=route,
                failure_reason="No supporting evidence retrieved",
            )

        try:
            provenance_chain = self.provenance_builder.build(retrieval_matches, query=query)
        except ProvenanceChainError as exc:
            return QueryAgentResult(
                query=query,
                status="unverifiable",
                answer=None,
                provenance_chain=None,
                retrieval_matches=retrieval_matches,
                page_index_matches=page_index_matches,
                route=route,
                failure_reason=str(exc),
            )

        retrieval_matches = self._rerank_matches_for_parts_list_query(
            query=query,
            retrieval_matches=retrieval_matches,
        )

        answer = self._synthesize_answer(
            query=query,
            retrieval_matches=retrieval_matches,
        )
        if not answer:
            return QueryAgentResult(
                query=query,
                status="unverifiable",
                answer=None,
                provenance_chain=None,
                retrieval_matches=retrieval_matches,
                page_index_matches=page_index_matches,
                route=route,
                failure_reason="Retrieved evidence did not yield an answerable snippet",
            )

        return QueryAgentResult(
            query=query,
            status="verified",
            answer=answer,
            provenance_chain=provenance_chain,
            retrieval_matches=retrieval_matches,
            page_index_matches=page_index_matches,
            route=route,
            failure_reason=None,
        )

    def _assisted_retrieve(
        self,
        *,
        query: str,
        page_index_matches: tuple[PageIndexMatch, ...],
        top_k: int,
        record_type: str,
    ) -> list[VectorStoreMatch]:
        collected: list[VectorStoreMatch] = []
        seen_record_ids: set[str] = set()
        retrieval_queries = self._build_assisted_retrieval_queries(query)
        intent = self._detect_query_intent(canonicalize_text(query).lower())
        per_query_top_k = top_k
        if len(retrieval_queries) > 1 or any(intent.values()):
            per_query_top_k = max(top_k, 3)
        max_candidates = max(top_k, top_k * len(retrieval_queries))

        for section_match in page_index_matches:
            for retrieval_query in retrieval_queries:
                vector_matches = self.vector_backend.query(
                    retrieval_query,
                    top_k=per_query_top_k,
                    section_path=section_match.section_path,
                    record_type=record_type,
                )
                for vector_match in vector_matches:
                    if vector_match.record_id in seen_record_ids:
                        continue
                    seen_record_ids.add(vector_match.record_id)
                    collected.append(vector_match)
                    if len(collected) >= max_candidates:
                        return collected
        return collected

    def _baseline_retrieve(
        self,
        *,
        query: str,
        top_k: int,
        record_type: str,
    ) -> list[VectorStoreMatch]:
        retrieval_queries = self._build_assisted_retrieval_queries(query)
        collected: list[VectorStoreMatch] = []
        seen_record_ids: set[str] = set()
        intent = self._detect_query_intent(canonicalize_text(query).lower())
        per_query_top_k = top_k
        if len(retrieval_queries) > 1 or any(intent.values()):
            per_query_top_k = max(top_k, 3)
        for retrieval_query in retrieval_queries:
            vector_matches = self.vector_backend.query(
                retrieval_query,
                top_k=per_query_top_k,
                record_type=record_type,
            )
            for vector_match in vector_matches:
                if vector_match.record_id in seen_record_ids:
                    continue
                seen_record_ids.add(vector_match.record_id)
                collected.append(vector_match)
        return collected

    def _synthesize_answer(
        self,
        *,
        query: str,
        retrieval_matches: tuple[VectorStoreMatch, ...],
    ) -> str:
        if self._is_parts_list_query(query):
            for match in retrieval_matches:
                snippet = canonicalize_text(match.text)
                normalized = self._normalize_parts_list_snippet(query=query, snippet=snippet)
                if normalized:
                    return normalized

        snippets: list[str] = []
        seen: set[str] = set()
        for match in retrieval_matches:
            snippet = canonicalize_text(match.text)
            if not snippet or snippet in seen:
                continue
            seen.add(snippet)
            snippets.append(snippet)
            if len(snippets) == 2:
                break
        return " ".join(snippets)

    def _pageindex_lexical_fallback(
        self,
        *,
        tree: PageIndexTree,
        query: str,
        page_index_matches: tuple[PageIndexMatch, ...],
    ) -> QueryAgentResult | None:
        """Fallback path when vector retrieval returns no matches.

        Uses PageIndex summaries directly to answer the query and constructs
        a minimal provenance chain from the best-matching node.
        """

        if not page_index_matches:
            return None

        best_match = self._select_best_pageindex_fallback_match(
            query=query,
            page_index_matches=page_index_matches,
        )

        if best_match is None or not best_match.summary:
            return None

        nodes_by_id = {node.node_id: node for node in tree.nodes}
        node = nodes_by_id.get(best_match.node_id)
        if node is None or node.bbox is None:
            return None

        snippet = canonicalize_text(best_match.summary)
        if not snippet:
            return None

        provenance = ProvenanceRef(
            document_name=tree.doc_id,
            doc_id=tree.doc_id,
            page_number=node.start_page,
            bbox=node.bbox,
            content_hash=ProvenanceRef.make_hash(snippet),
            strategy_used=ExtractionStrategy.strategy_a,
            confidence_score=1.0,
        )
        entry = ProvenanceChainEntry(
            record_id=node.node_id,
            record_type="pageindex",
            section_path=node.section_path,
            snippet=snippet,
            distance=None,
            provenance=provenance,
        )
        provenance_chain = ProvenanceChain(entries=(entry,), query=query)

        return QueryAgentResult(
            query=query,
            status="verified",
            answer=snippet,
            provenance_chain=provenance_chain,
            retrieval_matches=(),
            page_index_matches=(best_match,),
            route="pageindex_lexical",
            failure_reason=None,
        )

    def _select_best_pageindex_fallback_match(
        self,
        *,
        query: str,
        page_index_matches: tuple[PageIndexMatch, ...],
    ) -> PageIndexMatch | None:
        summarized = [candidate for candidate in page_index_matches if candidate.summary]
        if not summarized:
            return None

        normalized_query = canonicalize_text(query).lower()
        if not self._is_parts_list_query(normalized_query):
            return summarized[0]

        target_phrase = self._extract_parts_target_phrase(normalized_query)
        scored: list[tuple[int, int, PageIndexMatch]] = []
        for index, candidate in enumerate(summarized):
            summary = canonicalize_text(candidate.summary or "").lower()
            score = 0
            if "parts of" in summary:
                score += 2
            if "should contain" in summary:
                score += 2
            if target_phrase and target_phrase in summary:
                score += 3
            scored.append((score, index, candidate))

        scored_sorted = sorted(scored, key=lambda item: (-item[0], item[1]))
        return scored_sorted[0][2]

    def _rerank_matches_for_definitional_query(
        self,
        *,
        query: str,
        retrieval_matches: tuple[VectorStoreMatch, ...],
    ) -> tuple[VectorStoreMatch, ...]:
        """Prefer definition-like chunks for definitional queries.

        Heuristics:
        - Trigger only for queries that look like "what is X", "define X", or "definition of X".
        - Within those, boost matches whose snippets:
          - Contain the key phrase from the query.
          - Start with "what is"/"what are".
          - Contain patterns like "<term> is" near the beginning.
          - Come from earlier pages (weak prior that definitions appear earlier).
        """

        normalized_query = canonicalize_text(query).lower()
        definitional_prefixes = (
            "what is ",
            "what is a ",
            "what is an ",
            "what are ",
            "define ",
            "definition of ",
        )

        if not any(normalized_query.startswith(prefix) for prefix in definitional_prefixes):
            return retrieval_matches

        key_phrase = normalized_query
        for prefix in definitional_prefixes:
            if key_phrase.startswith(prefix):
                key_phrase = key_phrase[len(prefix) :]
                break
        key_phrase = key_phrase.strip(" ?!.,;:\"'")

        key_tokens = set(re.findall(r"[a-z0-9]+", key_phrase))

        scored: list[tuple[int, int, VectorStoreMatch]] = []
        any_positive = False

        for index, match in enumerate(retrieval_matches):
            snippet = canonicalize_text(match.text).lower()
            score = 0

            if snippet.startswith("what is ") or snippet.startswith("what are "):
                score += 3

            if key_phrase and key_phrase in snippet:
                score += 3

            for token in key_tokens:
                if not token:
                    continue
                pattern_variants = (
                    f"{token} is ",
                    f"{token} is a ",
                    f"{token} is an ",
                )
                if any(variant in snippet for variant in pattern_variants):
                    score += 2
                    break

            is_pos = snippet.find(" is ")
            if 0 <= is_pos <= 80:
                score += 1

            page_number = match.metadata.get("page_number")
            if isinstance(page_number, int) and page_number <= 3:
                score += 1

            if score > 0:
                any_positive = True

            scored.append((score, index, match))

        if not any_positive:
            return retrieval_matches

        scored_sorted = sorted(scored, key=lambda item: (-item[0], item[1]))
        return tuple(match for _, _, match in scored_sorted)

    def _rerank_matches_for_query(
        self,
        *,
        query: str,
        retrieval_matches: tuple[VectorStoreMatch, ...],
    ) -> tuple[VectorStoreMatch, ...]:
        normalized_query = canonicalize_text(query).lower()
        tokens = self._query_tokens(normalized_query)
        if not tokens:
            return retrieval_matches

        intent = self._detect_query_intent(normalized_query)
        is_content_query = (
            intent["is_definitional"]
            or intent["is_section"]
            or intent["is_numeric_financial"]
            or len(tokens) >= 2
        )

        scored: list[tuple[int, float, int, VectorStoreMatch]] = []
        any_positive = False
        for index, match in enumerate(retrieval_matches):
            snippet = canonicalize_text(match.text).lower()
            section_text = self._section_text(match)
            haystack = f"{section_text} {snippet}".strip()

            overlap = sum(1 for token in tokens if token in haystack)
            section_overlap = sum(1 for token in tokens if token in section_text)
            score = overlap + (2 * section_overlap)

            if intent["is_definitional"]:
                if snippet.startswith("what is ") or snippet.startswith("what are "):
                    score += 2
                if " is " in snippet:
                    score += 1
                if any(term in snippet for term in ("definition", "defined as", "refers to")):
                    score += 2

            if intent["is_section"]:
                if any(keyword in section_text for keyword in self._section_query_keywords()):
                    score += 3
                if section_text and re.search(r"\b\d+(\.\d+)*\b", section_text):
                    score += 1

            if intent["is_numeric_financial"]:
                if any(term in haystack for term in self._financial_query_keywords()):
                    score += 2
                if re.search(r"\b(19|20)\d{2}\b", normalized_query) and re.search(r"\b(19|20)\d{2}\b", haystack):
                    score += 2
                if re.search(r"\d", snippet):
                    score += 1
                if any(marker in snippet for marker in ("|", ";", "table", "highlights")):
                    score += 1

            if is_content_query and self._looks_like_non_core_section(haystack):
                score -= 4

            if score > 0:
                any_positive = True

            distance = match.distance if isinstance(match.distance, (float, int)) else 1e9
            scored.append((score, float(distance), index, match))

        if not any_positive:
            return retrieval_matches

        scored_sorted = sorted(scored, key=lambda item: (-item[0], item[1], item[2]))
        return tuple(match for _, _, _, match in scored_sorted)

    def _rerank_matches_for_parts_list_query(
        self,
        *,
        query: str,
        retrieval_matches: tuple[VectorStoreMatch, ...],
    ) -> tuple[VectorStoreMatch, ...]:
        normalized_query = canonicalize_text(query).lower()
        if not self._is_parts_list_query(normalized_query):
            return retrieval_matches

        desired_phrase = self._extract_parts_target_phrase(normalized_query)
        scored: list[tuple[int, int, VectorStoreMatch]] = []
        any_positive = False

        for index, match in enumerate(retrieval_matches):
            snippet = canonicalize_text(match.text).lower()
            score = 0

            if "parts of" in snippet:
                score += 3
            if "should contain the following" in snippet:
                score += 2
            if desired_phrase and desired_phrase in snippet:
                score += 3
            if "chapter" in snippet or "title page" in snippet:
                score += 1

            if score > 0:
                any_positive = True
            scored.append((score, index, match))

        if not any_positive:
            return retrieval_matches

        scored_sorted = sorted(scored, key=lambda item: (-item[0], item[1]))
        return tuple(match for _, _, match in scored_sorted)

    def _normalize_parts_list_snippet(self, *, query: str, snippet: str) -> str:
        normalized = canonicalize_text(snippet)
        if not normalized:
            return ""

        lowered = normalized.lower()
        marker = "parts of"
        target_phrase = self._extract_parts_target_phrase(canonicalize_text(query).lower())
        if target_phrase:
            candidate = f"parts of {target_phrase}"
            marker_index = lowered.find(candidate)
            if marker_index >= 0:
                normalized = normalized[marker_index:]
                lowered = normalized.lower()

        if marker in lowered:
            marker_index = lowered.find(marker)
            normalized = normalized[marker_index:]

        return normalized

    def _is_parts_list_query(self, query: str) -> bool:
        normalized = canonicalize_text(query).lower()
        parts_patterns = (
            "what are the parts of ",
            "what are parts of ",
            "what are the components of ",
            "what are components of ",
            "list the parts of ",
            "list the components of ",
        )
        return any(normalized.startswith(pattern) for pattern in parts_patterns)

    def _extract_parts_target_phrase(self, query: str) -> str:
        normalized = canonicalize_text(query).lower()
        prefixes = (
            "what are the parts of ",
            "what are parts of ",
            "what are the components of ",
            "what are components of ",
            "list the parts of ",
            "list the components of ",
        )
        for prefix in prefixes:
            if normalized.startswith(prefix):
                return normalized[len(prefix) :].strip(" ?!.,;:\"'")
        return ""

    def _build_assisted_retrieval_queries(self, query: str) -> tuple[str, ...]:
        base = canonicalize_text(query).strip()
        if not base:
            return (query,)

        queries: list[str] = [base]
        normalized = base.lower()
        intent = self._detect_query_intent(normalized)

        if intent["is_definitional"]:
            key_phrase = self._extract_definitional_key_phrase(normalized)
            if key_phrase:
                queries.append(f"{key_phrase} definition")
                queries.append(f"{key_phrase} is")
                queries.append(f"what is {key_phrase}")

        if intent["is_section"]:
            queries.append(f"{base} section")
            queries.append(f"{base} heading")

        if intent["is_numeric_financial"]:
            queries.append(f"{base} financial highlights")
            queries.append(f"{base} table")

        if self._is_parts_list_query(base):
            target = self._extract_parts_target_phrase(base)
            if target:
                queries.append(f"parts of {target}")
                queries.append(f"{target} should contain the following")
            queries.append("parts should contain the following")

        seen: set[str] = set()
        deduped: list[str] = []
        for candidate in queries:
            lowered = candidate.lower().strip()
            if not lowered or lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(candidate)
        return tuple(deduped)

    def _detect_query_intent(self, normalized_query: str) -> dict[str, bool]:
        definitional_prefixes = (
            "what is ",
            "what is a ",
            "what is an ",
            "what are ",
            "define ",
            "definition of ",
        )
        is_definitional = any(normalized_query.startswith(prefix) for prefix in definitional_prefixes)
        is_section = any(keyword in normalized_query for keyword in self._section_query_keywords())
        has_year = re.search(r"\b(19|20)\d{2}\b", normalized_query) is not None
        is_numeric_financial = has_year or any(
            keyword in normalized_query for keyword in self._financial_query_keywords()
        )
        return {
            "is_definitional": is_definitional,
            "is_section": is_section,
            "is_numeric_financial": is_numeric_financial,
        }

    def _extract_definitional_key_phrase(self, normalized_query: str) -> str:
        prefixes = (
            "what is ",
            "what is a ",
            "what is an ",
            "what are ",
            "define ",
            "definition of ",
        )
        key_phrase = normalized_query
        for prefix in prefixes:
            if key_phrase.startswith(prefix):
                key_phrase = key_phrase[len(prefix) :]
                break
        return key_phrase.strip(" ?!.,;:\"'")

    def _query_tokens(self, query: str) -> tuple[str, ...]:
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "by",
            "for",
            "from",
            "in",
            "is",
            "of",
            "on",
            "or",
            "the",
            "to",
            "what",
        }
        tokens = [token for token in re.findall(r"[a-z0-9]+", query) if token not in stopwords]
        return tuple(dict.fromkeys(tokens))

    def _section_query_keywords(self) -> tuple[str, ...]:
        return (
            "research question",
            "research questions",
            "objective",
            "objectives",
            "methodology",
            "results",
            "conclusion",
            "statement of the problem",
        )

    def _financial_query_keywords(self) -> tuple[str, ...]:
        return (
            "ratio",
            "profit",
            "assets",
            "liabilities",
            "premium",
            "premiums",
            "claims",
            "cash flow",
            "cashflows",
        )

    def _section_text(self, match: VectorStoreMatch) -> str:
        section_path_value = match.metadata.get("section_path")
        if not isinstance(section_path_value, list):
            return ""
        section_path = [str(item) for item in section_path_value if isinstance(item, str)]
        return canonicalize_text(" ".join(section_path)).lower()

    def _looks_like_non_core_section(self, text: str) -> bool:
        non_core_markers = (
            "title page",
            "author",
            "student id",
            "student number",
            "acknowledgment",
            "acknowledgement",
            "references",
            "appendix",
            "table of contents",
        )
        return any(marker in text for marker in non_core_markers)

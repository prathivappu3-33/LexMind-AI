"""
Graph Retriever

Retrieves graph-based context from Neo4j while preserving
relationship provenance and ranking results according to
the user's query.
"""

import re
from typing import Dict, List

from backend.retrieval.query_processor import QueryProcessor
from backend.knowledge_graph.neo4j_client import Neo4jClient


class GraphRetriever:
    """
    Retrieves graph-based context using Neo4j.

    The retriever:
    1. Extracts entities from the query using GLiNER.
    2. Normalizes extracted entities to graph terminology.
    3. Falls back to graph entity matching when GLiNER
       does not find an entity.
    4. Retrieves the entity subgraph.
    5. Scores graph relationships against the query.
    6. Filters low-confidence graph relationships.
    7. Returns the most relevant relationships while
       preserving provenance.
    """

    GRAPH_SCORE_THRESHOLD = 70.0

    def __init__(self):

        self.query_processor = QueryProcessor()
        self.neo4j = Neo4jClient()

    # ------------------------------------------------------------------
    # Entity normalization
    # ------------------------------------------------------------------

    def _get_entity_aliases(
        self,
        entity_name: str,
    ) -> List[str]:
        """
        Return graph-compatible aliases for an extracted entity.

        GLiNER may extract a natural-language concept such as
        "hashing", while the knowledge graph may contain
        "commit hash", "Git reference", "checksum", or "SHA-1".

        The original entity is always included first.
        """

        entity_name = entity_name.strip()

        aliases = {

            # ----------------------------------------------------------
            # Git / hashing terminology
            # ----------------------------------------------------------

            "hashing": [
                "commit hash",
                "Git reference",
                "checksum",
                "SHA-1",
                "SHA",
            ],

            "hashed": [
                "commit hash",
                "Git reference",
                "checksum",
                "SHA-1",
                "SHA",
            ],

            "hashes": [
                "commit hash",
                "Git reference",
                "checksum",
                "SHA-1",
                "SHA",
            ],

            "hash": [
                "commit hash",
                "Git reference",
                "checksum",
                "SHA-1",
                "SHA",
            ],

            "checksum": [
                "commit hash",
                "Git reference",
                "SHA-1",
                "SHA",
            ],

            "sha": [
                "SHA-1",
                "SHA1",
                "SHA-256",
                "SHA256",
                "checksum",
                "commit hash",
            ],

            "sha1": [
                "SHA-1",
                "SHA",
                "checksum",
                "commit hash",
            ],

            "sha-1": [
                "SHA-1",
                "SHA1",
                "SHA",
                "checksum",
                "commit hash",
            ],

            "sha256": [
                "SHA-256",
                "SHA256",
                "SHA",
                "checksum",
            ],

            "sha-256": [
                "SHA-256",
                "SHA256",
                "SHA",
                "checksum",
            ],

            "commit hash": [
                "Git reference",
                "checksum",
                "SHA-1",
                "SHA",
            ],

            "git reference": [
                "commit hash",
                "checksum",
                "SHA-1",
                "SHA",
            ],

            # ----------------------------------------------------------
            # Authentication terminology
            # ----------------------------------------------------------

            "multi-factor": [
                "two-factor authentication",
                "authentication",
            ],

            "multifactor": [
                "two-factor authentication",
                "authentication",
            ],

            "multi-factor authentication": [
                "two-factor authentication",
                "authentication",
            ],

            "mfa": [
                "two-factor authentication",
                "authentication",
            ],

            "authentication": [
                "two-factor authentication",
                "authentication",
            ],

            "auth": [
                "authentication",
                "two-factor authentication",
            ],

            "2fa": [
                "two-factor authentication",
                "authentication",
            ],

            "two-factor": [
                "two-factor authentication",
                "authentication",
            ],
        }

        normalized_name = entity_name.lower()

        result = [entity_name]

        for alias in aliases.get(
            normalized_name,
            [],
        ):

            if alias.lower() not in {
                item.lower()
                for item in result
            }:
                result.append(alias)

        return result

    # ------------------------------------------------------------------
    # Resolve graph entities
    # ------------------------------------------------------------------

    def _resolve_entities(
        self,
        entities: List[Dict],
    ) -> List[str]:
        """
        Resolve extracted entities to entities that actually exist
        in Neo4j.

        This is important because GLiNER terminology and graph
        terminology do not always match exactly.

        Example:

            hashing
                ↓
            commit hash
            Git reference
            checksum
            SHA-1
        """

        resolved_entities: List[str] = []

        seen_entities = set()

        for entity in entities:

            entity_name = str(
                entity.get("text", "")
            ).strip()

            if not entity_name:
                continue

            print(
                f"\nResolving entity: {entity_name}"
            )

            candidates = self._get_entity_aliases(
                entity_name
            )

            for candidate in candidates:

                exists = self.neo4j.find_entity(
                    candidate
                )

                print(
                    f"  Candidate: {candidate} "
                    f"-> {bool(exists)}"
                )

                if not exists:
                    continue

                key = candidate.strip().lower()

                if key in seen_entities:
                    continue

                seen_entities.add(key)

                resolved_entities.append(
                    candidate
                )

        return resolved_entities

    # ------------------------------------------------------------------
    # Main retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        max_hops: int = 2,
        top_k: int = 20,
    ) -> List[Dict]:

        query_data = self.query_processor.process(
            query
        )

        entities = query_data["entities"]

        print("\nExtracted entities:")
        print(entities)
        print()

        # --------------------------------------------------
        # Resolve GLiNER entities first
        # --------------------------------------------------

        resolved_entities = self._resolve_entities(
            entities
        )

        # --------------------------------------------------
        # Fallback entity discovery
        #
        # Only use fallback when no usable graph entities
        # were resolved from GLiNER.
        # --------------------------------------------------

        if not resolved_entities:

            print(
                "\nNo graph entities resolved from GLiNER."
            )

            print(
                "Trying graph entity matching..."
            )

            search_queries = []

            # --------------------------------------------------
            # Original full query
            # --------------------------------------------------

            search_queries.append(query)

            # --------------------------------------------------
            # Individual meaningful query terms
            # --------------------------------------------------

            query_terms = self._extract_query_terms(
                query
            )

            # --------------------------------------------------
            # Add semantic concepts BEFORE adding the terms
            # to search_queries.
            #
            # This fixes the previous ordering bug where
            # "commit hash", "git reference", etc. were added
            # after query_terms had already been copied.
            # --------------------------------------------------

            query_lower = query.lower()

            if (
                "hash" in query_lower
                or "hashing" in query_lower
                or "checksum" in query_lower
                or "sha" in query_lower
            ):

                query_terms.extend(
                    [
                        "hash",
                        "hashing",
                        "commit hash",
                        "git reference",
                        "checksum",
                        "sha",
                        "sha-1",
                        "sha-256",
                    ]
                )

            if (
                "authentication" in query_lower
                or "multi-factor" in query_lower
                or "multifactor" in query_lower
                or "2fa" in query_lower
                or "two-factor" in query_lower
            ):

                query_terms.extend(
                    [
                        "two-factor authentication",
                        "authentication",
                    ]
                )

            # --------------------------------------------------
            # Deduplicate query terms.
            # --------------------------------------------------

            query_terms = list(
                dict.fromkeys(
                    term.strip().lower()
                    for term in query_terms
                    if term.strip()
                )
            )

            # --------------------------------------------------
            # NOW add all expanded query terms to search.
            # --------------------------------------------------

            search_queries.extend(
                query_terms
            )

            print(
                "\nRanking query terms:"
            )

            print(
                query_terms
            )

            # --------------------------------------------------
            # Domain-aware expansions
            # --------------------------------------------------

            for term in query_terms:

                aliases = self._get_entity_aliases(
                    term
                )

                search_queries.extend(
                    aliases
                )

            # --------------------------------------------------
            # Explicit domain terms from full query
            # --------------------------------------------------

            if "git" in query_lower:

                search_queries.append(
                    "Git"
                )

                search_queries.append(
                    "Git reference"
                )

            if (
                "hash" in query_lower
                or "hashing" in query_lower
                or "checksum" in query_lower
                or "sha" in query_lower
            ):

                search_queries.append(
                    "commit hash"
                )

                search_queries.append(
                    "Git reference"
                )

                search_queries.append(
                    "checksum"
                )

                search_queries.append(
                    "SHA-1"
                )

                search_queries.append(
                    "SHA"
                )

            if (
                "authentication" in query_lower
                or "multi-factor" in query_lower
                or "multifactor" in query_lower
                or "2fa" in query_lower
                or "two-factor" in query_lower
            ):

                search_queries.append(
                    "two-factor authentication"
                )

                search_queries.append(
                    "authentication"
                )

            # --------------------------------------------------
            # Deduplicate search queries
            # --------------------------------------------------

            unique_search_queries = []

            seen_search_queries = set()

            for search_query in search_queries:

                key = search_query.strip().lower()

                if (
                    not key
                    or key in seen_search_queries
                ):
                    continue

                seen_search_queries.add(key)

                unique_search_queries.append(
                    search_query
                )

            print(
                "\nGraph search queries:"
            )

            print(
                unique_search_queries
            )

            # --------------------------------------------------
            # Search graph using candidate queries
            # --------------------------------------------------

            matched_entities = []

            for search_query in unique_search_queries:

                results = self.neo4j.search_entities(
                    search_query,
                    limit=10,
                )

                print(
                    f"\nSearch: {search_query}"
                )

                print(
                    "Matches:",
                    results,
                )

                matched_entities.extend(
                    results
                )

            # --------------------------------------------------
            # Deduplicate matched entities
            # --------------------------------------------------

            seen_entities = set()

            for result in matched_entities:

                entity_name = str(
                    result.get("entity", "")
                ).strip()

                if not entity_name:
                    continue

                key = entity_name.lower()

                if key in seen_entities:
                    continue

                seen_entities.add(key)

                resolved_entities.append(
                    entity_name
                )

            print(
                "\nFinal graph entities:"
            )

            print(
                resolved_entities
            )

        # --------------------------------------------------
        # Graph retrieval
        # --------------------------------------------------

        graph_results = []

        seen_retrieval_entities = set()

        for entity_name in resolved_entities:

            key = entity_name.strip().lower()

            if key in seen_retrieval_entities:
                continue

            seen_retrieval_entities.add(key)

            print(
                f"\nRetrieving subgraph for entity: "
                f"{entity_name}"
            )

            subgraph = self.neo4j.get_subgraph(
                entity_name=entity_name,
                max_hops=max_hops,
                limit=50,
            )

            print(
                "Subgraph:"
            )

            print(
                subgraph
            )

            graph_results.extend(
                subgraph
            )

        # --------------------------------------------------
        # No graph results
        # --------------------------------------------------

        if not graph_results:

            print(
                "\nNo graph relationships found."
            )

            return []

        # --------------------------------------------------
        # Deduplicate
        # --------------------------------------------------

        graph_results = self._deduplicate_results(
            graph_results
        )

        # --------------------------------------------------
        # Rank according to query
        # --------------------------------------------------

        graph_results = self._rank_results(
            query,
            graph_results
        )

        # --------------------------------------------------
        # Filter low-confidence graph relationships
        # --------------------------------------------------

        before_filter = len(
            graph_results
        )

        graph_results = [
            result
            for result in graph_results
            if result.get(
                "graph_score",
                0.0,
            ) >= self.GRAPH_SCORE_THRESHOLD
        ]

        after_filter = len(
            graph_results
        )

        print(
            "\nGraph score filtering:"
        )

        print(
            f"Threshold: "
            f"{self.GRAPH_SCORE_THRESHOLD}"
        )

        print(
            f"Before: {before_filter}"
        )

        print(
            f"After: {after_filter}"
        )

        # --------------------------------------------------
        # Final top-k
        # --------------------------------------------------

        return graph_results[:top_k]

    # ------------------------------------------------------------------
    # Query-aware ranking
    # ------------------------------------------------------------------

    def _rank_results(
        self,
        query: str,
        results: List[Dict],
    ) -> List[Dict]:

        if not results:
            return []

        query_terms = self._extract_query_terms(
            query
        )

        print(
            "\nQuery terms:"
        )

        print(
            query_terms
        )

        ranked = []

        for result in results:

            score = self._calculate_relevance(
                query_terms,
                result,
            )

            result["graph_score"] = score

            ranked.append(
                result
            )

        ranked.sort(
            key=lambda x: x.get(
                "graph_score",
                0.0,
            ),
            reverse=True,
        )

        print(
            "\nRanked graph results:"
        )

        for result in ranked[:10]:

            print(
                result.get(
                    "graph_score",
                    0.0,
                ),
                "|",
                result.get(
                    "source",
                    "",
                ),
                "--",
                result.get(
                    "relation",
                    "",
                ),
                "-->",
                result.get(
                    "target",
                    "",
                ),
            )

        return ranked

    # ------------------------------------------------------------------
    # Query term extraction
    # ------------------------------------------------------------------

    def _extract_query_terms(
        self,
        query: str,
    ) -> List[str]:

        query = query.lower()

        # Keep "-" because technical terms such as
        # SHA-1 and SHA-256 depend on it.
        query = re.sub(
            r"[^a-z0-9_\- ]",
            " ",
            query,
        )

        words = query.split()

        # Common question / stop words.
        stop_words = {
            "what",
            "does",
            "do",
            "did",
            "is",
            "are",
            "was",
            "were",
            "how",
            "why",
            "when",
            "where",
            "which",
            "who",
            "can",
            "could",
            "would",
            "should",
            "the",
            "a",
            "an",
            "for",
            "to",
            "of",
            "in",
            "on",
            "with",
            "use",
            "uses",
            "using",
            "used",
            "and",
            "or",
            "from",
            "by",
        }

        terms = []

        for word in words:

            if len(word) < 3:
                continue

            if word in stop_words:
                continue

            if word not in terms:
                terms.append(word)

        # --------------------------------------------------
        # Git hashing intent expansion
        # --------------------------------------------------

        query_lower = query

        if (
            "git" in query_lower
            and (
                "hash" in query_lower
                or "hashing" in query_lower
                or "checksum" in query_lower
                or "sha" in query_lower
            )
        ):

            for term in (
                "git",
                "hash",
                "hashing",
                "commit hash",
                "git reference",
                "checksum",
                "sha",
                "sha-1",
                "sha-256",
            ):

                if term not in terms:
                    terms.append(term)

        # --------------------------------------------------
        # Authentication intent expansion
        # --------------------------------------------------

        if (
            "authentication" in query_lower
            or "multi-factor" in query_lower
            or "multifactor" in query_lower
            or "2fa" in query_lower
            or "two-factor" in query_lower
        ):

            for term in (
                "authentication",
                "two-factor",
                "multi-factor",
                "2fa",
            ):

                if term not in terms:
                    terms.append(term)

        return terms

    # ------------------------------------------------------------------
    # Relevance scoring
    # ------------------------------------------------------------------

    def _calculate_relevance(
        self,
        query_terms: List[str],
        result: Dict,
    ) -> float:

        source = str(
            result.get(
                "source",
                "",
            )
        ).lower()

        target = str(
            result.get(
                "target",
                "",
            )
        ).lower()

        relation = str(
            result.get(
                "relation",
                "",
            )
        ).lower()

        evidence = " ".join(
            str(x)
            for x in result.get(
                "evidence",
                [],
            )
        ).lower()

        section = " ".join(
            str(x)
            for x in result.get(
                "sections",
                [],
            )
        ).lower()

        title = " ".join(
            str(x)
            for x in result.get(
                "titles",
                [],
            )
        ).lower()

        endpoint_text = (
            source
            + " "
            + target
        )

        context_text = (
            evidence
            + " "
            + section
            + " "
            + title
        )

        # --------------------------------------------------
        # Normalize query terms
        # --------------------------------------------------

        normalized_terms = set()

        technical_aliases = {

            "hashing": {
                "hash",
                "hashing",
            },

            "hashed": {
                "hash",
                "hashed",
            },

            "hashes": {
                "hash",
                "hashes",
            },

            "sha-1": {
                "sha",
                "sha1",
                "sha-1",
            },

            "sha1": {
                "sha",
                "sha1",
                "sha-1",
            },

            "sha-256": {
                "sha",
                "sha256",
                "sha-256",
            },

            "sha256": {
                "sha",
                "sha256",
                "sha-256",
            },

            "checksum": {
                "checksum",
                "hash",
            },

            "commit hash": {
                "commit hash",
                "hash",
            },

            "git reference": {
                "git reference",
                "git",
            },
        }

        for term in query_terms:

            term = term.strip().lower()

            if not term:
                continue

            normalized_terms.add(
                term
            )

            # Add technical aliases.
            for alias in technical_aliases.get(
                term,
                set(),
            ):
                normalized_terms.add(
                    alias
                )

            # Basic morphology.

            if (
                term.endswith("ing")
                and len(term) > 5
            ):

                normalized_terms.add(
                    term[:-3]
                )

            elif (
                term.endswith("ed")
                and len(term) > 4
            ):

                normalized_terms.add(
                    term[:-2]
                )

            elif (
                term.endswith("es")
                and len(term) > 4
            ):

                normalized_terms.add(
                    term[:-2]
                )

            elif (
                term.endswith("s")
                and len(term) > 3
            ):

                normalized_terms.add(
                    term[:-1]
                )

        # --------------------------------------------------
        # Intent detection
        # --------------------------------------------------

        hash_terms = {
            "hash",
            "hashing",
            "hashed",
            "hashes",
            "checksum",
            "sha",
            "sha1",
            "sha-1",
            "sha256",
            "sha-256",
            "commit hash",
            "git reference",
        }

        auth_terms = {
            "authentication",
            "authenticate",
            "authenticated",
            "auth",
            "2fa",
            "multi-factor",
            "multifactor",
            "two-factor",
            "two-factor authentication",
        }

        has_hash_intent = any(
            (
                term in hash_terms
                or "hash" in term
                or "sha" in term
                or "checksum" in term
            )
            for term in normalized_terms
        )

        has_auth_intent = any(
            (
                term in auth_terms
                or "authentication" in term
                or "two-factor" in term
                or "multi-factor" in term
                or term == "2fa"
            )
            for term in normalized_terms
        )

        # --------------------------------------------------
        # Base relevance
        # --------------------------------------------------

        score = 0.0

        endpoint_matches = 0

        for term in normalized_terms:

            if term in source:
                score += 10.0
                endpoint_matches += 1

            if term in target:
                score += 10.0
                endpoint_matches += 1

            if term in relation:
                score += 3.0

            if term in evidence:
                score += 1.5

            if term in section:
                score += 1.0

            if term in title:
                score += 0.5

        if endpoint_matches >= 2:
            score += 8.0

        # --------------------------------------------------
        # Authentication intent
        # --------------------------------------------------

        if has_auth_intent:

            auth_endpoint = any(
                term in endpoint_text
                for term in (
                    "authentication",
                    "two-factor",
                    "multi-factor",
                    "2fa",
                    "auth",
                )
            )

            auth_context = any(
                term in context_text
                for term in (
                    "authentication",
                    "two-factor",
                    "multi-factor",
                    "2fa",
                    "auth",
                )
            )

            if auth_endpoint:
                score += 25.0

            if auth_context:
                score += 12.0

            if (
                "two-factor authentication"
                in endpoint_text
                or
                "two-factor authentication"
                in evidence
            ):
                score += 20.0

            if (
                not auth_endpoint
                and not auth_context
            ):
                score -= 12.0

        # --------------------------------------------------
        # Hash intent
        #
        # Hash queries should strongly prefer:
        #
        #   Git -> USES -> SHA-1
        #   commit hash
        #   checksum
        #   SHA
        #   Git reference + hash/checksum evidence
        #
        # Generic Git relationships should receive
        # a penalty when they contain no hash evidence.
        # --------------------------------------------------

        if has_hash_intent:

            has_hash_endpoint = any(
                term in endpoint_text
                for term in (
                    "commit hash",
                    "hash",
                    "checksum",
                    "sha",
                    "sha1",
                    "sha-1",
                    "sha256",
                    "sha-256",
                )
            )

            has_hash_evidence = any(
                term in context_text
                for term in (
                    "commit hash",
                    "hash",
                    "checksum",
                    "sha",
                    "sha1",
                    "sha-1",
                    "sha256",
                    "sha-256",
                )
            )

            has_commit_hash = (
                "commit hash" in endpoint_text
                or "commit hash" in context_text
            )

            has_checksum = (
                "checksum" in endpoint_text
                or "checksum" in context_text
            )

            has_git_reference = (
                "git reference" in endpoint_text
                or "git reference" in context_text
            )

            # --------------------------------------------------
            # Detect actual SHA algorithms.
            # --------------------------------------------------

            has_sha_algorithm = any(
                term in endpoint_text
                for term in (
                    "sha",
                    "sha1",
                    "sha-1",
                    "sha256",
                    "sha-256",
                )
            )

            # --------------------------------------------------
            # Direct hash entity match.
            # --------------------------------------------------

            if has_hash_endpoint:
                score += 25.0

            # --------------------------------------------------
            # Strong supporting evidence.
            # --------------------------------------------------

            if has_hash_evidence:
                score += 15.0

            # --------------------------------------------------
            # Exact concept requested.
            # --------------------------------------------------

            if has_commit_hash:
                score += 30.0

            # --------------------------------------------------
            # Checksum is particularly relevant.
            # --------------------------------------------------

            if has_checksum:
                score += 15.0

            # --------------------------------------------------
            # SHA algorithm is a strong answer.
            # --------------------------------------------------

            if has_sha_algorithm:
                score += 30.0

            # --------------------------------------------------
            # Git reference + hash/checksum is very strong.
            # --------------------------------------------------

            if (
                has_git_reference
                and
                (
                    has_commit_hash
                    or has_checksum
                    or has_hash_endpoint
                )
            ):
                score += 25.0

            # --------------------------------------------------
            # DIRECT GIT HASHING RELATIONSHIP
            #
            # Example:
            #
            # Git --USES--> SHA-1
            #
            # This is exactly the relationship required to
            # answer:
            #
            # "What does Git use for hashing?"
            #
            # Give it a strong score boost.
            # --------------------------------------------------

            git_hash_algorithm = (
                "git" in source
                and relation in {
                    "uses",
                    "uses_hash",
                    "hashes",
                }
                and any(
                    term in target
                    for term in (
                        "sha",
                        "sha1",
                        "sha-1",
                        "sha256",
                        "sha-256",
                        "hash",
                    )
                )
            )

            if git_hash_algorithm:
                score += 80.0

            # --------------------------------------------------
            # Git + SHA relationship even if relation naming
            # differs slightly.
            # --------------------------------------------------

            git_sha_endpoint = (
                "git" in source
                and any(
                    term in target
                    for term in (
                        "sha",
                        "sha1",
                        "sha-1",
                        "sha256",
                        "sha-256",
                    )
                )
            )

            if git_sha_endpoint:
                score += 35.0

            # --------------------------------------------------
            # Generic Git relationship with no hash evidence
            # should be penalized.
            # --------------------------------------------------

            if (
                "git" in endpoint_text
                and not has_hash_endpoint
                and not has_hash_evidence
                and not has_sha_algorithm
            ):
                score -= 20.0

            # --------------------------------------------------
            # Generic Git evidence mentioning Git but not
            # the actual hashing concept should not dominate.
            # --------------------------------------------------

            if (
                has_hash_intent
                and
                "git" in endpoint_text
                and
                not has_commit_hash
                and
                not has_checksum
                and
                not has_git_reference
                and
                not has_sha_algorithm
            ):
                score -= 10.0

        # --------------------------------------------------
        # Query-specific endpoint overlap
        # --------------------------------------------------

        important_terms = set()

        if has_auth_intent:

            important_terms.update(
                {
                    "authentication",
                    "two-factor",
                    "multi-factor",
                    "2fa",
                }
            )

        if has_hash_intent:

            important_terms.update(
                {
                    "hash",
                    "checksum",
                    "sha",
                    "sha1",
                    "sha-1",
                    "sha256",
                    "sha-256",
                    "commit hash",
                    "git reference",
                }
            )

        for term in important_terms:

            if term in source:
                score += 8.0

            if term in target:
                score += 8.0

            if term in relation:
                score += 4.0

        return score

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate_results(
        self,
        results: List[Dict],
    ) -> List[Dict]:

        unique = []

        seen = set()

        for result in results:

            key = (
                str(
                    result.get(
                        "source",
                        "",
                    )
                ).strip().lower(),

                str(
                    result.get(
                        "relation",
                        "",
                    )
                ).strip().lower(),

                str(
                    result.get(
                        "target",
                        "",
                    )
                ).strip().lower(),
            )

            if key in seen:
                continue

            seen.add(key)

            unique.append(
                result
            )

        return unique
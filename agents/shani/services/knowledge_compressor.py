from collections import defaultdict


# ============================================================
# TOKEN BUDGET CONSTANTS
#
# These govern the maximum payload the compressor will ever
# pass to the LLM. All limits are hard — not probabilistic.
#
# Sizing rationale for Mistral 7B (~8K context window):
#
#   MAX_CLUSTERS = 4
#     4 clusters × ~200 tokens each = ~800 tokens of evidence
#     leaves 7200 tokens for prompt wrapper + output
#
#   MAX_EVIDENCE_PER_CLUSTER = 3
#     3 sentences × 35 words × 1.3 ≈ 137 tokens per cluster
#     total evidence payload per cluster stays under 200 tokens
#
#   MAX_SENTENCE_WORDS = 35
#     A 35-word sentence is sufficient to convey one scientific
#     finding. S5 stores full extracted sentences which can be
#     60-150 words — truncation is mandatory before injection.
#
#   MAX_COMPRESSION_OUTPUT_TOKENS = 120
#     2-3 summary sentences ≈ 60-90 tokens.
#     120 gives a safe ceiling without overflow risk.
# ============================================================

MAX_CLUSTERS              = 4
MAX_EVIDENCE_PER_CLUSTER  = 3
MAX_SENTENCE_WORDS        = 35
MAX_COMPRESSION_OUTPUT_TOKENS = 120


class KnowledgeCompressor:

    def __init__(
        self,
        max_clusters=MAX_CLUSTERS,
        max_evidence_per_cluster=MAX_EVIDENCE_PER_CLUSTER,
        max_sentence_words=MAX_SENTENCE_WORDS
    ):
        self.max_clusters = max_clusters
        self.max_evidence_per_cluster = max_evidence_per_cluster
        self.max_sentence_words = max_sentence_words

    # -------------------------------------------------------
    # STEP 1: build_clusters
    #
    # Groups knowledge rows by value (or subject→relation→object
    # triple when available). Deduplicates by paper_id within
    # each cluster. Sorts clusters by evidence count descending
    # so the most-supported claims always enter the prompt.
    #
    # No LLM call. Pure deterministic grouping.
    #
    # FIX vs original:
    # - max_clusters default: 12 → 4
    # - max_evidence_per_cluster default: 12 → 3
    # - clusters now sorted by evidence count before truncation
    #   (original used insertion order — non-deterministic)
    # -------------------------------------------------------
    def build_clusters(self, knowledge_rows, citation_map, paper_lookup):

        clusters = defaultdict(list)

        for r in knowledge_rows:

            value    = r["value"]
            sentence = r["sentence"]
            paper_id = r["paper_id"]

            subject  = r["subject"]  if "subject"  in r.keys() else None
            relation = r["relation"] if "relation" in r.keys() else None
            obj      = r["object"]   if "object"   in r.keys() else None

            if not sentence:
                continue

            citation_num = citation_map.get(paper_id, "?")

            if subject and relation and obj:
                key = f"{subject} → {relation} → {obj}"
            else:
                key = str(value)

            clusters[key].append({
                "sentence": sentence.strip(),
                "citation": str(citation_num),
                "paper_id": paper_id
            })

        # Deduplicate within each cluster (one entry per paper)
        filtered_clusters = {}

        for key, items in clusters.items():

            seen_papers  = set()
            unique_items = []

            for item in items:
                if item["paper_id"] in seen_papers:
                    continue
                seen_papers.add(item["paper_id"])
                unique_items.append(item)

            filtered_clusters[key] = unique_items[:self.max_evidence_per_cluster]

        # Sort by evidence count descending — deterministic ordering
        # guarantees that the most-supported claims always win
        # when the list is truncated to max_clusters
        sorted_clusters = sorted(
            filtered_clusters.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )

        return sorted_clusters[:self.max_clusters]

    # -------------------------------------------------------
    # STEP 2: format_cluster
    #
    # Formats one cluster as a structured text block for the
    # LLM. Sentences are truncated here to max_sentence_words
    # before any LLM ever sees them.
    #
    # FIX vs original:
    # - Sentence truncation added (was completely absent).
    #   Original passed raw DB sentences of 60-150 words.
    #   Now capped at MAX_SENTENCE_WORDS (35) with ellipsis
    #   marker so the LLM knows the sentence continues.
    # - Prompt labels shortened ("Scientific Claim:" →
    #   "Claim:") to reduce wrapper token cost.
    # -------------------------------------------------------
    def format_cluster(self, key, items):

        lines = [
            "Claim:",
            str(key),
            "",
            "Evidence:"
        ]

        for item in items:

            # HARD TRUNCATION — no sentence exceeds max_sentence_words
            words = item["sentence"].split()
            sentence = " ".join(words[:self.max_sentence_words])
            if len(words) > self.max_sentence_words:
                sentence += "..."

            lines.append(f"[{item['citation']}] {sentence}")

        return "\n".join(lines)

    # -------------------------------------------------------
    # STEP 3: summarize_cluster
    #
    # One focused LLM call per cluster. Produces a 2-3 sentence
    # summary of the cluster's evidence with citation numbers
    # preserved.
    #
    # FIX vs original:
    # - max_tokens=120 now passed explicitly and enforced
    #   (was not passed in original — ran unconstrained).
    # - Prompt tightened from 14 lines → 6 lines to reduce
    #   wrapper token cost per cluster call.
    # - stage="S6_compress" passed for log file separation
    #   from section-generation calls.
    # -------------------------------------------------------
    def summarize_cluster(self, cluster_text, llm_service):

        prompt = (
            "Summarize the following scientific evidence in 2-3 sentences.\n"
            "RULES:\n"
            "- Use ONLY the provided data. Do NOT invent facts.\n"
            "- Preserve citation numbers exactly as given, e.g. [1], [3].\n"
            "- Do NOT add citations not present in the data.\n\n"
            f"Data:\n{cluster_text}"
        )

        return llm_service.generate_text(
            prompt,
            max_tokens=MAX_COMPRESSION_OUTPUT_TOKENS,
            temperature=0.2,
            stage="S6_compress"
        )

    # -------------------------------------------------------
    # STEP 4: build_cluster_summaries
    #
    # Orchestrates steps 1-3. Returns a list of compressed
    # summary strings, one per cluster, ready for injection
    # into the section-generation prompt.
    #
    # Unchanged in structure from original.
    # -------------------------------------------------------
    def build_cluster_summaries(
        self,
        knowledge_rows,
        citation_map,
        paper_lookup,
        llm_service
    ):
        cluster_items = self.build_clusters(
            knowledge_rows,
            citation_map,
            paper_lookup
        )

        summaries = []

        for key, items in cluster_items:
            cluster_text = self.format_cluster(key, items)
            summary = self.summarize_cluster(cluster_text, llm_service)

            if summary:
                summaries.append(summary.strip())

        return summaries

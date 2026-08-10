# ============================================================
# search_papers.py  — S2
#
# ROLE IN PIPELINE:
#   Wide retrieval only. This module's single responsibility is
#   to collect the largest possible candidate pool from all
#   three external sources (Semantic Scholar, OpenAlex, arXiv)
#   and hand it to paper_ingestor for scoring and ranked
#   insertion.
#
# WHAT CHANGED vs previous version:
#   - compute_score() and _PARAM_BOOST_TERMS removed entirely.
#     All scoring now lives in paper_ingestor.py.
#   - Direct pr.create_paper() calls replaced with a single
#     call to ingest_search_results(repo, workflow_id, raw_pool,
#     config=config, limit=FINAL_PAPER_LIMIT).
#   - MIN_RELEVANCE_SCORE gate removed — ingestor decides what
#     to keep after scoring.
#   - Config is fetched here and passed to ingestor so material/
#     synthesis/focus data is available for scoring without a
#     second DB round-trip.
#
# WHAT IS UNCHANGED:
#   - All three source helpers (search_semantic_scholar,
#     search_openalex, search_arxiv).
#   - OpenAlex abstract reconstruction.
#   - Safe request wrapper with 429 retry.
#   - Retrieval volume constants.
#   - Query generation via generate_queries().
# ============================================================

import requests
import time

import repositories.workflow_repo as wr

from tools.search_arxiv import search_arxiv
from tools.generate_queries import generate_queries
from tools.paper_ingestor import ingest_search_results


# =========================================================
# RETRIEVAL LIMITS
# =========================================================

MAX_RESULTS_PER_SOURCE = 100
FINAL_PAPER_LIMIT      = 500
MIN_PAPERS             = 15
MAX_QUERIES            = 15

STOPWORDS = {
    "of", "and", "in", "on", "for", "with",
    "can", "the", "a", "an", "moreover"
}


# =========================================================
# SAFE REQUEST
# =========================================================

def safe_request(url, params):
    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 429:
            time.sleep(2)
            response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            return None

        return response.json()

    except Exception as e:
        print(f"[S2] Request error: {e}")
        return None


# =========================================================
# OPENALEX ABSTRACT RECONSTRUCTION
# =========================================================

def parse_openalex_abstract(inverted_index):
    if not inverted_index:
        return ""

    words = []
    for word, positions in inverted_index.items():
        for pos in positions:
            words.append((pos, word))

    words.sort()
    return " ".join([w for _, w in words])


# =========================================================
# SOURCE SEARCHES
# =========================================================

def search_semantic_scholar(query):

    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query":  query,
        "limit":  MAX_RESULTS_PER_SOURCE,
        "fields": "title,abstract,openAccessPdf,year,externalIds"
    }

    results = []
    data = safe_request(url, params)
    if not data:
        return results

    for paper in data.get("data", []):
        pdf = paper.get("openAccessPdf")
        external_ids = paper.get("externalIds") or {}
        doi = external_ids.get("DOI")

        results.append({
            "title":   paper.get("title"),
            "summary": paper.get("abstract") or "",
            "pdf_url": pdf.get("url") if pdf else None,
            "year":    paper.get("year"),
            "source":  "semantic_scholar",
            "doi":     doi
        })

    return results


def search_openalex(query):

    url = "https://api.openalex.org/works"
    params = {
        "search":   query,
        "per_page": MAX_RESULTS_PER_SOURCE
    }

    results = []
    data = safe_request(url, params)
    if not data:
        return results

    for work in data.get("results", []):
        pdf_url = None
        if work.get("primary_location"):
            pdf_url = work["primary_location"].get("pdf_url")

        abstract = parse_openalex_abstract(
            work.get("abstract_inverted_index")
        )

        doi = work.get("doi")
        if doi and doi.startswith("https://doi.org/"):
            doi = doi[len("https://doi.org/"):]

        results.append({
            "title":   work.get("title"),
            "summary": abstract,
            "pdf_url": pdf_url,
            "year":    work.get("publication_year"),
            "source":  "openalex",
            "doi":     doi
        })

    return results


# =========================================================
# FETCH FROM ALL SOURCES
# =========================================================

def fetch_from_sources(query):

    results = []
    sources = [
        ("arxiv",            search_arxiv),
        ("semantic_scholar", search_semantic_scholar),
        ("openalex",         search_openalex),
    ]

    for name, func in sources:
        try:
            res = func(query)
            if res:
                results.extend(res)
            print(f"[S2]   {name}: {len(res)} papers")
        except Exception as e:
            print(f"[S2 WARNING] {name} failed: {e}")

    return results


# =========================================================
# MAIN TOOL — S2
# =========================================================

def search_papers(repo, workflow_id, execution_attempt_id=None, **kwargs):

    workflow = wr.get_workflow(repo, workflow_id)
    if not workflow:
        return {"status": "error", "data": None, "error": "Workflow not found"}

    # --------------------------------------------------
    # FETCH CONFIG — passed to ingestor for scoring
    # --------------------------------------------------
    config = repo.fetch_one(
        """
        SELECT material, structure, focus, method, properties, characterization, max_papers
        FROM WorkflowResearchConfig
        WHERE workflow_id = ?
        """,
        (workflow_id,)
    )

    config_dict = dict(config) if config else {}

    # --------------------------------------------------
    # GENERATE QUERIES
    # --------------------------------------------------
    query_result = generate_queries(repo, workflow_id)
    if query_result["status"] != "success":
        return query_result

    all_queries = list(dict.fromkeys(query_result["data"]))[:MAX_QUERIES]

    # --------------------------------------------------
    # WIDE RETRIEVAL — collect all candidates
    # --------------------------------------------------
    raw_pool = []

    for query in all_queries:
        print(f"[S2] Searching: {query}")
        results = fetch_from_sources(query)
        for p in results:
            if p.get("title"):
                raw_pool.append(p)

    print(f"\n[S2] Raw pool: {len(raw_pool)} candidates before scoring")

    if len(raw_pool) < MIN_PAPERS:
        return {
            "status": "error",
            "data":   None,
            "error":  f"Insufficient papers found ({len(raw_pool)})"
        }

    # --------------------------------------------------
    # DELEGATE scoring, dedup, ranking, and insertion
    # to paper_ingestor — the single scoring authority.
    # --------------------------------------------------
    effective_limit = config_dict.get("max_papers") or FINAL_PAPER_LIMIT
    inserted_ids = ingest_search_results(
        repo,
        workflow_id,
        raw_pool,
        config=config_dict,
        limit=effective_limit
    )

    print(f"[S2] Inserted: {len(inserted_ids)} ranked papers (cap={effective_limit})")

    return {"status": "success", "data": inserted_ids, "error": None}

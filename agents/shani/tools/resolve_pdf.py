import os
import json
import requests

from urllib.parse import urlparse, urlunparse


# ============================================================
# UNPAYWALL CONFIG
#
# FIX [10]: EMAIL was hardcoded as a placeholder string
# "your_real_email@gmail.com". Unpaywall requires a real
# contact email in every API request per their terms of
# service. A placeholder causes Unpaywall to eventually
# rate-limit or block all requests from this system.
#
# Fix: EMAIL is now read from the UNPAYWALL_EMAIL environment
# variable. If not set, Unpaywall lookups are skipped
# entirely with a clear warning — the pipeline continues
# using arXiv and original URL candidates only.
#
# To enable Unpaywall: set the environment variable before
# running SHANI:
#   export UNPAYWALL_EMAIL=your@email.com
# ============================================================

UNPAYWALL_API = "https://api.unpaywall.org/v2/"
EMAIL = os.environ.get("UNPAYWALL_EMAIL", "").strip()

if not EMAIL:
    print("[WARN] UNPAYWALL_EMAIL environment variable not set.")
    print("[WARN] Unpaywall PDF lookups will be skipped.")
    print("[WARN] Set: export UNPAYWALL_EMAIL=your@email.com")


# ============================================================
# URL NORMALIZATION
# ============================================================

def normalize_url(url):
    try:
        parsed = urlparse(url)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
    except:
        return url


# ============================================================
# URL VALIDATION
# ============================================================

def is_valid_url(url):
    return isinstance(url, str) and url.startswith("http")


# ============================================================
# UNPAYWALL LOOKUP
#
# Skipped entirely if EMAIL is not configured.
# ============================================================

def get_unpaywall_pdf(doi):

    if not EMAIL:
        return None

    try:
        url = f"{UNPAYWALL_API}{doi}"
        params = {"email": EMAIL}

        res = requests.get(url, params=params, timeout=10)

        if res.status_code != 200:
            return None

        data = res.json()
        loc = data.get("best_oa_location")

        if loc and loc.get("url_for_pdf"):
            return loc["url_for_pdf"]

    except Exception as e:
        print(f"[WARN] Unpaywall error: {e}")

    return None


# ============================================================
# ARXIV LOOKUP
# Unchanged — correct as written.
# ============================================================

def lookup_arxiv(doi=None, title=None):
    try:
        query = None

        if doi:
            query = f"doi:{doi}"
        elif title:
            query = title

        if not query:
            return None

        url = "http://export.arxiv.org/api/query"
        params = {"search_query": query, "max_results": 1}

        res = requests.get(url, params=params, timeout=10)

        if res.status_code != 200:
            return None

        if "<id>http://arxiv.org/abs/" in res.text:
            start = res.text.find("<id>http://arxiv.org/abs/")
            end = res.text.find("</id>", start)

            abs_url = res.text[start + 4:end]
            return abs_url.replace("/abs/", "/pdf/") + ".pdf"

    except Exception as e:
        print(f"[WARN] arXiv lookup error: {e}")

    return None


# ============================================================
# MAIN TOOL — S2_5
#
# Enriches papers that have metadata but no confirmed PDF.
# For each paper with pdf_status='metadata':
#   1. Try original pdf_url from S2
#   2. Try Unpaywall (if EMAIL configured)
#   3. Try arXiv
#   4. Deduplicate candidates by normalized URL
#   5. Store candidates JSON + set pdf_status='enriched'
#      or 'metadata_only' if no candidates found
#
# S3 (download_papers) then reads pdf_candidates and tries
# each in priority order.
# ============================================================

def resolve_pdf(repo, workflow_id, execution_attempt_id=None, **kwargs):

    print("\n[RESOLVE_PDF] Starting candidate enrichment...")

    papers = repo.fetch_all(
        """
        SELECT id, title, pdf_url, doi
        FROM Paper
        WHERE workflow_id = ?
        AND pdf_status = 'metadata'
        """,
        (workflow_id,)
    )

    total = len(papers)
    print(f"[RESOLVE_PDF] Found {total} papers")

    for idx, p in enumerate(papers, 1):

        paper_id     = p["id"]
        title        = p["title"]
        original_url = p["pdf_url"] if "pdf_url" in p.keys() else None
        doi          = p["doi"]     if "doi"     in p.keys() else None

        print(f"\n[{idx}/{total}] Processing: {title[:60]}...")

        candidates = []

        # 1. ORIGINAL URL
        if is_valid_url(original_url):
            candidates.append({
                "source":   "original",
                "url":      original_url,
                "priority": 5
            })

        # 2. UNPAYWALL (skipped silently if EMAIL not set)
        if doi:
            up_url = get_unpaywall_pdf(doi)
            if is_valid_url(up_url):
                candidates.append({
                    "source":   "unpaywall",
                    "url":      up_url,
                    "priority": 1
                })

        # 3. ARXIV
        arxiv_url = lookup_arxiv(doi=doi, title=title)
        if is_valid_url(arxiv_url):
            candidates.append({
                "source":   "arxiv",
                "url":      arxiv_url,
                "priority": 2
            })

        # 4. DEDUPLICATE by normalized URL, keep highest priority
        url_map = {}

        for c in candidates:
            if not is_valid_url(c["url"]):
                continue

            norm = normalize_url(c["url"])

            if norm not in url_map or c["priority"] < url_map[norm]["priority"]:
                url_map[norm] = c

        unique_candidates = sorted(
            url_map.values(),
            key=lambda x: x["priority"]
        )[:5]

        print(f"[CANDIDATES] {len(unique_candidates)} found")

        # 5. STORE RESULT
        if not unique_candidates:
            print("[SKIP] No valid candidates")

            with repo.transaction() as cursor:
                cursor.execute(
                    """
                    UPDATE Paper
                    SET pdf_status = 'metadata_only'
                    WHERE id = ?
                    """,
                    (paper_id,)
                )
            continue

        with repo.transaction() as cursor:
            cursor.execute(
                """
                UPDATE Paper
                SET pdf_candidates = ?, pdf_status = 'enriched'
                WHERE id = ?
                """,
                (json.dumps(unique_candidates), paper_id)
            )

    print("\n[RESOLVE_PDF] Completed")

    return {"status": "success", "data": total, "error": None}

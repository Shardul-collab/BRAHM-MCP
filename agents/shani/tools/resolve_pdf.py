import os
import re
import json
import difflib
import requests

from urllib.parse import urlparse, urlunparse

# ============================================================
# ARXIV TITLE-MATCH THRESHOLD
#
# FIX (2026-09-09): lookup_arxiv() ran a free-text arXiv search on the
# paper's title, took the FIRST result unconditionally, and never compared
# what came back to what was asked for. The function carried the comment
# "Unchanged — correct as written."
#
# It was not. In workflow 2 it returned arXiv 1411.4413 — "Observation of
# the rare B0s -> mu+mu- decay from the combined analysis of CMS and LHCb
# data", a CERN particle-physics paper — as the match for BOTH:
#     "Formation of Beta-Indium Selenide Layers Grown via Selenium
#      Passivation of InP(111)B Substrate"          (paper 6)
#     "Molecular Beam Epitaxy of Twin-Free Bi2Se3 and Sb2Te3 on
#      In2Se3/InP(111)"                             (paper 13)
#
# Both then DOWNLOADED it, because the unverified arXiv guess was given
# priority 2 while each paper's own known-good URL sat at priority 5, and
# download_papers tries candidates in ascending priority order. 6.pdf and
# 13.pdf on disk are byte-identical (md5 bf3ce1f1...), and S4 extracted the
# LHCb author list from both.
#
# Consequences measured on the live corpus:
#   - paper 6 alone contributed 635 knowledge rows: 58% of the whole
#     corpus and 87% of the material axis, every value an author surname
#     carrying an affiliation index ('Bediaga1', 'Miranda1', 'Gomes1')
#   - papers 6 + 13 together consumed 152,332 of S5's 472,807 input
#     characters — 32% of the extraction budget — on the wrong paper
#
# Two changes below: scope the query to the title field, and REQUIRE the
# returned title to actually match before the URL is offered as a candidate.
# ============================================================

ARXIV_TITLE_MATCH_THRESHOLD = 0.75


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

def _normalise_title(t):
    """Lowercase, strip markup and punctuation, collapse whitespace."""
    if not t:
        return ""
    # Subscript/superscript tags are removed WITHOUT leaving a space, so a
    # publisher's "Mn<sub>2</sub>In<sub>2</sub>Se<sub>5</sub>" normalises to
    # the same string as arXiv's plain "Mn2In2Se5". Replacing them with a
    # space instead splits the formula ('mn 2 in 2 se 5') and needlessly
    # depresses the similarity score for a paper that is in fact the same.
    t = re.sub(r"</?(?:sub|sup)>", "", t, flags=re.IGNORECASE)
    t = re.sub(r"<[^>]+>", " ", t)          # any other markup: treat as a break
    t = t.lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def titles_match(requested, returned, threshold=ARXIV_TITLE_MATCH_THRESHOLD):
    """
    Is `returned` plausibly the same paper as `requested`?

    Uses a similarity ratio rather than equality because arXiv titles differ
    from publisher titles in punctuation, subscript markup and line breaks.
    The threshold only has to separate 'same paper, formatted differently'
    from 'completely different paper', which is a wide gap: the LHCb title
    scores 0.19 against the two In2Se3 titles it was returned for.
    """
    a, b = _normalise_title(requested), _normalise_title(returned)
    if not a or not b:
        return False
    if a == b:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= threshold


def _parse_arxiv_entry(xml):
    """Pull (abs_url, title) out of the first <entry> of an arXiv Atom feed."""
    entry_start = xml.find("<entry>")
    if entry_start == -1:
        return None, None
    entry = xml[entry_start:xml.find("</entry>", entry_start)]

    id_m = re.search(r"<id>(http://arxiv\.org/abs/[^<]+)</id>", entry)
    ti_m = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
    return (id_m.group(1) if id_m else None,
            (ti_m.group(1).strip() if ti_m else None))


def lookup_arxiv(doi=None, title=None):
    """
    Find an arXiv PDF for a paper.

    Returns (url, match_method) where match_method is 'doi' or 'title', or
    (None, None). The caller uses the method to decide how much to trust it
    — see the priority assignment in resolve_pdf().

    A title-based hit is only returned when the arXiv title actually matches
    the requested one. Previously any top hit was accepted; see the note at
    the top of this module for what that cost.
    """
    try:
        if doi:
            query = f"doi:{doi}"
            method = "doi"
        elif title:
            # Scope to the title field. A bare free-text search matches
            # abstracts and author names too, which is how an LHCb paper
            # came back for an indium selenide title.
            cleaned = _normalise_title(title)
            if not cleaned:
                return None, None
            query = f'ti:"{cleaned}"'
            method = "title"
        else:
            return None, None

        res = requests.get(
            "http://export.arxiv.org/api/query",
            params={"search_query": query, "max_results": 5},
            timeout=10,
        )
        if res.status_code != 200:
            return None, None

        abs_url, arxiv_title = _parse_arxiv_entry(res.text)
        if not abs_url:
            return None, None

        # A DOI query is exact; a title query is not and must be verified.
        if method == "title":
            if not titles_match(title, arxiv_title):
                print(f"[ARXIV] rejected mismatch for {title[:55]!r}")
                print(f"[ARXIV]   arXiv returned {str(arxiv_title)[:70]!r}")
                return None, None

        return abs_url.replace("/abs/", "/pdf/") + ".pdf", method

    except Exception as e:
        print(f"[WARN] arXiv lookup error: {e}")

    return None, None


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
        #
        # Priority now depends on how the match was made. A DOI lookup is
        # exact and stays ahead of the original URL. A title lookup is a
        # similarity judgement, so even after verification it sits BEHIND
        # the original URL (priority 5) rather than in front of it.
        #
        # This ordering is the second half of the paper-6 bug: an unverified
        # title guess at priority 2 was tried before each paper's own known
        # -good link at priority 5, so the correct URL was never fetched at
        # all. Verification alone would have fixed the wrong content; this
        # also makes the known-good source win when both are available.
        arxiv_url, arxiv_method = lookup_arxiv(doi=doi, title=title)
        if is_valid_url(arxiv_url):
            candidates.append({
                "source":   f"arxiv:{arxiv_method}",
                "url":      arxiv_url,
                "priority": 2 if arxiv_method == "doi" else 6
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

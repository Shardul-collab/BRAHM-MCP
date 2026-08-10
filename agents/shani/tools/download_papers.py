import os
import json
import re
import requests

import repositories.paper_repo as pr
import repositories.failure_repo as failure_repo

PAPERS_DIR = "papers"
MAX_RETRIES = 2
MIN_FILE_SIZE_KB = 50

BLOCKED_DOMAINS = ["academic.oup.com", "sciencedirect.com"]

# CHITRAGUPTA deduplication — endpoint on localhost:8003
CHITRAGUPTA_BASE = "http://localhost:8003"
CHITRAGUPTA_TIMEOUT = 5   # seconds — never block the pipeline on this


# =========================================================
# CHITRAGUPTA HELPERS
# =========================================================

def _chit_check(doi: str | None, title: str | None) -> dict | None:
    """
    Returns CHITRAGUPTA response dict if reachable, None if API is down.
    Never raises — dedup failure must not break downloads.
    """
    try:
        r = requests.post(
            f"{CHITRAGUPTA_BASE}/v1/papers/check",
            json={"doi": doi, "title": title},
            timeout=CHITRAGUPTA_TIMEOUT,
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"[DEDUP] CHITRAGUPTA unreachable: {e} — skipping dedup check")
    return None


def _chit_register(
    title: str,
    doi: str | None,
    abstract: str | None,
    workflow_id: int,
    shani_paper_id: int,
) -> int | None:
    """
    Register a newly downloaded paper in brahm.db.
    Returns global_paper_id or None on failure.
    Never raises.
    """
    try:
        r = requests.post(
            f"{CHITRAGUPTA_BASE}/v1/papers",
            json={
                "title":          title,
                "doi":            doi,
                "abstract":       abstract,
                "workflow_id":    workflow_id,
                "shani_paper_id": shani_paper_id,
            },
            timeout=CHITRAGUPTA_TIMEOUT,
        )
        if r.status_code == 200:
            return r.json().get("global_paper_id")
    except Exception as e:
        print(f"[DEDUP] Register failed: {e}")
    return None


def _chit_link(global_paper_id: int, workflow_id: int) -> None:
    """Link an existing paper to this workflow. Never raises."""
    try:
        requests.post(
            f"{CHITRAGUPTA_BASE}/v1/papers/{global_paper_id}/link",
            json={"project_id": 0, "workflow_id": workflow_id},
            timeout=CHITRAGUPTA_TIMEOUT,
        )
    except Exception as e:
        print(f"[DEDUP] Link failed: {e}")


# =========================================================
# DIRECTORY SETUP
# =========================================================

def ensure_papers_directory():
    if not os.path.exists(PAPERS_DIR):
        os.makedirs(PAPERS_DIR)


def get_workflow_folder(repo, workflow_id) -> str:
    """
    Returns a sanitized subfolder name for this workflow.
    e.g. "LiNbO3 Fundamentals" → "papers/LiNbO3_Fundamentals/"
    Falls back to "papers/workflow_{id}/" if name not found.
    """
    row = repo.fetch_one(
        "SELECT name FROM Workflow WHERE id = ?", (workflow_id,)
    )
    if row and row["name"]:
        safe = re.sub(r"[^\w\-]", "_", row["name"].strip())
        safe = re.sub(r"_+", "_", safe).strip("_")
    else:
        safe = f"workflow_{workflow_id}"

    folder = os.path.join(PAPERS_DIR, safe)
    os.makedirs(folder, exist_ok=True)
    return folder


# =========================================================
# FETCH PAPERS
# =========================================================

def get_target_papers(repo, workflow_id):
    query = """
        SELECT id, title, doi, abstract, pdf_candidates
        FROM Paper
        WHERE workflow_id = ?
        AND pdf_status = 'enriched';
    """
    return [dict(r) for r in repo.fetch_all(query, (workflow_id,))]


# =========================================================
# DOWNLOAD
# =========================================================

def download_pdf(url, filepath):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/pdf,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://scholar.google.com/"
    }

    response = requests.get(url, headers=headers, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(f"Download failed: {url}")

    with open(filepath, "wb") as f:
        f.write(response.content)


# =========================================================
# VALIDATION
# =========================================================

def validate_pdf(filepath):
    import fitz

    if not os.path.exists(filepath):
        return False

    if os.path.getsize(filepath) / 1024 < MIN_FILE_SIZE_KB:
        return False

    try:
        doc = fitz.open(filepath)
        return doc.page_count > 0
    except:
        return False


# =========================================================
# SAFE DB EXECUTION
# =========================================================

def safe_update(repo, query, params):
    with repo.transaction() as cursor:
        cursor.execute(query, params)


# =========================================================
# MAIN TOOL
# =========================================================

def download_papers(repo, workflow_id, execution_attempt_id=None, **kwargs):

    ensure_papers_directory()
    workflow_folder = get_workflow_folder(repo, workflow_id)
    print(f"[DIR] Saving PDFs to: {workflow_folder}")

    papers = get_target_papers(repo, workflow_id)

    if not papers:
        print("No enriched papers found.")
        return {"status": "success", "data": [], "error": None}

    print(f"Found {len(papers)} papers to download.")

    downloaded = []
    deduped    = []

    for paper in papers:

        paper_id = paper["id"]
        title    = paper["title"]
        doi      = paper.get("doi") or None
        abstract = paper.get("abstract") or None

        print(f"\n[PROCESSING] {paper_id}: {title}")

        # =========================================================
        # DEDUPLICATION GATE — check brahm.db before downloading
        # =========================================================

        chit = _chit_check(doi, title)
        if chit and chit.get("exists"):
            global_paper_id = chit.get("global_paper_id")
            print(f"[DEDUP] Already in brahm.db as global_paper_id={global_paper_id} — skipping download")

            # Link this workflow to the existing paper
            _chit_link(global_paper_id, workflow_id)

            # Mark in SHANI DB so S4 can still extract content if file_path exists
            safe_update(
                repo,
                "UPDATE Paper SET pdf_status='deduplicated', last_error=? WHERE id=?",
                (f"global_paper_id={global_paper_id}", paper_id),
            )
            pr.update_paper_status(repo, paper_id, "processing")
            deduped.append(paper_id)
            continue   # skip download entirely

        # =========================================================
        # LOAD CANDIDATES SAFELY
        # =========================================================

        pdf_candidates_raw = paper["pdf_candidates"] if "pdf_candidates" in paper.keys() else None

        try:
            candidates = json.loads(pdf_candidates_raw) if pdf_candidates_raw else []
        except:
            candidates = []

        if not candidates:
            print("[SKIP] No candidates")

            safe_update(
                repo,
                "UPDATE Paper SET pdf_status='metadata_only', last_error=? WHERE id=?",
                ("no_candidates", paper_id)
            )
            continue

        filepath = os.path.join(workflow_folder, f"{paper_id}.pdf")

        success = False
        failed_sources = []
        last_error = None

        # =========================================================
        # CANDIDATE LOOP
        # =========================================================

        for cand in candidates:

            if not isinstance(cand, dict):
                continue

            url = cand.get("url")
            source = cand.get("source", "unknown")

            if not url or not isinstance(url, str):
                failed_sources.append(f"{source}_invalid_url")
                continue

            url = url.strip()

            if not url.startswith("http"):
                failed_sources.append(f"{source}_bad_url")
                continue

            if any(domain in url for domain in BLOCKED_DOMAINS):
                failed_sources.append(f"{source}_blocked")
                continue

            print(f"Trying [{source}]...")

            for attempt in range(MAX_RETRIES):

                try:
                    download_pdf(url, filepath)

                    if validate_pdf(filepath):
                        print(f"[SUCCESS] {source}")

                        pr.update_paper_file_path(repo, paper_id, filepath)

                        safe_update(
                            repo,
                            "UPDATE Paper SET pdf_status='downloaded' WHERE id=?",
                            (paper_id,)
                        )

                        success = True
                        break

                    else:
                        raise RuntimeError("Validation failed")

                except Exception as e:
                    last_error = str(e)
                    print(f"Attempt {attempt+1} failed: {last_error}")

            if success:
                break
            else:
                failed_sources.append(f"{source}_failed")

        # =========================================================
        # RESULT
        # =========================================================

        if success:

            pr.update_paper_status(repo, paper_id, "processing")
            downloaded.append(filepath)

            # ── Register new paper in brahm.db ───────────────────
            global_paper_id = _chit_register(
                title=title,
                doi=doi,
                abstract=abstract,
                workflow_id=workflow_id,
                shani_paper_id=paper_id,
            )
            if global_paper_id:
                print(f"[DEDUP] Registered as global_paper_id={global_paper_id}")
            else:
                print("[DEDUP] Registration skipped (CHITRAGUPTA down or error)")

        else:

            print(f"[FAILED] {paper_id}")

            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass

            safe_update(
                repo,
                """
                UPDATE Paper
                SET pdf_status='metadata_only',
                    failed_candidates=?,
                    last_error=?
                WHERE id=?
                """,
                (json.dumps(failed_sources), last_error, paper_id)
            )

            failure_repo.log_failure(
                repo,
                workflow_id,
                "DOWNLOAD_ERROR",
                last_error or "All candidates failed",
                paper_id=paper_id
            )

    return {
        "status": "success" if (downloaded or deduped) else "error",
        "data": downloaded,
        "deduped": deduped,
        "error": None if (downloaded or deduped) else "No valid downloads"
    }

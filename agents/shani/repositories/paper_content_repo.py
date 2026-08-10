import json
from datetime import datetime
from repositories.repository import Repository


# ============================================================
# PAPER CONTENT
# ============================================================

def create_paper_content(
    repo: Repository,
    paper_id: int,
    section_name: str,
    content: str,
    latex_text: str = None
):
    """
    latex_text: optional block-reconstructed LaTeX for this
                section. NULL when keyword fallback was used.
    """
    with repo.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO PaperContent (paper_id, section_name, content, latex_text)
            VALUES (?, ?, ?, ?)
            """,
            (paper_id, section_name, content, latex_text)
        )


def get_paper_content(repo: Repository, paper_id: int):
    rows = repo.fetch_all(
        """
        SELECT section_name, content
        FROM PaperContent
        WHERE paper_id = ?
        """,
        (paper_id,)
    )
    if not rows:
        return None
    return {r["section_name"]: r["content"] for r in rows}


# ============================================================
# PAPER EQUATION
# ============================================================

def create_paper_equation(
    repo: Repository,
    paper_id: int,
    equation_id: str,
    raw_text: str,
    normalized_latex: str,
    context_before: str = None,
    context_after: str = None,
    section_source: str = None,
    page_number: int = None,
    position_index: int = None
) -> int:
    """
    Stores one detected and normalized equation from S4.

    equation_id:      unique per paper, e.g. "42_eq_3"
    raw_text:         as extracted from PDF blocks
    normalized_latex: rule-normalized LaTeX wrapped in
                      \\begin{equation}...\\end{equation}
    context_before:   sentence before equation (defines meaning)
    context_after:    sentence after equation
    """
    timestamp = datetime.utcnow().isoformat()

    with repo.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO PaperEquation (
                paper_id, equation_id, raw_text, normalized_latex,
                context_before, context_after, section_source,
                page_number, position_index, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                paper_id, equation_id, raw_text, normalized_latex,
                context_before, context_after, section_source,
                page_number, position_index, timestamp
            )
        )
        return cursor.lastrowid


def get_equations_for_paper(repo: Repository, paper_id: int) -> list:
    rows = repo.fetch_all(
        """
        SELECT id, paper_id, equation_id, raw_text, normalized_latex,
               context_before, context_after, section_source,
               page_number, position_index
        FROM PaperEquation
        WHERE paper_id = ?
        ORDER BY position_index ASC
        """,
        (paper_id,)
    )
    return [dict(r) for r in rows]


def get_equations_for_workflow(
    repo: Repository,
    workflow_id: int,
    section_source: str = None
) -> list:
    """
    Returns equations for all papers in a workflow.
    Optionally filtered by section_source (e.g. 'methodology').
    Used by S6 to inject equation references into prompts.
    """
    if section_source:
        rows = repo.fetch_all(
            """
            SELECT pe.id, pe.paper_id, pe.equation_id,
                   pe.raw_text, pe.normalized_latex,
                   pe.context_before, pe.context_after,
                   pe.section_source, pe.page_number, pe.position_index
            FROM PaperEquation pe
            JOIN Paper p ON pe.paper_id = p.id
            WHERE p.workflow_id = ?
              AND pe.section_source = ?
            ORDER BY pe.paper_id, pe.position_index ASC
            """,
            (workflow_id, section_source)
        )
    else:
        rows = repo.fetch_all(
            """
            SELECT pe.id, pe.paper_id, pe.equation_id,
                   pe.raw_text, pe.normalized_latex,
                   pe.context_before, pe.context_after,
                   pe.section_source, pe.page_number, pe.position_index
            FROM PaperEquation pe
            JOIN Paper p ON pe.paper_id = p.id
            WHERE p.workflow_id = ?
            ORDER BY pe.paper_id, pe.position_index ASC
            """,
            (workflow_id,)
        )
    return [dict(r) for r in rows]


# ============================================================
# PAPER FIGURE
# ============================================================

def create_paper_figure(
    repo: Repository,
    paper_id: int,
    figure_id: str,
    image_path: str,
    caption: str = None,
    section_hint: str = None,
    page_number: int = None
) -> int:
    timestamp = datetime.utcnow().isoformat()
    with repo.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO PaperFigure (
                paper_id, figure_id, image_path,
                caption, section_hint, page_number, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (paper_id, figure_id, image_path,
             caption, section_hint, page_number, timestamp)
        )
        return cursor.lastrowid


def get_figures_for_paper(repo: Repository, paper_id: int) -> list:
    rows = repo.fetch_all(
        """
        SELECT id, paper_id, figure_id, image_path,
               caption, section_hint, page_number
        FROM PaperFigure
        WHERE paper_id = ?
        ORDER BY page_number ASC
        """,
        (paper_id,)
    )
    return [dict(r) for r in rows]


def get_figures_for_workflow(
    repo: Repository,
    workflow_id: int,
    section_hint: str = None
) -> list:
    if section_hint:
        rows = repo.fetch_all(
            """
            SELECT pf.id, pf.paper_id, pf.figure_id, pf.image_path,
                   pf.caption, pf.section_hint, pf.page_number
            FROM PaperFigure pf
            JOIN Paper p ON pf.paper_id = p.id
            WHERE p.workflow_id = ? AND pf.section_hint = ?
            ORDER BY pf.paper_id, pf.page_number ASC
            """,
            (workflow_id, section_hint)
        )
    else:
        rows = repo.fetch_all(
            """
            SELECT pf.id, pf.paper_id, pf.figure_id, pf.image_path,
                   pf.caption, pf.section_hint, pf.page_number
            FROM PaperFigure pf
            JOIN Paper p ON pf.paper_id = p.id
            WHERE p.workflow_id = ?
            ORDER BY pf.paper_id, pf.page_number ASC
            """,
            (workflow_id,)
        )
    return [dict(r) for r in rows]


# ============================================================
# PAPER TABLE
# ============================================================

def create_paper_table(
    repo: Repository,
    paper_id: int,
    table_id: str,
    table_type: str,
    headers: list = None,
    rows: list = None,
    image_path: str = None,
    caption: str = None,
    section_hint: str = None,
    page_number: int = None
) -> int:
    timestamp    = datetime.utcnow().isoformat()
    headers_json = json.dumps(headers) if headers is not None else None
    rows_json    = json.dumps(rows)    if rows    is not None else None

    with repo.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO PaperTable (
                paper_id, table_id, table_type,
                headers_json, rows_json, image_path,
                caption, section_hint, page_number, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (paper_id, table_id, table_type,
             headers_json, rows_json, image_path,
             caption, section_hint, page_number, timestamp)
        )
        return cursor.lastrowid


def get_tables_for_paper(repo: Repository, paper_id: int) -> list:
    rows = repo.fetch_all(
        """
        SELECT id, paper_id, table_id, table_type,
               headers_json, rows_json, image_path,
               caption, section_hint, page_number
        FROM PaperTable
        WHERE paper_id = ?
        ORDER BY page_number ASC
        """,
        (paper_id,)
    )
    result = []
    for r in rows:
        d = dict(r)
        d["headers"] = json.loads(d["headers_json"]) if d["headers_json"] else []
        d["rows"]    = json.loads(d["rows_json"])    if d["rows_json"]    else []
        result.append(d)
    return result


def get_tables_for_workflow(
    repo: Repository,
    workflow_id: int,
    section_hint: str = None
) -> list:
    if section_hint:
        rows = repo.fetch_all(
            """
            SELECT pt.id, pt.paper_id, pt.table_id, pt.table_type,
                   pt.headers_json, pt.rows_json, pt.image_path,
                   pt.caption, pt.section_hint, pt.page_number
            FROM PaperTable pt
            JOIN Paper p ON pt.paper_id = p.id
            WHERE p.workflow_id = ? AND pt.section_hint = ?
            ORDER BY pt.paper_id, pt.page_number ASC
            """,
            (workflow_id, section_hint)
        )
    else:
        rows = repo.fetch_all(
            """
            SELECT pt.id, pt.paper_id, pt.table_id, pt.table_type,
                   pt.headers_json, pt.rows_json, pt.image_path,
                   pt.caption, pt.section_hint, pt.page_number
            FROM PaperTable pt
            JOIN Paper p ON pt.paper_id = p.id
            WHERE p.workflow_id = ?
            ORDER BY pt.paper_id, pt.page_number ASC
            """,
            (workflow_id,)
        )
    result = []
    for r in rows:
        d = dict(r)
        d["headers"] = json.loads(d["headers_json"]) if d["headers_json"] else []
        d["rows"]    = json.loads(d["rows_json"])    if d["rows_json"]    else []
        result.append(d)
    return result

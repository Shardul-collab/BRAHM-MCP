"""
research_analyzer.py  —  Chitragupta Analysis Layer
====================================================
Location (deploy to): /mnt/d/chitragupta/analysis/research_analyzer.py

Stateless analysis engine over SHANI's SQLite research database.
Each public method opens its own read-only connection and closes it on exit —
safe to call concurrently from the MCP server.

All methods return the standard Group D envelope:
{
  "status": "success" | "error",
  "tool": "<method_name>",
  "scope": { workflow_ids, papers_analyzed, knowledge_rows_analyzed },
  "results": { ... },
  "interpretation_hints": [...],
  "suggested_next_tools": [...]
}
"""

import re
import sqlite3
from collections import defaultdict
from statistics import mean
from typing import Optional

DB_PATH = "/mnt/d/SQL_IMP_AI_Project/database/research_workflow.db"

VALID_CATEGORIES = {
    "material", "synthesis_method", "characterization",
    "application", "computational_method"
}


class ResearchAnalyzer:
    """
    Read-only analysis engine. Thread-safe — the MCP server may call any
    method from concurrent async tasks. Each method creates and destroys
    its own sqlite3 connection; no shared state is held between calls.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    # ─────────────────────────── connection helper ──────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # Read-only — analysis must never write to the pipeline DB
        conn.execute("PRAGMA query_only = ON;")
        return conn

    # ─────────────────────────── scope helper ───────────────────────────────

    def _scope(self, conn: sqlite3.Connection, workflow_id: Optional[int]) -> dict:
        if workflow_id:
            papers = conn.execute(
                "SELECT COUNT(*) FROM Paper WHERE workflow_id = ?", (workflow_id,)
            ).fetchone()[0]
            rows = conn.execute(
                """SELECT COUNT(*) FROM ResearchKnowledge rk
                   JOIN Paper p ON p.id = rk.paper_id
                   WHERE p.workflow_id = ?""", (workflow_id,)
            ).fetchone()[0]
            wf_ids = [workflow_id]
        else:
            papers = conn.execute("SELECT COUNT(*) FROM Paper").fetchone()[0]
            rows = conn.execute("SELECT COUNT(*) FROM ResearchKnowledge").fetchone()[0]
            wf_ids = [
                r[0] for r in conn.execute("SELECT DISTINCT id FROM Workflow").fetchall()
            ]
        return {"workflow_ids": wf_ids, "papers_analyzed": papers,
                "knowledge_rows_analyzed": rows}

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP D — Tool 1: technique_frequency
    # ═══════════════════════════════════════════════════════════════════════

    def technique_frequency(
        self,
        category: str,
        workflow_id: Optional[int] = None,
        top_n: int = 20,
        min_count: int = 2,
    ) -> dict:
        """
        Count how often each value appears in a given knowledge category.
        Answers: "what synthesis methods dominate this corpus?"
        """
        if category != "all" and category not in VALID_CATEGORIES:
            return {"status": "error",
                    "error": f"Invalid category '{category}'. "
                             f"Valid: all, {', '.join(sorted(VALID_CATEGORIES))}"}
        try:
            conn = self._conn()
            scope = self._scope(conn, workflow_id)

            if category == "all":
                cat_filter = ""
                params: tuple = ()
            else:
                cat_filter = "AND rk.category = ?"
                params = (category,)

            wf_filter = "AND p.workflow_id = ?" if workflow_id else ""
            if workflow_id:
                params = params + (workflow_id,)

            rows = conn.execute(
                f"""
                SELECT rk.category,
                       rk.value,
                       COUNT(*)                    AS mention_count,
                       COUNT(DISTINCT rk.paper_id) AS paper_count,
                       COUNT(DISTINCT p.workflow_id) AS workflow_count
                FROM ResearchKnowledge rk
                JOIN Paper p ON p.id = rk.paper_id
                WHERE 1=1 {cat_filter} {wf_filter}
                GROUP BY rk.category, rk.value
                HAVING COUNT(*) >= ?
                ORDER BY mention_count DESC
                LIMIT ?
                """,
                params + (min_count, top_n),
            ).fetchall()
            conn.close()

            results = [
                {
                    "category": r["category"],
                    "value": r["value"],
                    "mention_count": r["mention_count"],
                    "paper_count": r["paper_count"],
                    "workflow_count": r["workflow_count"],
                }
                for r in rows
            ]

            hints = []
            if results:
                top = results[0]
                hints.append(
                    f"'{top['value']}' is the most frequent {top['category']} "
                    f"with {top['mention_count']} mentions across "
                    f"{top['paper_count']} papers."
                )
                if len(results) >= 3:
                    hints.append(
                        f"Top 3 values cover "
                        f"{sum(r['mention_count'] for r in results[:3])} of "
                        f"{scope['knowledge_rows_analyzed']} total knowledge rows."
                    )

            suggested = []
            if results:
                top_val = results[0]["value"]
                top_cat = results[0]["category"]
                suggested.append(
                    f"analysis_trend_report with primary_category='{top_cat}' "
                    f"filter_value='{top_val}' to see co-occurrences"
                )
                suggested.append(
                    f"analysis_find_gaps with category_a='{top_cat}' "
                    f"category_b='application' to find unexplored pairings"
                )

            return {
                "status": "success",
                "tool": "analysis_technique_frequency",
                "scope": scope,
                "category": category,
                "results": results,
                "interpretation_hints": hints,
                "suggested_next_tools": suggested,
            }

        except Exception as exc:
            return {"status": "error", "error": str(exc),
                    "tool": "analysis_technique_frequency"}

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP D — Tool 2: trend_report  (co-occurrence matrix)
    # ═══════════════════════════════════════════════════════════════════════

    def trend_report(
        self,
        primary_category: str,
        secondary_category: Optional[str] = None,
        filter_value: Optional[str] = None,
        workflow_id: Optional[int] = None,
        min_co_occurrence: int = 3,
    ) -> dict:
        """
        Find co-occurrence patterns between two knowledge categories.
        e.g. "what characterization methods appear with magnetron sputtering?"
        """
        if primary_category not in VALID_CATEGORIES:
            return {"status": "error",
                    "error": f"Invalid primary_category '{primary_category}'."}

        try:
            conn = self._conn()
            scope = self._scope(conn, workflow_id)

            wf_join = (
                "JOIN Paper p1 ON p1.id = rk1.paper_id "
                "JOIN Paper p2 ON p2.id = rk2.paper_id "
            )
            wf_filter = ""
            base_params: list = [primary_category]

            if filter_value:
                base_params.append(filter_value.lower())

            if secondary_category and secondary_category in VALID_CATEGORIES:
                sec_filter = "AND rk2.category = ?"
                secondary_params = [secondary_category]
            else:
                sec_filter = ""
                secondary_params = []

            wf_param: list = []
            if workflow_id:
                wf_filter = "AND p1.workflow_id = ? AND p2.workflow_id = ?"
                wf_param = [workflow_id, workflow_id]

            filter_clause = (
                "AND LOWER(rk1.value) = ?" if filter_value else ""
            )

            rows = conn.execute(
                f"""
                SELECT rk1.value  AS primary_val,
                       rk2.value  AS secondary_val,
                       rk2.category AS secondary_cat,
                       COUNT(DISTINCT rk1.paper_id) AS co_paper_count
                FROM ResearchKnowledge rk1
                JOIN ResearchKnowledge rk2
                    ON rk1.paper_id = rk2.paper_id
                    AND rk1.value != rk2.value
                {wf_join}
                WHERE rk1.category = ?
                  {filter_clause}
                  {sec_filter}
                  {wf_filter}
                GROUP BY rk1.value, rk2.value, rk2.category
                HAVING COUNT(DISTINCT rk1.paper_id) >= ?
                ORDER BY co_paper_count DESC
                LIMIT 50
                """,
                base_params + secondary_params + wf_param + [min_co_occurrence],
            ).fetchall()

            # Gather sample titles for top 5 pairs
            pairs = []
            for r in rows:
                pair = {
                    "primary": r["primary_val"],
                    "secondary": r["secondary_val"],
                    "secondary_category": r["secondary_cat"],
                    "paper_count": r["co_paper_count"],
                }
                # Fetch up to 3 sample titles
                titles = conn.execute(
                    """
                    SELECT DISTINCT p.title FROM ResearchKnowledge rk1
                    JOIN ResearchKnowledge rk2 ON rk1.paper_id = rk2.paper_id
                    JOIN Paper p ON p.id = rk1.paper_id
                    WHERE rk1.category = ? AND LOWER(rk1.value) = LOWER(?)
                      AND LOWER(rk2.value) = LOWER(?)
                    LIMIT 3
                    """,
                    (primary_category, r["primary_val"], r["secondary_val"]),
                ).fetchall()
                pair["sample_titles"] = [t[0][:80] for t in titles]
                pairs.append(pair)

            conn.close()

            hints = []
            if pairs:
                top = pairs[0]
                hints.append(
                    f"'{top['primary']}' most commonly co-occurs with "
                    f"'{top['secondary']}' in {top['paper_count']} papers."
                )
                if len(pairs) >= 5:
                    unique_primaries = len({p["primary"] for p in pairs})
                    hints.append(
                        f"Found {len(pairs)} co-occurrence pairs across "
                        f"{unique_primaries} distinct primary values."
                    )

            suggested = []
            if pairs:
                suggested.append(
                    f"analysis_find_gaps with category_a='{primary_category}' "
                    f"category_b='{secondary_category or 'application'}' "
                    f"to identify unexplored combinations"
                )

            return {
                "status": "success",
                "tool": "analysis_trend_report",
                "scope": scope,
                "pivot": primary_category,
                "co_occurrence_axis": secondary_category,
                "filter_applied": filter_value,
                "results": {"pairs": pairs, "total_pairs": len(pairs)},
                "interpretation_hints": hints,
                "suggested_next_tools": suggested,
            }

        except Exception as exc:
            return {"status": "error", "error": str(exc),
                    "tool": "analysis_trend_report"}

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP D — Tool 3: find_gaps
    # ═══════════════════════════════════════════════════════════════════════

    def find_gaps(
        self,
        category_a: str,
        category_b: str,
        known_values_a: Optional[list] = None,
        known_values_b: Optional[list] = None,
        gap_threshold: int = 2,
    ) -> dict:
        """
        Identify under-explored or unexplored A×B combinations.
        The most strategically valuable analysis tool.
        """
        for cat, name in [(category_a, "category_a"), (category_b, "category_b")]:
            if cat not in VALID_CATEGORIES:
                return {"status": "error",
                        "error": f"Invalid {name} '{cat}'."}

        try:
            conn = self._conn()
            scope = self._scope(conn, None)

            # Discover all distinct values for A and B if not constrained
            if known_values_a:
                values_a = [v.lower() for v in known_values_a]
            else:
                rows_a = conn.execute(
                    """SELECT DISTINCT LOWER(value) FROM ResearchKnowledge
                       WHERE category = ? ORDER BY value LIMIT 30""",
                    (category_a,),
                ).fetchall()
                values_a = [r[0] for r in rows_a]

            if known_values_b:
                values_b = [v.lower() for v in known_values_b]
            else:
                rows_b = conn.execute(
                    """SELECT DISTINCT LOWER(value) FROM ResearchKnowledge
                       WHERE category = ? ORDER BY value LIMIT 30""",
                    (category_b,),
                ).fetchall()
                values_b = [r[0] for r in rows_b]

            # Build the co-occurrence map for all A×B combinations
            co_rows = conn.execute(
                """
                SELECT LOWER(rk1.value) AS va, LOWER(rk2.value) AS vb,
                       COUNT(DISTINCT rk1.paper_id) AS cnt
                FROM ResearchKnowledge rk1
                JOIN ResearchKnowledge rk2
                    ON rk1.paper_id = rk2.paper_id
                WHERE rk1.category = ? AND rk2.category = ?
                GROUP BY LOWER(rk1.value), LOWER(rk2.value)
                """,
                (category_a, category_b),
            ).fetchall()
            conn.close()

            co_map: dict = defaultdict(lambda: defaultdict(int))
            for r in co_rows:
                co_map[r["va"]][r["vb"]] = r["cnt"]

            gap_combos = []
            well_explored = []
            total = len(values_a) * len(values_b)

            for va in values_a:
                for vb in values_b:
                    cnt = co_map[va].get(vb, 0)
                    if cnt <= gap_threshold:
                        # Find nearest explored variant for context
                        nearest = max(
                            (
                                (other_va, co_map[other_va].get(vb, 0))
                                for other_va in values_a if other_va != va
                            ),
                            key=lambda x: x[1],
                            default=(None, 0),
                        )
                        gap_combos.append({
                            "a": va,
                            "b": vb,
                            "paper_count": cnt,
                            "gap_type": "unexplored" if cnt == 0 else "underexplored",
                            "nearest_explored": (
                                f"{nearest[0]} + {vb} ({nearest[1]} papers)"
                                if nearest[0] and nearest[1] > 0
                                else "none in corpus"
                            ),
                        })
                    else:
                        well_explored.append({
                            "a": va, "b": vb, "paper_count": cnt
                        })

            gap_combos.sort(key=lambda x: x["paper_count"])
            well_explored.sort(key=lambda x: -x["paper_count"])

            unexplored_count = sum(1 for g in gap_combos if g["gap_type"] == "unexplored")
            hints = []
            if unexplored_count:
                hints.append(
                    f"{unexplored_count} of {total} possible combinations are "
                    f"completely unexplored ({category_a} × {category_b})."
                )
            if gap_combos:
                top_gap = gap_combos[0]
                hints.append(
                    f"Biggest gap: '{top_gap['a']}' + '{top_gap['b']}' "
                    f"({top_gap['paper_count']} papers — {top_gap['gap_type']})."
                )
            if well_explored:
                top_well = well_explored[0]
                hints.append(
                    f"Best-covered: '{top_well['a']}' + '{top_well['b']}' "
                    f"with {top_well['paper_count']} papers."
                )

            suggested = []
            if gap_combos:
                top_gap = gap_combos[0]
                suggested.append(
                    f"shani_batch_run with focus on '{top_gap['a']} {top_gap['b']}' "
                    f"to fill the biggest gap"
                )
                suggested.append(
                    f"research_find_papers_by_topic keywords=['{top_gap['a']}', "
                    f"'{top_gap['b']}'] to verify corpus coverage"
                )

            return {
                "status": "success",
                "tool": "analysis_find_gaps",
                "scope": scope,
                "results": {
                    "category_a": category_a,
                    "category_b": category_b,
                    "total_possible_combinations": total,
                    "explored_combinations": len(well_explored),
                    "gap_combinations": gap_combos[:30],
                    "well_explored": well_explored[:10],
                },
                "interpretation_hints": hints,
                "suggested_next_tools": suggested,
            }

        except Exception as exc:
            return {"status": "error", "error": str(exc),
                    "tool": "analysis_find_gaps"}

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP D — Tool 4: parameter_distribution
    # ═══════════════════════════════════════════════════════════════════════

    def parameter_distribution(
        self,
        parameter_keywords: list,
        workflow_id: Optional[int] = None,
        extract_numbers: bool = True,
        group_by_material: bool = True,
    ) -> dict:
        """
        Find and aggregate numeric parameter values (bandgap, temperatures, etc.)
        from ResearchKnowledge sentences matching the given keywords.
        """
        if not parameter_keywords:
            return {"status": "error", "error": "parameter_keywords cannot be empty"}

        try:
            conn = self._conn()
            scope = self._scope(conn, workflow_id)

            kw_lower = [k.lower() for k in parameter_keywords]

            wf_filter = "AND p.workflow_id = ?" if workflow_id else ""
            wf_params = (workflow_id,) if workflow_id else ()

            # Search both rk.value and rk.context for the keywords
            rows = conn.execute(
                f"""
                SELECT rk.id, rk.paper_id, rk.value, rk.context,
                       p.title, p.workflow_id
                FROM ResearchKnowledge rk
                JOIN Paper p ON p.id = rk.paper_id
                WHERE (
                    {' OR '.join(
                        ["LOWER(rk.value) LIKE ? OR LOWER(rk.context) LIKE ?"
                         for _ in kw_lower]
                    )}
                ) {wf_filter}
                LIMIT 500
                """,
                [item for kw in kw_lower for item in (f"%{kw}%", f"%{kw}%")]
                + list(wf_params),
            ).fetchall()

            # Optionally also get material context for grouping
            mat_map: dict = {}
            if group_by_material:
                mat_rows = conn.execute(
                    f"""
                    SELECT rk.paper_id, GROUP_CONCAT(rk.value, '|') AS materials
                    FROM ResearchKnowledge rk
                    JOIN Paper p ON p.id = rk.paper_id
                    WHERE rk.category = 'material' {wf_filter}
                    GROUP BY rk.paper_id
                    """,
                    list(wf_params),
                ).fetchall()
                for mr in mat_rows:
                    mat_map[mr["paper_id"]] = mr["materials"].split("|")[0]

            conn.close()

            total_mentions = len(rows)
            raw_sentences = []
            by_material: dict = defaultdict(list)

            for r in rows:
                sentence = r["context"] or r["value"] or ""
                nums = self._extract_numbers(sentence) if extract_numbers else []

                material = mat_map.get(r["paper_id"], "unknown") if group_by_material else "all"

                if nums:
                    by_material[material].extend(nums)

                if len(raw_sentences) < 10:
                    raw_sentences.append({
                        "sentence": sentence[:200],
                        "value": r["value"],
                        "paper_id": r["paper_id"],
                        "title": r["title"][:80],
                        "numbers_extracted": nums[:5],
                    })

            # Compute stats per material
            by_material_stats = {}
            for mat, vals in by_material.items():
                if vals:
                    by_material_stats[mat] = {
                        "values": vals[:50],
                        "mean": round(mean(vals), 4),
                        "min": round(min(vals), 4),
                        "max": round(max(vals), 4),
                        "paper_count": sum(
                            1 for r in rows
                            if mat_map.get(r["paper_id"], "unknown") == mat
                        ),
                    }

            hints = []
            hints.append(
                f"Found {total_mentions} knowledge entries matching "
                f"{parameter_keywords} across {scope['papers_analyzed']} papers."
            )
            if by_material_stats:
                best = max(
                    by_material_stats.items(),
                    key=lambda x: x[1]["paper_count"]
                )
                hints.append(
                    f"Most data for material '{best[0]}': "
                    f"mean={best[1]['mean']}, "
                    f"range=[{best[1]['min']}, {best[1]['max']}] "
                    f"from {best[1]['paper_count']} papers."
                )

            return {
                "status": "success",
                "tool": "analysis_parameter_distribution",
                "scope": scope,
                "results": {
                    "parameter_keywords": parameter_keywords,
                    "total_mentions": total_mentions,
                    "by_material": by_material_stats,
                    "raw_sentences": raw_sentences,
                },
                "interpretation_hints": hints,
                "suggested_next_tools": [
                    f"analysis_trend_report with filter_value='{parameter_keywords[0]}' "
                    f"to see which synthesis methods pair with this parameter"
                ],
            }

        except Exception as exc:
            return {"status": "error", "error": str(exc),
                    "tool": "analysis_parameter_distribution"}

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP D — Tool 5: workflow_comparison
    # ═══════════════════════════════════════════════════════════════════════

    def workflow_comparison(
        self,
        workflow_ids: list,
        compare_by: str = "knowledge",
    ) -> dict:
        """
        Compare two or more workflows to identify overlap and unique coverage.
        """
        if len(workflow_ids) < 2:
            return {"status": "error",
                    "error": "workflow_comparison requires at least 2 workflow_ids"}

        valid_modes = {"papers", "techniques", "materials", "knowledge"}
        if compare_by not in valid_modes:
            return {"status": "error",
                    "error": f"compare_by must be one of {sorted(valid_modes)}"}

        try:
            conn = self._conn()

            wf_summaries = []
            for wid in workflow_ids:
                wf = conn.execute(
                    "SELECT id, name FROM Workflow WHERE id = ?", (wid,)
                ).fetchone()
                if not wf:
                    conn.close()
                    return {"status": "error",
                            "error": f"Workflow {wid} not found"}

                papers = conn.execute(
                    "SELECT COUNT(*) FROM Paper WHERE workflow_id = ?", (wid,)
                ).fetchone()[0]
                rk_rows = conn.execute(
                    """SELECT COUNT(*) FROM ResearchKnowledge rk
                       JOIN Paper p ON p.id = rk.paper_id WHERE p.workflow_id = ?""",
                    (wid,)
                ).fetchone()[0]

                wf_summaries.append({
                    "id": wid, "name": dict(wf)["name"],
                    "papers": papers, "knowledge_rows": rk_rows,
                })

            # Build per-workflow sets for comparison
            def get_set(wid: int, category_filter: Optional[str]) -> set:
                if category_filter:
                    rows = conn.execute(
                        """SELECT LOWER(rk.value) FROM ResearchKnowledge rk
                           JOIN Paper p ON p.id = rk.paper_id
                           WHERE p.workflow_id = ? AND rk.category = ?""",
                        (wid, category_filter),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT LOWER(rk.value) FROM ResearchKnowledge rk
                           JOIN Paper p ON p.id = rk.paper_id
                           WHERE p.workflow_id = ?""",
                        (wid,),
                    ).fetchall()
                return {r[0] for r in rows}

            def get_paper_titles(wid: int) -> set:
                rows = conn.execute(
                    "SELECT LOWER(title) FROM Paper WHERE workflow_id = ?", (wid,)
                ).fetchall()
                return {r[0] for r in rows}

            if compare_by == "papers":
                sets = {wid: get_paper_titles(wid) for wid in workflow_ids}
            elif compare_by == "materials":
                sets = {wid: get_set(wid, "material") for wid in workflow_ids}
            elif compare_by == "techniques":
                sets = {wid: get_set(wid, "synthesis_method") for wid in workflow_ids}
            else:  # knowledge
                sets = {wid: get_set(wid, None) for wid in workflow_ids}

            conn.close()

            # Intersection (shared across ALL workflows)
            shared = set.intersection(*sets.values()) if len(sets) > 0 and all(sets.values()) else set()
            # Unique to each workflow
            unique_per_wf = {}
            for wid in workflow_ids:
                other_sets = [sets[w] for w in workflow_ids if w != wid]
                others = set.union(*other_sets) if other_sets else set()
                unique_per_wf[wid] = sorted(sets[wid] - others)[:20]

            hints = []
            hints.append(
                f"{len(shared)} values are shared across all "
                f"{len(workflow_ids)} workflows."
            )
            for wid in workflow_ids:
                name = next(w["name"] for w in wf_summaries if w["id"] == wid)
                hints.append(
                    f"Workflow '{name}' has "
                    f"{len(unique_per_wf[wid])} unique values not found elsewhere."
                )

            return {
                "status": "success",
                "tool": "analysis_workflow_comparison",
                "scope": {
                    "workflow_ids": workflow_ids,
                    "papers_analyzed": sum(w["papers"] for w in wf_summaries),
                    "knowledge_rows_analyzed": sum(
                        w["knowledge_rows"] for w in wf_summaries
                    ),
                },
                "results": {
                    "compare_by": compare_by,
                    "workflows": wf_summaries,
                    "shared_count": len(shared),
                    "shared_values": sorted(shared)[:30],
                    "unique_per_workflow": {
                        str(wid): vals for wid, vals in unique_per_wf.items()
                    },
                },
                "interpretation_hints": hints,
                "suggested_next_tools": [
                    "analysis_find_gaps to identify unexplored combination spaces",
                ],
            }

        except Exception as exc:
            return {"status": "error", "error": str(exc),
                    "tool": "analysis_workflow_comparison"}

    # ─────────────────────────── private helpers ────────────────────────────

    def _extract_numbers(self, text: str) -> list:
        """
        Extract numeric values with scientific units from a text string.
        Targets common materials-science units: eV, nm, K, °C, %, cm⁻³, etc.
        """
        pattern = (
            r"(\d+\.?\d*(?:[eE][+-]?\d+)?)"
            r"\s*"
            r"(?:eV|nm|cm|cm-3|cm\^-3|K|°C|°F|%|mol|at\.%|"
            r"GPa|MPa|Ω|Ohm|S/cm|μm|mm|m|g|mg|MHz|GHz|mW|W)"
        )
        matches = re.findall(pattern, text, re.IGNORECASE)
        result = []
        for m in matches:
            try:
                result.append(float(m))
            except ValueError:
                pass
        return result

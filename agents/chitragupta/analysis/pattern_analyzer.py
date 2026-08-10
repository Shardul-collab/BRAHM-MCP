# analysis/pattern_analyzer.py

"""
Pattern Analyzer — schema-driven insights from stored Notion data.

Responsibilities:
- Fetch all pages from a Notion database via query_database()
- Extract field values using the local schema as the guide
- Compute basic statistics: averages, counts, keyword frequency
- Detect simple trends, correlations, and consistency patterns
- Return a structured insights dict

Design contract:
- Zero hardcoded field names — everything driven by the loaded schema
- No heavy ML — stdlib math only (no numpy/pandas required)
- Works for any database schema: numeric, categorical, text, date, checkbox
- Empty databases and missing values are handled gracefully throughout
"""

import logging
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from statistics import mean, stdev, StatisticsError
from typing import Any

from notion.schema_manager import load_schema
from notion.notion_client import query_database

logger = logging.getLogger("chitragupta.pattern_analyzer")


# ── Notion page → flat dict ───────────────────────────────────────────────────

def _extract_value(prop: dict[str, Any], ftype: str) -> Any:
    """
    Pull the Python-native value out of a single Notion property object.

    Returns None when the field is empty or the type is unhandled.
    """
    if prop is None:
        return None

    try:
        if ftype == "title":
            items = prop.get("title", [])
            return " ".join(t.get("plain_text", "") for t in items).strip() or None

        if ftype == "rich_text":
            items = prop.get("rich_text", [])
            return " ".join(t.get("plain_text", "") for t in items).strip() or None

        if ftype == "number":
            return prop.get("number")          # already int/float or None

        if ftype == "select":
            sel = prop.get("select")
            return sel.get("name") if sel else None

        if ftype == "multi_select":
            return [o.get("name") for o in prop.get("multi_select", [])]

        if ftype == "checkbox":
            return prop.get("checkbox")        # bool

        if ftype == "date":
            d = prop.get("date")
            return d.get("start") if d else None

        if ftype == "url":
            return prop.get("url")

        if ftype == "email":
            return prop.get("email")

        if ftype == "phone_number":
            return prop.get("phone_number")

        if ftype == "relation":
            return [r.get("id") for r in prop.get("relation", [])]

    except (AttributeError, TypeError, KeyError):
        pass

    return None


def _page_to_flat(page: dict[str, Any], fields: list[dict]) -> dict[str, Any]:
    """
    Convert a raw Notion page object into a flat {field_name: value} dict
    using the schema field list to know which properties to extract.
    """
    props = page.get("properties", {})
    row: dict[str, Any] = {
        "_page_id":   page.get("id", ""),
        "_created":   page.get("created_time", ""),
        "_last_edited": page.get("last_edited_time", ""),
    }

    for field in fields:
        name  = field["name"]
        ftype = field["type"]
        prop  = props.get(name)
        row[name] = _extract_value(prop, ftype) if prop else None

    return row


# ── Column splitter ───────────────────────────────────────────────────────────

def _split_columns(
    rows: list[dict[str, Any]],
    fields: list[dict],
) -> dict[str, list[Any]]:
    """
    Pivot a list of flat row dicts into per-column value lists.

    {field_name: [value, value, ...]}  — None entries are kept so
    index alignment is preserved (needed for correlation later).
    """
    columns: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        for field in fields:
            columns[field["name"]].append(row.get(field["name"]))
    return dict(columns)


# ── Basic statistics ──────────────────────────────────────────────────────────

def compute_basic_stats(
    rows: list[dict[str, Any]],
    fields: list[dict],
) -> dict[str, Any]:
    """
    Compute per-field statistics appropriate to each field type.

    Number fields  → min, max, mean, stdev, non-null count
    Select /
    Multi-select   → value counts (top 10)
    Checkbox       → true/false counts
    Text fields    → non-empty count, top keywords
    Date fields    → earliest, latest

    Args:
        rows:   List of flat row dicts from _page_to_flat().
        fields: Schema field list.

    Returns:
        Dict keyed by field name; value is a stats sub-dict.
    """
    stats: dict[str, Any] = {}
    columns = _split_columns(rows, fields)

    for field in fields:
        name  = field["name"]
        ftype = field["type"]
        values = columns.get(name, [])

        if ftype == "number":
            nums = [v for v in values if isinstance(v, (int, float))]
            if nums:
                stats[name] = {
                    "type":      "number",
                    "count":     len(nums),
                    "missing":   len(values) - len(nums),
                    "min":       min(nums),
                    "max":       max(nums),
                    "mean":      round(mean(nums), 3),
                    "stdev":     round(stdev(nums), 3) if len(nums) > 1 else 0.0,
                }
            else:
                stats[name] = {"type": "number", "count": 0, "missing": len(values)}

        elif ftype in ("select", "multi_select"):
            flat: list[str] = []
            for v in values:
                if isinstance(v, list):
                    flat.extend(v)
                elif isinstance(v, str):
                    flat.append(v)
            counter = Counter(flat)
            stats[name] = {
                "type":        ftype,
                "total_tags":  len(flat),
                "unique":      len(counter),
                "top_values":  counter.most_common(10),
            }

        elif ftype == "checkbox":
            bools = [v for v in values if isinstance(v, bool)]
            true_count  = sum(1 for v in bools if v)
            false_count = len(bools) - true_count
            stats[name] = {
                "type":    "checkbox",
                "true":    true_count,
                "false":   false_count,
                "missing": len(values) - len(bools),
            }

        elif ftype in ("title", "rich_text"):
            texts = [v for v in values if isinstance(v, str) and v]
            keywords = _top_keywords(texts)
            stats[name] = {
                "type":           ftype,
                "non_empty":      len(texts),
                "missing":        len(values) - len(texts),
                "top_keywords":   keywords,
            }

        elif ftype == "date":
            dates = sorted(v for v in values if isinstance(v, str) and v)
            stats[name] = {
                "type":     "date",
                "count":    len(dates),
                "earliest": dates[0]  if dates else None,
                "latest":   dates[-1] if dates else None,
            }

        else:
            # url, email, phone, relation, people — just count non-nulls
            non_null = [v for v in values if v is not None]
            stats[name] = {
                "type":     ftype,
                "non_null": len(non_null),
                "missing":  len(values) - len(non_null),
            }

    return stats


def _top_keywords(
    texts: list[str],
    top_n: int = 10,
    min_length: int = 4,
) -> list[tuple[str, int]]:
    """
    Extract the most frequent meaningful words across a list of strings.

    Strips punctuation, lowercases, removes very short words, and returns
    the top_n (word, count) tuples.
    """
    _STOPWORDS = {
        "this", "that", "with", "have", "been", "from", "they", "were",
        "their", "there", "what", "when", "where", "which", "will", "would",
        "could", "should", "then", "than", "also", "some", "into", "about",
        "your", "just", "like", "more", "very",
    }

    counter: Counter = Counter()
    for text in texts:
        words = re.findall(r"[a-z]+", text.lower())
        for word in words:
            if len(word) >= min_length and word not in _STOPWORDS:
                counter[word] += 1

    return counter.most_common(top_n)


# ── Pattern detection ─────────────────────────────────────────────────────────

def detect_patterns(
    rows: list[dict[str, Any]],
    fields: list[dict],
) -> dict[str, Any]:
    """
    Detect higher-level patterns across the dataset.

    Computes:
    - Numeric trends (is the field going up, down, or flat over time?)
    - Pairwise correlations between numeric fields (Pearson-r, no libraries)
    - Entry consistency score (how regularly are entries being made?)

    Args:
        rows:   List of flat row dicts sorted by _created ascending.
        fields: Schema field list.

    Returns:
        Dict with keys: trends, correlations, consistency.
    """
    patterns: dict[str, Any] = {}

    # ── Sort rows by creation time ────────────────────────────────────────────
    def _parse_dt(s: str) -> datetime:
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return datetime.min.replace(tzinfo=timezone.utc)

    sorted_rows = sorted(rows, key=lambda r: _parse_dt(r.get("_created", "")))

    numeric_fields = [f for f in fields if f["type"] == "number"]

    # ── Trends ────────────────────────────────────────────────────────────────
    trends: dict[str, str] = {}
    for field in numeric_fields:
        name = field["name"]
        series = [
            r[name] for r in sorted_rows
            if isinstance(r.get(name), (int, float))
        ]
        trends[name] = _trend_direction(series)

    patterns["trends"] = trends

    # ── Pairwise Pearson correlations ─────────────────────────────────────────
    correlations: list[dict[str, Any]] = []

    for i in range(len(numeric_fields)):
        for j in range(i + 1, len(numeric_fields)):
            name_a = numeric_fields[i]["name"]
            name_b = numeric_fields[j]["name"]

            pairs = [
                (r[name_a], r[name_b])
                for r in sorted_rows
                if isinstance(r.get(name_a), (int, float))
                and isinstance(r.get(name_b), (int, float))
            ]

            if len(pairs) < 3:          # need at least 3 points for r
                continue

            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            r  = _pearson(xs, ys)

            if r is None:
                continue

            correlations.append({
                "field_a":     name_a,
                "field_b":     name_b,
                "pearson_r":   round(r, 3),
                "strength":    _correlation_label(r),
                "sample_size": len(pairs),
            })

    # Sort strongest correlations first
    correlations.sort(key=lambda c: abs(c["pearson_r"]), reverse=True)
    patterns["correlations"] = correlations

    # ── Consistency ───────────────────────────────────────────────────────────
    patterns["consistency"] = _compute_consistency(sorted_rows)

    return patterns


def _trend_direction(series: list[float | int]) -> str:
    """
    Label a time-ordered numeric series as 'increasing', 'decreasing',
    or 'stable' using a simple first-half vs second-half mean comparison.
    """
    if len(series) < 4:
        return "insufficient data"

    mid   = len(series) // 2
    first = mean(series[:mid])
    last  = mean(series[mid:])
    delta = last - first

    try:
        threshold = stdev(series) * 0.25    # 25 % of one stdev = meaningful
    except StatisticsError:
        threshold = 0.1

    if delta > threshold:
        return "increasing"
    if delta < -threshold:
        return "decreasing"
    return "stable"


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """
    Compute the Pearson correlation coefficient between two equal-length lists.
    Returns None if the computation is undefined (zero variance).
    """
    n = len(xs)
    mx, my = mean(xs), mean(ys)

    num   = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs) ** 0.5
    den_y = sum((y - my) ** 2 for y in ys) ** 0.5

    if den_x == 0 or den_y == 0:
        return None

    return num / (den_x * den_y)


def _correlation_label(r: float) -> str:
    abs_r = abs(r)
    direction = "positive" if r >= 0 else "negative"
    if abs_r >= 0.7:
        strength = "strong"
    elif abs_r >= 0.4:
        strength = "moderate"
    else:
        strength = "weak"
    return f"{strength} {direction}"


def _compute_consistency(sorted_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Measure how regularly entries are being created.

    Computes:
    - total entries
    - date range (first → last)
    - average gap between entries (days)
    - consistency score 0–100 (higher = more regular)
    """
    if len(sorted_rows) < 2:
        return {
            "total_entries":   len(sorted_rows),
            "date_range_days": 0,
            "avg_gap_days":    None,
            "score":           0,
            "label":           "insufficient data",
        }

    def _to_dt(row: dict) -> datetime | None:
        s = row.get("_created", "")
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    timestamps = [_to_dt(r) for r in sorted_rows]
    timestamps = [t for t in timestamps if t is not None]

    if len(timestamps) < 2:
        return {"total_entries": len(sorted_rows), "score": 0,
                "label": "insufficient data"}

    gaps_days = [
        (timestamps[i] - timestamps[i - 1]).total_seconds() / 86_400
        for i in range(1, len(timestamps))
    ]

    total_days = (timestamps[-1] - timestamps[0]).total_seconds() / 86_400
    avg_gap    = mean(gaps_days)

    # Score: perfect daily = 100; avg gap of 7+ days = ~0
    raw_score = max(0.0, 100.0 * (1.0 - (avg_gap - 1.0) / 6.0))
    score     = min(100, round(raw_score))

    if score >= 80:
        label = "excellent"
    elif score >= 60:
        label = "good"
    elif score >= 40:
        label = "moderate"
    else:
        label = "irregular"

    return {
        "total_entries":   len(timestamps),
        "date_range_days": round(total_days, 1),
        "avg_gap_days":    round(avg_gap, 2),
        "score":           score,
        "label":           label,
    }


# ── Observations generator ────────────────────────────────────────────────────

def _build_observations(
    stats: dict[str, Any],
    patterns: dict[str, Any],
) -> list[str]:
    """
    Translate numeric stats and pattern dicts into plain-English sentences.

    Returns a list of observation strings — easy to print or display.
    """
    obs: list[str] = []

    # Consistency observation
    c = patterns.get("consistency", {})
    if c.get("label") not in (None, "insufficient data"):
        obs.append(
            f"Entry consistency is {c['label']} "
            f"(score {c['score']}/100, avg gap {c.get('avg_gap_days')} days)."
        )

    # Trend observations
    for field, direction in patterns.get("trends", {}).items():
        if direction not in ("insufficient data", "stable"):
            obs.append(f"'{field}' has been {direction} over time.")

    # Correlation observations (top 3 only to avoid noise)
    for corr in patterns.get("correlations", [])[:3]:
        obs.append(
            f"'{corr['field_a']}' and '{corr['field_b']}' show a "
            f"{corr['strength']} correlation (r={corr['pearson_r']}, "
            f"n={corr['sample_size']})."
        )

    # High-frequency keywords
    for field_name, field_stats in stats.items():
        keywords = field_stats.get("top_keywords", [])
        if keywords:
            top = ", ".join(f"'{w}'" for w, _ in keywords[:3])
            obs.append(f"Most frequent words in '{field_name}': {top}.")

    if not obs:
        obs.append("Not enough data yet to generate observations.")

    return obs


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_database(database_name: str) -> dict[str, Any]:
    """
    Full analysis pipeline for one Notion database.

    Steps:
        1. Load local schema
        2. Fetch all pages from Notion (paginated via query_database)
        3. Flatten each page into a row dict
        4. Compute basic statistics per field
        5. Detect trends, correlations, consistency
        6. Generate plain-English observations

    Args:
        database_name: Must match a locally saved schema with a
                       notion_database_id (i.e. the DB exists in Notion).

    Returns:
        Structured insights dict:
        {
            "database":          str,
            "total_entries":     int,
            "averages":          {field: mean} for numeric fields,
            "field_stats":       {field: stats_dict},
            "top_keywords":      [(word, count), ...],
            "consistency_score": int,
            "trends":            {field: direction},
            "correlations":      [...],
            "observations":      [str, ...],
        }

    Raises:
        SchemaMissingError:  if no local schema exists.
        AnalysisError:       if the Notion DB ID is missing or query fails.
    """
    schema = load_schema(database_name)
    fields: list[dict] = schema["fields"]

    db_id: str = schema.get("notion_database_id", "").strip()
    if not db_id:
        raise AnalysisError(
            f"Database '{database_name}' has no Notion ID in its schema. "
            "Create the database first before running analysis."
        )

    # ── Fetch all pages ───────────────────────────────────────────────────────
    logger.info("Fetching pages for analysis | db='%s' id=%s",
                database_name, db_id)
    try:
        pages = query_database(db_id)
    except Exception as exc:
        raise AnalysisError(
            f"Failed to fetch data from Notion for '{database_name}': {exc}"
        ) from exc

    if not pages:
        logger.warning("analyze_database: no entries found in '%s'.", database_name)
        return {
            "database":          database_name,
            "total_entries":     0,
            "averages":          {},
            "field_stats":       {},
            "top_keywords":      [],
            "consistency_score": 0,
            "trends":            {},
            "correlations":      [],
            "observations":      ["No entries found in this database yet."],
        }

    # ── Flatten pages ─────────────────────────────────────────────────────────
    rows = [_page_to_flat(p, fields) for p in pages]
    logger.info("Flattened %d pages for analysis.", len(rows))

    # ── Compute stats ─────────────────────────────────────────────────────────
    stats    = compute_basic_stats(rows, fields)
    patterns = detect_patterns(rows, fields)

    # ── Convenience: pull averages to top level ───────────────────────────────
    averages = {
        name: s["mean"]
        for name, s in stats.items()
        if s.get("type") == "number" and "mean" in s
    }

    # ── Top keywords across ALL text fields ───────────────────────────────────
    all_keywords: Counter = Counter()
    for field_stats in stats.values():
        for word, count in field_stats.get("top_keywords", []):
            all_keywords[word] += count
    top_keywords = all_keywords.most_common(10)

    # ── Plain-English observations ────────────────────────────────────────────
    observations = _build_observations(stats, patterns)

    consistency = patterns.get("consistency", {})

    result = {
        "database":          database_name,
        "total_entries":     len(rows),
        "averages":          averages,
        "field_stats":       stats,
        "top_keywords":      top_keywords,
        "consistency_score": consistency.get("score", 0),
        "consistency_label": consistency.get("label", "unknown"),
        "trends":            patterns.get("trends", {}),
        "correlations":      patterns.get("correlations", []),
        "observations":      observations,
    }

    logger.info(
        "Analysis complete | db='%s' entries=%d observations=%d",
        database_name, len(rows), len(observations),
    )
    return result


def print_report(insights: dict[str, Any]) -> None:
    """
    Print a formatted human-readable report from analyze_database() output.

    Args:
        insights: Dict returned by analyze_database().
    """
    sep   = "─" * 54
    heavy = "═" * 54

    print(f"\n\033[1m{heavy}\033[0m")
    print(f"\033[1m  Analysis Report: {insights['database']}\033[0m")
    print(f"\033[1m{heavy}\033[0m")
    print(f"  Total entries : {insights['total_entries']}")
    print(f"  Consistency   : {insights['consistency_label']} "
          f"({insights['consistency_score']}/100)")

    if insights["averages"]:
        print(f"\n  \033[96mAverages\033[0m")
        print(f"  {sep}")
        for field, avg in insights["averages"].items():
            print(f"    {field:<28} {avg}")

    if insights["trends"]:
        print(f"\n  \033[96mTrends\033[0m")
        print(f"  {sep}")
        for field, direction in insights["trends"].items():
            print(f"    {field:<28} {direction}")

    if insights["correlations"]:
        print(f"\n  \033[96mCorrelations\033[0m")
        print(f"  {sep}")
        for c in insights["correlations"]:
            print(
                f"    {c['field_a']} ↔ {c['field_b']}: "
                f"{c['strength']}  (r={c['pearson_r']}, n={c['sample_size']})"
            )

    if insights["top_keywords"]:
        print(f"\n  \033[96mTop Keywords\033[0m")
        print(f"  {sep}")
        kw_line = ",  ".join(f"{w} ({n})" for w, n in insights["top_keywords"])
        print(f"    {kw_line}")

    print(f"\n  \033[96mObservations\033[0m")
    print(f"  {sep}")
    for obs in insights["observations"]:
        print(f"    • {obs}")

    print(f"\033[1m{heavy}\033[0m\n")


# ── Custom exceptions ─────────────────────────────────────────────────────────

class AnalysisError(Exception):
    """Raised when analysis cannot proceed due to missing data or API failure."""
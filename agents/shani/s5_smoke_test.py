#!/usr/bin/env python3
r"""
s5_smoke_test.py — isolated single-paper S5 re-run to verify the JSON parser fix.

What this script does:
  1. Copies the live research_workflow.db to a throwaway temp path
  2. In the COPY only:
       - clears paper 10's existing ResearchKnowledge + ResearchRelation rows
       - sets paper 10 back to status='extracted' (so S5 picks it up)
       - sets all other workflow-1 papers to status='knowledge_ready' (so S5 skips them)
  3. Monkeypatches tools.normalise_paper_content.DB_PATH and VectorDBService
     to point at the copy + a throwaway FAISS index
  4. Runs extract_research_knowledge against the copy
  5. Reports knowledge count before (baseline: 58) vs. after the fix
  6. Does NOT touch the production DB at any point

USAGE:
    python3 s5_smoke_test.py

The script is read-only with respect to your production database.
All writes go to a temp copy that is deleted at exit.
"""

import os
import sys
import shutil
import sqlite3
import tempfile
import importlib
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SHANI_ROOT   = Path("/mnt/d/brahm/agents/shani")
LIVE_DB      = SHANI_ROOT / "database" / "research_workflow.db"
LIVE_FAISS   = SHANI_ROOT / "database" / "vector_index.faiss"
LIVE_MAP_NPY = SHANI_ROOT / "database" / "vector_index.faiss.map.npy"

WORKFLOW_ID  = 1
PAPER_ID     = 10
BASELINE_KNOWLEDGE = 58
BASELINE_RELATIONS = 289

# ── Bootstrap SHANI's import path ─────────────────────────────────────────────
sys.path.insert(0, str(SHANI_ROOT))

# ── Step 1: Create a temp working directory with copies of everything S5 touches ──
tmpdir = Path(tempfile.mkdtemp(prefix="s5_smoke_"))
print(f"[SMOKE] Temp dir: {tmpdir}")

temp_db      = tmpdir / "research_workflow.db"
temp_faiss   = tmpdir / "vector_index.faiss"
temp_map_npy = tmpdir / "vector_index.faiss.map.npy"

shutil.copy2(LIVE_DB, temp_db)
print(f"[SMOKE] Copied DB to: {temp_db}")

if LIVE_FAISS.exists():
    shutil.copy2(LIVE_FAISS, temp_faiss)
if LIVE_MAP_NPY.exists():
    shutil.copy2(LIVE_MAP_NPY, temp_map_npy)
print(f"[SMOKE] Copied FAISS index to temp dir")

# ── Step 2: Prepare the temp DB ───────────────────────────────────────────────
conn = sqlite3.connect(str(temp_db))
cur  = conn.cursor()

# Record baseline counts from the temp copy (should match production)
cur.execute("SELECT COUNT(*) FROM ResearchKnowledge WHERE paper_id=?", (PAPER_ID,))
pre_knowledge = cur.fetchone()[0]

cur.execute("SELECT COUNT(*) FROM ResearchRelation WHERE paper_id=?", (PAPER_ID,))
pre_relations = cur.fetchone()[0]

print(f"\n[SMOKE] Baseline counts in temp DB:")
print(f"  ResearchKnowledge (paper {PAPER_ID}): {pre_knowledge}  (expected {BASELINE_KNOWLEDGE})")
print(f"  ResearchRelation  (paper {PAPER_ID}): {pre_relations}  (expected {BASELINE_RELATIONS})")

if pre_knowledge != BASELINE_KNOWLEDGE:
    print(f"[WARN] Baseline knowledge count {pre_knowledge} != expected {BASELINE_KNOWLEDGE}. "
          f"Production DB may have changed since last session.")

# Clear existing knowledge/relations for paper 10 so we get a clean re-extraction
cur.execute("DELETE FROM ResearchKnowledge WHERE paper_id=?", (PAPER_ID,))
cur.execute("DELETE FROM ResearchRelation  WHERE paper_id=?", (PAPER_ID,))

# Reset paper 10 to 'extracted' so S5 picks it up
cur.execute("UPDATE Paper SET status='extracted' WHERE id=?", (PAPER_ID,))

# Set all OTHER workflow-1 papers that are currently 'extracted' to 'knowledge_ready'
# so S5 processes ONLY paper 10
cur.execute(
    "UPDATE Paper SET status='knowledge_ready' WHERE workflow_id=? AND status='extracted' AND id!=?",
    (WORKFLOW_ID, PAPER_ID)
)

# Verify the isolation
cur.execute("SELECT COUNT(*) FROM Paper WHERE workflow_id=? AND status='extracted'", (WORKFLOW_ID,))
target_count = cur.fetchone()[0]
print(f"\n[SMOKE] Papers with status='extracted' after isolation: {target_count} (expected 1)")
assert target_count == 1, f"Isolation failed: {target_count} papers are 'extracted'"

conn.commit()
conn.close()
print(f"[SMOKE] Temp DB prepared cleanly.")

# ── Step 3: Monkeypatch hardcoded paths ───────────────────────────────────────

# normalise_paper_content.py has a hardcoded DB_PATH at module level.
# We must patch it BEFORE importing tools that import it, or the import will
# cache the module with the live path baked in.
import tools.normalise_paper_content as _norm_mod
_norm_mod.DB_PATH = str(temp_db)
print(f"\n[SMOKE] Monkeypatched normalise_paper_content.DB_PATH → {temp_db}")

# VectorDBService defaults to the live FAISS index. We'll pass the temp path
# directly at instantiation — see the monkey-patch below via importlib.
# We override the module-level default so any internal instantiation without
# arguments also goes to the temp index.
import services.vector_db_service as _vdb_mod
_vdb_mod._DEFAULT_INDEX = str(temp_faiss)
print(f"[SMOKE] Monkeypatched vector_db_service._DEFAULT_INDEX → {temp_faiss}")

# ── Step 4: Wire Repository to the temp DB and run S5 ─────────────────────────
from repositories.repository import Repository
from tools.extract_research_knowledge import extract_research_knowledge

repo = Repository(db_path=str(temp_db))
print(f"\n[SMOKE] Running S5 against temp DB...")
print(f"[SMOKE] ─────────────────────────────────────────────\n")

try:
    result = extract_research_knowledge(repo, WORKFLOW_ID)
finally:
    repo.close()

print(f"\n[SMOKE] ─────────────────────────────────────────────")
print(f"[SMOKE] S5 result: {result.get('status')} | error: {result.get('error')}")

# ── Step 5: Count results in temp DB ──────────────────────────────────────────
conn = sqlite3.connect(str(temp_db))
cur  = conn.cursor()

cur.execute("SELECT COUNT(*) FROM ResearchKnowledge WHERE paper_id=?", (PAPER_ID,))
post_knowledge = cur.fetchone()[0]

cur.execute("SELECT COUNT(*) FROM ResearchRelation WHERE paper_id=?", (PAPER_ID,))
post_relations = cur.fetchone()[0]

# Also show category breakdown so we can see what the fix actually recovered
cur.execute(
    "SELECT category, COUNT(*) as cnt FROM ResearchKnowledge WHERE paper_id=? GROUP BY category ORDER BY cnt DESC",
    (PAPER_ID,)
)
categories = cur.fetchall()

conn.close()

print(f"\n[SMOKE] ══════════════════════════════════════════════")
print(f"[SMOKE] RESULTS — paper {PAPER_ID}")
print(f"[SMOKE]   ResearchKnowledge: {pre_knowledge} → {post_knowledge}  "
      f"({'▲ +' + str(post_knowledge - pre_knowledge) if post_knowledge >= pre_knowledge else '▼ ' + str(post_knowledge - pre_knowledge)} items)")
print(f"[SMOKE]   ResearchRelation:  {pre_relations} → {post_relations}")
print(f"\n[SMOKE] Category breakdown:")
for cat, cnt in categories:
    print(f"  {cat:<28} {cnt}")
print(f"[SMOKE] ══════════════════════════════════════════════")

if post_knowledge > pre_knowledge:
    print(f"\n[SMOKE] ✅ Fix confirmed: recovered {post_knowledge - pre_knowledge} previously dropped knowledge items.")
elif post_knowledge == pre_knowledge:
    print(f"\n[SMOKE] ⚠️  Same count as baseline. Fix may not have triggered on this paper, "
          f"or all chunks were already parsing cleanly.")
else:
    print(f"\n[SMOKE] ❌ Fewer items than baseline — investigate before proceeding.")

# ── Step 6: Cleanup temp dir ──────────────────────────────────────────────────
shutil.rmtree(tmpdir)
print(f"[SMOKE] Temp dir cleaned up. Production DB was not touched.")

"""
test_gap_detection_real.py
============================
Real-DB test for find_research_gap. READ-ONLY — only calls fetch_all(),
never touches transaction(). Safe to run against the live MoS2 workflow.
"""

import sys
sys.path.insert(0, "/mnt/d/brahm")
sys.path.insert(0, "/mnt/d/brahm/agents/shani")

from dotenv import load_dotenv
load_dotenv("/mnt/d/brahm/agents/shani/.env")

from repositories.repository import Repository
from brahm.mcp.gap_detection import find_research_gap

WORKFLOW_ID = 1  # MoS2 FET Synthesis and Optoelectronics

def main():
    repo = Repository()
    result = find_research_gap(repo, WORKFLOW_ID)
    repo.close()

    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"status: {result['status']}")

    if result["status"] == "error":
        print(f"error: {result['error']}")
        return

    data = result["data"]
    print(f"\nGap statement:\n  {data.get('gap_statement')}")
    print(f"\nSupporting paper IDs: {data.get('supporting_paper_ids')}")
    print(f"\nReasoning:\n  {data.get('reasoning')}")

if __name__ == "__main__":
    main()

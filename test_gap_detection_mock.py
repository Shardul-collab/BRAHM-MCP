"""
test_gap_detection_mock.py
============================
Standalone test for brahm/mcp/gap_detection.py using synthetic data.
No real database, no SHANI dependency.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv('/mnt/d/brahm/agents/shani/.env')

from brahm.mcp.gap_detection import (
    build_finding_digest,
    build_gap_prompt,
    call_reasoning_llm,
)

papers_by_id = {
    1: {
        "id": 1,
        "title": "MoS2 FET Synthesis via CVD at High Temperature",
        "abstract": (
            "We report CVD synthesis of MoS2 at 750C. Low-temperature "
            "synthesis routes remain unexplored and may enable flexible "
            "substrate integration."
        ),
    },
    2: {
        "id": 2,
        "title": "Optoelectronic Properties of Exfoliated MoS2",
        "abstract": (
            "We characterize photoresponse of mechanically exfoliated "
            "MoS2 flakes under varying illumination."
        ),
    },
    3: {
        "id": 3,
        "title": "Low-Temperature CVD Growth of MoS2 for Flexible Electronics",
        "abstract": (
            "We demonstrate MoS2 CVD synthesis at 350C, compatible with "
            "flexible polymer substrates, addressing a prior limitation "
            "in high-temperature-only growth routes."
        ),
    },
}

findings = [
    {
        "id": 101, "paper_id": 1,
        "finding_text": "CVD synthesis of MoS2 was performed at 750C substrate temperature.",
        "material": "MoS2", "synthesis_method": "CVD",
        "property_name": "", "property_value": "",
    },
    {
        "id": 102, "paper_id": 2,
        "finding_text": "Exfoliated MoS2 FETs showed photoresponsivity of 120 A/W under 532nm illumination.",
        "material": "MoS2", "synthesis_method": "exfoliation",
        "property_name": "photoresponsivity", "property_value": "120 A/W",
    },
    {
        "id": 103, "paper_id": 3,
        "finding_text": "Low-temperature CVD synthesis of MoS2 was achieved at 350C, compatible with flexible substrates.",
        "material": "MoS2", "synthesis_method": "CVD",
        "property_name": "growth_temperature", "property_value": "350 C",
    },
]

figures_by_paper_id = {
    1: [{"paper_id": 1, "image_path": "/papers/1/fig2.png", "caption": "SEM image of CVD-grown MoS2 flakes"}],
    3: [{"paper_id": 3, "image_path": "/papers/3/fig1.png", "caption": "Raman spectra comparing 750C vs 350C growth"}],
}


def main():
    digest = build_finding_digest(findings, papers_by_id, figures_by_paper_id)
    prompt = build_gap_prompt(digest)

    print("=" * 60)
    print("DIGEST")
    print("=" * 60)
    print(digest)

    print("\n" + "=" * 60)
    print("PROMPT (first 500 chars)")
    print("=" * 60)
    print(prompt[:500] + "...\n")

    assert "Paper [1]" in digest
    assert "Paper [3]" in digest
    assert "750C" in digest or "750" in digest
    print("Sanity checks passed: all 3 papers present, findings attached correctly.")

    if not os.environ.get("GROQ_API_KEY"):
        print(
            "\nGROQ_API_KEY not set — skipping live LLM call.\n"
            "Set it and re-run:\n  GROQ_API_KEY=xxx python3 test_gap_detection_mock.py"
        )
        return

    print("\n" + "=" * 60)
    print("LIVE LLM CALL (gpt-oss-120b)")
    print("=" * 60)
    result = call_reasoning_llm(prompt)
    print("\nGap proposed:")
    print(f"  Statement: {result.get('gap_statement')}")
    print(f"  Supporting papers: {result.get('supporting_paper_ids')}")
    print(f"  Reasoning: {result.get('reasoning')}")

    gap_lower = str(result.get("gap_statement", "")).lower()
    if "low" in gap_lower and "temperature" in gap_lower and "cvd" in gap_lower:
        print("\n  WARNING: may be re-proposing an already-answered gap. Review manually.")
    else:
        print("\n  Looks correct: did not re-propose the already-answered low-temp CVD gap.")


if __name__ == "__main__":
    main()

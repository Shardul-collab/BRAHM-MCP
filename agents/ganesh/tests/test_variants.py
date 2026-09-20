import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ganesh.writing.grounding import Evidence            # noqa: E402
from ganesh.writing.variants import Writer, write_section  # noqa: E402

EV = [Evidence("E1", 13, "growth_temperature", "280 °C [substrate temperature]", "Grown at 280 °C.", "full text"),
      Evidence("E2", 14, "growth_temperature", "350 °C [substrate temperature]", "Grown at 350 °C.", "full text"),
      Evidence("E3", 9, "characterization", "XRD", "Phase was identified by XRD.", "full text")]


def fake_llm(prompt, max_tokens=400, model=None, temperature=0.3):
    if "EDITED SECTION" in prompt:          # polish: keep text, add an invented number once
        body = prompt.split("PARAGRAPHS\n", 1)[1].rsplit("\n\nEDITED SECTION", 1)[0]
        return body + " It also grew at 999 °C [E1]."
    if "REWRITTEN PARAGRAPH" in prompt:      # repair: drop the bad sentence
        return "Growth temperatures of 280 °C [E1] and 350 °C [E2] were reported."
    if "REVISED PARAGRAPH" in prompt:        # style pass
        return "Two groups grew films hot: one at 280 °C [E1], the other, strikingly, at 350 °C [E2]."
    ids = re.findall(r"\[(E\d+)\]", prompt)
    return ("Films were grown at 280 °C [E1] and 350 °C [E2]. A third report used 400 °C [E1]. "
            "Structure was checked by XRD [E3]." if "E1" in ids else "Phase was checked by XRD [E3].")


def test_variants_run_and_never_keep_unsupported_claims():
    for v in ("L3", "L4", "L5"):
        w = Writer(fake_llm, "small", "big", "In2Se3", {"In2Se3"})
        r = write_section(w, v, "Synthesis Methods", EV)
        assert "400" not in r["text"] and "999" not in r["text"], (v, r["text"])
        assert "[E1]" in r["text"]
        assert r["polish"].startswith("applied") or r["polish"].startswith("reverted")


def test_g5_assembly_keeps_sections_whole_and_resolves_citations():
    from ganesh.writing.assemble import render_document, grounded_abstract
    long_section = "Films were grown at 280 °C [E1]. " * 60            # ~2,000 chars
    secs = [{"section_name": "Synthesis Methods", "content": long_section}]
    lookup = {"E1": {"paper_id": 13, "title": "MBE of Mn2In2Se5", "year": 2026, "doi": "",
                     "value": "280 °C", "sentence": "Grown at 280 °C.", "source": "full text"}}
    abstract, dropped = grounded_abstract(
        lambda p, max_tokens=500: "Growth near 280 °C is common. Films melt at 1200 °C.", "T", secs)
    assert abstract == "Growth near 280 °C is common." and len(dropped) == 1
    doc, prov = render_document("T", abstract, secs, ["Research Gaps"], lookup)
    assert doc.count("Films were grown at 280 °C [1].") == 60              # nothing truncated
    assert "[1] MBE of Mn2In2Se5 (2026)." in doc and "could not be generated" in doc
    assert prov["trail"][0]["eid"] == "E1"


def test_flow_pass_is_skipped_when_the_backend_output_limit_is_too_small(monkeypatch):
    # Groq free tier: 1,000 output tokens/minute; a section needs ~1,800-3,600 (2026-09-12)
    import ganesh.writing.variants as V
    w = V.Writer(fake_llm, "groq:qwen/qwen3.8-27b", "groq:qwen/qwen3.8-27b", "In2Se3 films", {"In2Se3"})
    paras = [{"text": "Growth at 280 °C [E1]. " * 60, "evidence": EV}]
    r = w.polish("Synthesis Methods", paras)
    assert r["polish"].startswith("skipped (") and "1000 allowed" in r["polish"]
    assert r["text"] == paras[0]["text"].strip() or r["text"] == "\n\n".join(p["text"] for p in paras)
    assert not any(c["step"] == "polish" for c in w.log)

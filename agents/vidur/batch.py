"""Folder -> plan -> plot-ready CSVs.

Two phases, because an MCP tool call returns once and cannot wait for an answer:

  scan(folder)             walks, groups, guesses, and RETURNS THE QUESTIONS.
                           Writes nothing.
  process(folder, answers) writes the CSVs.

Uncertainty is an output, never a silent guess.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from extractor import extract
from auto_detector import detect
from router import route
import prep
import series as series_mod

# Not instrument data. The 2026-09-19 harness handed VIDUR a README.md and it
# tried to parse it as UV-Vis.
_SKIP_SUFFIX = {".md", ".json", ".yaml", ".yml", ".py", ".pdf", ".docx", ".xlsx",
                ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".zip", ".gz"}
_SKIP_NAME = {"readme", "license", "notes", "index"}

KNOWN_TECHNIQUES = ("XRD", "Raman", "UV-Vis", "SEM_EDX", "IV", "IT")
# Named so a question can say "I do not know this one yet" rather than guess.
NOT_YET_SUPPORTED = ("XPS", "gas_sensing", "PL")


def _evidence(p: Path) -> dict:
    """What an operator needs to answer "what technique is this?" without opening
    the file. VIDUR is interface-less: the caller is a model holding these tools,
    so a question has to travel with its evidence or it cannot be answered."""
    import series as _s
    ev = {"name_tokens": _s.tokenise(p.stem), "suffix": p.suffix.lower(),
          "size_bytes": p.stat().st_size if p.exists() else 0,
          "head": [], "column_labels": None, "n_columns": None,
          "first_data_row": None}
    try:
        with open(p, "r", errors="ignore") as fh:
            lines = [next(fh, "").rstrip("\n") for _ in range(40)]
        lines = [x for x in lines if x != ""]
        ev["head"] = [x[:160] for x in lines[:12]]
        import re as _re
        for line in lines:
            fields = _re.split(r"[,;\t]", line)
            if len(fields) < 2:
                continue
            try:
                float(fields[0].replace(",", "."))
            except ValueError:
                ev["column_labels"] = [x.strip()[:24] for x in fields[:24]]
                ev["n_columns"] = len(fields)
                continue
            ev["first_data_row"] = [x.strip()[:24] for x in fields[:24]]
            ev["n_columns"] = ev["n_columns"] or len(fields)
            break
    except (UnicodeDecodeError, OSError):
        ev["head"] = ["<binary>"]
    return ev


def _is_data_candidate(p: Path) -> bool:
    return (p.is_file() and not p.name.startswith(".")
            and p.suffix.lower() not in _SKIP_SUFFIX
            and p.stem.lower() not in _SKIP_NAME)


def scan(folder: str | Path) -> dict:
    """Walk `folder`, identify each file, group into series, and list what is
    unresolved. Writes nothing."""
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(str(folder))

    files, skipped = [], []
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        if not _is_data_candidate(p):
            skipped.append({"file": p.name, "why": "not instrument data by name/extension"})
            continue
        entry = {"file": p.name, "technique": None, "confidence": 0.0,
                 "best_guess": None, "points": 0, "axis_name": None,
                 "axis_calibrated": None, "error": None}
        try:
            d = extract(str(p))
            det = detect(d)
            res = route(det, d)
            pd_ = res.get("parsed_data") or {}
            entry.update(technique=res.get("technique"),
                         confidence=round(det.get("confidence", 0.0), 3),
                         best_guess=res.get("best_guess"),
                         points=len(pd_.get("axis") or []),
                         axis_name=pd_.get("axis_name"),
                         axis_calibrated=(pd_.get("metadata") or {}).get("axis_calibrated"),
                         error=res.get("error"))
            entry["wavelength_a"] = (pd_.get("metadata") or {}).get("wavelength_a")
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        files.append(entry)

    grouped = series_mod.group([Path(f["file"]).stem for f in files])
    stem_to_sample = {m["stem"]: (g["series"], m["sample"])
                      for g in grouped for m in g["members"]}
    for f in files:
        s, sample = stem_to_sample.get(Path(f["file"]).stem, ("", ""))
        f["series"], f["sample"] = s, sample

    # Evidence is expensive to carry: 17 files x (12 head lines + 21 column
    # labels) made one scan unreadable through the live connector. Only a file
    # that raises a question needs it -- measured 2026-09-19.
    def _needs_evidence(f):
        return (f["technique"] in (None, "Unknown", "Uncertain")
                or not f["points"] or f["axis_calibrated"] is False)

    for f in files:
        if _needs_evidence(f):
            f["evidence"] = _evidence(folder / f["file"])

    questions = []
    for f in files:
        if f["technique"] in (None, "Unknown", "Uncertain"):
            questions.append({
                "file": f["file"], "kind": "technique",
                "ask": (f"I cannot tell what technique this is"
                        + (f" (best guess {f['best_guess']}, confidence {f['confidence']})"
                           if f.get("best_guess") else "")
                        + ". What is it?"),
                "known": list(KNOWN_TECHNIQUES),
                "not_yet_supported": list(NOT_YET_SUPPORTED),
                "evidence": f.get("evidence"),
            })
        elif f["points"] == 0:
            questions.append({"file": f["file"], "kind": "unreadable",
                              "ask": f"Identified as {f['technique']} but produced no "
                                     f"data: {f['error']}"})
        elif f["axis_calibrated"] is False:
            questions.append({"file": f["file"], "kind": "uncalibrated",
                              "ask": "The file does not carry its axis calibration, so "
                                     "the x column is channel index, not a physical unit."})

    ungrouped = [g["members"][0]["stem"] for g in grouped if not g["inferred"]]
    if ungrouped:
        # One question, not one per file: nine identical asks in a row is noise.
        questions.append({"file": None, "kind": "series", "files": ungrouped,
                          "ask": f"{len(ungrouped)} files did not group with any other, so each "
                                 "one's whole name is used as its sample name. Correct?"})

    needs = {}
    xrd_files = [f for f in files if f["technique"] == "XRD" and f["points"]]
    if xrd_files and not all(f.get("wavelength_a") for f in xrd_files):
        needs["wavelength_a"] = ("XRD d-spacing and q columns need the wavelength for the "
                                 "files that do not state one. Cu Ka1 is 1.5406 A but is "
                                 "not assumed.")
    if any(f["technique"] == "UV-Vis" and f["points"] for f in files):
        needs["thickness_cm"] = ("UV-Vis alpha and Tauc columns need the film thickness. "
                                 "Without it absorbance is NOT substituted for alpha.")
        needs["y_kind"] = "absorbance | transmittance_pct | reflectance"

    return {"folder": str(folder), "files": files, "series": grouped,
            "skipped": skipped, "questions": questions, "parameters_needed": needs}


def _write_csv(path: Path, columns: dict, axis_name: str, axis: np.ndarray) -> None:
    names = [axis_name] + list(columns)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(names)
        for i in range(len(axis)):
            w.writerow([axis[i]] + [columns[c][i] for c in columns])


def process(folder: str | Path, answers: dict | None = None,
            out_dir: str | Path | None = None) -> dict:
    """Write plot-ready CSVs. `answers` may carry:

        {"techniques": {filename: "XRD"},          # override a guess
         "params": {"wavelength_a": 1.5406, ...},  # global
         "file_params": {filename: {...}}}         # per file, wins
    """
    answers = answers or {}
    folder = Path(folder)
    out = Path(out_dir) if out_dir else folder / "vidur_out"
    out.mkdir(parents=True, exist_ok=True)

    plan = scan(folder)
    overrides = answers.get("techniques", {})
    gparams = answers.get("params", {})
    fparams = answers.get("file_params", {})

    written, skipped, notes = [], [], []
    per_series: dict[str, list[tuple[str, np.ndarray, np.ndarray, str]]] = {}

    for f in plan["files"]:
        technique = overrides.get(f["file"], f["technique"])
        if technique in (None, "Unknown", "Uncertain") or not f["points"]:
            skipped.append({"file": f["file"],
                            "why": f["error"] or f"technique unresolved ({technique})"})
            continue
        if technique in NOT_YET_SUPPORTED:
            skipped.append({"file": f["file"],
                            "why": f"{technique} has no parser in VIDUR yet"})
            continue

        d = extract(str(folder / f["file"]))
        det = detect(d)
        res = route(det, d)
        pd_ = res.get("parsed_data") or {}
        axis = np.asarray(pd_["axis"], dtype=float)
        y = np.asarray(pd_["intensity"], dtype=float)

        params = dict(gparams)
        # What the file says about its own axis outranks the caller's constants.
        md = pd_.get("metadata") or {}
        if "axis_is_two_theta" in md:
            params["axis_is_two_theta"] = md["axis_is_two_theta"]
        if md.get("wavelength_a"):
            params["wavelength_a"] = md["wavelength_a"]   # the file outranks the caller
        params.update(fparams.get(f["file"], {}))
        cols, n = prep.derive(technique, axis, y, params)
        notes.extend({"file": f["file"], "note": x} for x in n)

        csv_path = out / f"{Path(f['file']).stem}.csv"
        _write_csv(csv_path, cols, pd_.get("axis_name", "x"), axis)
        written.append({"file": f["file"], "csv": csv_path.name,
                        "technique": technique, "rows": len(axis),
                        "columns": [pd_.get("axis_name", "x")] + list(cols)})
        per_series.setdefault(f["series"], []).append(
            (f["sample"], axis, cols["norm_max"], pd_.get("axis_name", "x")))

    # Wide CSV per series, but only where the x axes actually agree -- true for a
    # growth series on one diffractometer, often false for Raman.
    wide = []
    for name, members in per_series.items():
        if len(members) < 2:
            continue
        ref = members[0][1]
        # A source-meter never returns the same setpoint twice: -1.0000072717667
        # against -0.9999924898148 is the SAME nominal -1 V. Demanding equality
        # to 1e-6 refused a wide CSV for every I-V series -- exactly the figure
        # this is for. Points are paired when they agree to 0.5% of the axis
        # span, and the largest deviation is recorded so nothing is hidden. No
        # values are resampled or altered; only rows are paired.
        span = float(np.nanmax(ref) - np.nanmin(ref)) or 1.0
        tol = 0.005 * span
        aligned = all(len(m[1]) == len(ref) and np.allclose(m[1], ref, rtol=0, atol=tol)
                      for m in members)
        if aligned:
            dev = max(float(np.nanmax(np.abs(m[1] - ref))) for m in members)
            path = out / f"{name}_wide.csv"
            cols = {s: y for s, _, y, _ in members}
            _write_csv(path, cols, members[0][3], ref)
            wide.append({"series": name, "csv": path.name,
                         "samples": [s for s, *_ in members], "value": "norm_max",
                         "axis_from": members[0][0],
                         "max_axis_deviation": dev,
                         "axis_tolerance": tol})
        else:
            notes.append({"file": name,
                          "note": f"wide CSV skipped: sample x-axes differ by more than "
                                  f"{tol:.4g} (0.5% of the axis span)"})

    meta = {"folder": str(folder), "out_dir": str(out), "written": written,
            "wide": wide, "skipped": skipped, "notes": notes,
            "questions_outstanding": [q for q in plan["questions"]
                                      if q.get("file") not in overrides]}
    (out / "_vidur_meta.json").write_text(json.dumps(meta, indent=2))
    return meta

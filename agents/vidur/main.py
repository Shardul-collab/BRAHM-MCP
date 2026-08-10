#!/usr/bin/env python3
# main.py
#
# Characterization Classifier Agent — Entry Point
#
# Usage:
#   python main.py <file_path>
#   python main.py <file_path> --verbose
#   python main.py <file_path> --json
#
# Pipeline:
#   file → extractor → auto_detector → router → parser → JSON output

import sys
import os
import json
import logging
import argparse


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  [%(name)s]  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def run_pipeline(file_path: str) -> dict:
    """
    Full pipeline: file → structured result dict.

    Returns:
        {
            "technique":   str,
            "confidence":  float,
            "signals":     list,
            "parsed_data": dict | None,
            "error":       str | None,
        }
    """
    # Validate file exists
    if not os.path.isfile(file_path):
        return {
            "technique":   "Unknown",
            "confidence":  0.0,
            "signals":     [],
            "parsed_data": None,
            "error":       f"File not found: {file_path}",
        }

    # Step 1: Extract text + numeric signals
    from extractor import extract
    data = extract(file_path)

    # Step 2: Auto-detect technique
    from auto_detector import detect
    detection = detect(data)

    # Step 3: Route to parser and parse
    from router import route
    result = route(detection, data)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="VIDUR — Characterization Classifier: "
                    "auto-detects and parses scientific instrument files."
    )
    parser.add_argument("file_path", help="Path to the input file")
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--json", "-j", action="store_true",
        help="Print raw JSON output (no formatting)"
    )
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    log = logging.getLogger("vidur.main")

    log.info(f"Processing: {args.file_path}")

    result = run_pipeline(args.file_path)

    # Output
    if args.json:
        print(json.dumps(result, indent=None))
    else:
        _pretty_print(result)

    # Exit code: 0 on success, 1 on error/unknown
    sys.exit(0 if result.get("parsed_data") is not None else 1)


def _pretty_print(result: dict):
    """Human-readable summary of the pipeline result."""
    tech       = result.get("technique", "?")
    confidence = result.get("confidence", 0.0)
    signals    = result.get("signals", [])
    parsed     = result.get("parsed_data")
    error      = result.get("error")

    bar = "─" * 55
    print(f"\n{bar}")
    print(f"  VIDUR — Characterization Classifier")
    print(bar)
    print(f"  Technique  : {tech}")
    print(f"  Confidence : {confidence:.2%}")

    if signals:
        print(f"  Signals    :")
        for s in signals:
            print(f"    • {s}")

    if error:
        print(f"  ⚠  Error    : {error}")

    if parsed:
        n_pts = len(parsed.get("axis", []))
        print(f"\n  Parsed Data:")
        print(f"    axis_name  : {parsed.get('axis_name', '?')}")
        print(f"    data_points: {n_pts}")
        meta = parsed.get("metadata", {})
        if meta:
            for k, v in meta.items():
                if k != "units":
                    print(f"    {k:<12}: {v}")

        # Preview first few + last few values
        axis      = parsed.get("axis", [])
        intensity = parsed.get("intensity", [])
        if axis and intensity:
            preview_n = min(3, len(axis))
            print(f"\n  Data Preview (first {preview_n} rows):")
            print(f"    {'axis':>12}  {'intensity':>14}")
            for i in range(preview_n):
                print(f"    {axis[i]:>12.4f}  {intensity[i]:>14.4f}")
            if len(axis) > preview_n:
                print(f"    {'...':>12}  {'...':>14}")

    print(f"{bar}\n")

    # Machine-readable JSON block
    import json
    print("JSON output:")
    # Truncate axis/intensity for readability
    compact = dict(result)
    if compact.get("parsed_data"):
        pd_copy = dict(compact["parsed_data"])
        for key in ("axis", "intensity"):
            arr = pd_copy.get(key, [])
            if len(arr) > 6:
                pd_copy[key] = arr[:3] + ["..."] + arr[-3:]
        compact["parsed_data"] = pd_copy
    # Remove verbose candidates list
    compact.pop("candidates", None)
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()

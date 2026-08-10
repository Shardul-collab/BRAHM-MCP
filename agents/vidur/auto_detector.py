# auto_detector.py
#
# Registers all parsers, calls can_parse() on each,
# ranks the scores, and returns a structured detection result.

import logging

logger = logging.getLogger("vidur.detector")

# Minimum confidence to declare a technique (not "Uncertain")
CONFIDENCE_THRESHOLD = 0.6

# All registered parsers: (technique_name, parser_module)
# Import lazily inside detect() to keep startup clean.
def _load_parsers() -> list[tuple[str, object]]:
    from parsers import xrd, uvvis, sem_eds, raman
    return [
        ("XRD",     xrd),
        ("UV-Vis",  uvvis),
        ("SEM_EDX", sem_eds),
        ("Raman",   raman),
    ]


def detect(data: dict) -> dict:
    """
    Run all parsers' can_parse() on the extracted data.
    Rank by score, apply threshold, return detection metadata.

    Args:
        data: output of extractor.extract()

    Returns:
        {
            "technique":   str,          # top-ranked technique or "Uncertain"/"Unknown"
            "confidence":  float,        # normalised score 0–1
            "signals":     list,         # signals from the winning parser
            "candidates":  list[dict],   # all parsers ranked by score
        }
    """
    parsers = _load_parsers()
    results = []

    for name, module in parsers:
        try:
            raw_score, signals = module.can_parse(data)
            results.append({
                "technique": name,
                "score":     round(float(raw_score), 4),
                "signals":   signals,
            })
            logger.debug(f"  [{name}] score={raw_score:.3f}  signals={signals}")
        except Exception as e:
            logger.warning(f"  [{name}] can_parse() raised: {e}")
            results.append({"technique": name, "score": 0.0, "signals": []})

    # Sort descending by score
    results.sort(key=lambda r: r["score"], reverse=True)

    best = results[0]

    # Normalise confidence: clip to [0, 1]
    confidence = min(best["score"], 1.0)

    # Decision logic
    if confidence < CONFIDENCE_THRESHOLD:
        technique = "Uncertain" if confidence > 0.0 else "Unknown"
        logger.info(
            f"Detection: {technique}  "
            f"(best_score={confidence:.3f} < threshold={CONFIDENCE_THRESHOLD})"
        )
    else:
        technique = best["technique"]
        logger.info(
            f"Detection: {technique}  confidence={confidence:.3f}  "
            f"signals={best['signals']}"
        )

    return {
        "technique":  technique,
        "confidence": confidence,
        "signals":    best["signals"],
        "candidates": results,          # full ranked list for debugging
    }

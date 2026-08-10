# router.py
#
# Selects the appropriate parser based on auto_detector output.
# Handles Unknown and Uncertain gracefully.

import logging

logger = logging.getLogger("vidur.router")

# Maps technique names to their parser modules (lazy-loaded on first call)
_PARSER_MAP = None

def _get_parser_map() -> dict:
    global _PARSER_MAP
    if _PARSER_MAP is None:
        from parsers import xrd, uvvis, sem_eds, raman
        _PARSER_MAP = {
            "XRD":     xrd,
            "UV-Vis":  uvvis,
            "SEM_EDX": sem_eds,
            "Raman":   raman,
        }
    return _PARSER_MAP


def route(detection: dict, data: dict) -> dict:
    """
    Route to the correct parser based on detection result.
    Parse the data and return structured output.

    Args:
        detection: output of auto_detector.detect()
        data:      output of extractor.extract()

    Returns:
        {
            "technique":   str,
            "confidence":  float,
            "signals":     list,
            "parsed_data": dict | None,
            "error":       str | None,
        }
    """
    technique  = detection["technique"]
    confidence = detection["confidence"]
    signals    = detection["signals"]

    # Base output shell
    result = {
        "technique":   technique,
        "confidence":  confidence,
        "signals":     signals,
        "parsed_data": None,
        "error":       None,
    }

    # Handle unresolvable cases
    if technique == "Unknown":
        msg = (
            f"Could not identify technique in '{data['filename']}'. "
            "No parser matched above the confidence threshold."
        )
        logger.warning(msg)
        result["error"] = msg
        return result

    if technique == "Uncertain":
        candidates = detection.get("candidates", [])
        top = candidates[0] if candidates else {}
        msg = (
            f"Technique is uncertain for '{data['filename']}'. "
            f"Best guess: {top.get('technique', '?')} "
            f"(score={top.get('score', 0):.3f}). "
            "Consider providing more context or a labelled filename."
        )
        logger.warning(msg)
        result["error"] = msg
        # Still attempt to parse with the best guess if it exists
        technique = top.get("technique", "")

    # Dispatch to parser
    parser_map = _get_parser_map()
    parser     = parser_map.get(technique)

    if parser is None:
        msg = f"No parser registered for technique '{technique}'."
        logger.error(msg)
        result["error"] = msg
        return result

    logger.info(f"Routing to [{technique}] parser for '{data['filename']}'")

    try:
        parsed = parser.parse(data)
        result["parsed_data"] = parsed
        # Promote the resolved technique (in case we fell through Uncertain)
        result["technique"] = parsed.get("technique", technique)
        logger.info(
            f"Parsed [{technique}]: "
            f"{len(parsed.get('axis', []))} data points"
        )
    except Exception as e:
        msg = f"Parser [{technique}] failed: {e}"
        logger.error(msg)
        result["error"] = msg

    return result

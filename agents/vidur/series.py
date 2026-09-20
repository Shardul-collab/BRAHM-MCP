"""Infer series structure from a set of filenames.

There is no fixed naming convention. The structure is inferred from the *set*:
tokens that are the same across a group name the series, tokens that vary name
the sample. `ZnSe_S1_UV`, `ZnSe_S2_UV`, `ZnSe_S3_UV` -> series "ZnSe_UV",
samples "S1", "S2", "S3".

Nothing here guesses what a token means. It only says which part varies.
"""
from __future__ import annotations

import re
from collections import defaultdict

_SEP = re.compile(r"[_\-\s]+")
# Runs of letters and runs of digits, for names written without separators.
_RUN = re.compile(r"\d+|[A-Za-z]+")
# "sample1" -> ("sample", "1"); leaves "S1"/"A" alone if there is no digit tail
_TAIL = re.compile(r"^(.*?)(\d+)$")


def tokenise(stem: str) -> list[str]:
    """Split a filename stem into tokens, separating a trailing number."""
    out = []
    parts = [x for x in _SEP.split(stem) if x]
    single_run = len(parts) == 1
    for part in parts:
        if not part:
            continue
        # Names written without separators carry their structure in the
        # letter/digit alternation: Shardul's photodetector files are
        # "16znse03iv150", i.e. 16 | znse | 03 | iv | 150, and without splitting
        # them every such file is a series of one. Split only when the whole
        # stem had no separators at all, so a name that IS punctuated keeps its
        # tokens whole -- that is what protects SnO2, TiO2 and In2Se3, which
        # would otherwise become ("SnO", "2").
        if single_run and len(_RUN.findall(part)) > 1:
            out.extend(_RUN.findall(part))
        else:
            out.append(part)
    return out


def group(stems: list[str]) -> list[dict]:
    """Group stems into series. Returns one dict per series:

        {"series": str, "members": [{"stem": str, "sample": str}, ...],
         "varying_positions": [int], "inferred": bool}

    `inferred` is False when a stem could not be grouped with anything and the
    whole stem is used as the sample name -- the caller should show those.
    """
    by_shape: dict[int, list[str]] = defaultdict(list)
    for s in stems:
        by_shape[len(tokenise(s))].append(s)

    # Same token count is not enough: ten unrelated files with four tokens each
    # would become one "series". Two stems join only if they agree, position by
    # position, on at least half their tokens.
    clusters: list[tuple[int, list[str]]] = []
    for n, members in sorted(by_shape.items()):
        toks = {s: tokenise(s) for s in members}
        # A series is "one thing varies": a stem joins when it agrees with the
        # cluster's first member on all but at most one position. Half-agreement
        # was too loose (four unrelated four-token names became one series);
        # exact agreement would never group anything.
        need = max(0, n - 1)
        for s in sorted(members):
            for _, group_members in clusters:
                rep = group_members[0]
                if len(toks.get(rep, tokenise(rep))) != n:
                    continue
                agree = sum(1 for i in range(n)
                            if toks[s][i].lower() == tokenise(rep)[i].lower())
                if agree >= need:
                    group_members.append(s)
                    break
            else:
                clusters.append((n, [s]))

    series = []
    for n, members in clusters:
        toks = {s: tokenise(s) for s in members}
        if len(members) == 1:
            s = members[0]
            series.append({"series": "_".join(toks[s]), "varying_positions": [],
                           "inferred": False,
                           "members": [{"stem": s, "sample": "_".join(toks[s])}]})
            continue

        varying = [i for i in range(n)
                   if len({toks[s][i].lower() for s in members}) > 1]
        if not varying:                      # identical stems, cannot happen with real paths
            varying = [n - 1]
        constant = [i for i in range(n) if i not in varying]
        name = "_".join(toks[members[0]][i] for i in constant) or "series"
        series.append({
            "series": name,
            "varying_positions": varying,
            "inferred": True,
            "members": [{"stem": s,
                         "sample": "_".join(toks[s][i] for i in varying)}
                        for s in sorted(members)],
        })
    return series

"""Text helpers shared by the runtime, the reference tools, and the evals."""

from __future__ import annotations

import unicodedata

GENERIC_WORDS = frozenset(
    {
        # legal forms
        "inc",
        "incorporated",
        "ltd",
        "limited",
        "llc",
        "llp",
        "lp",
        "plc",
        "corp",
        "corporation",
        "co",
        "company",
        "gmbh",
        "ag",
        "sa",
        "sas",
        "sarl",
        "srl",
        "spa",
        "bv",
        "nv",
        "ab",
        "as",
        "oy",
        "kk",
        "pty",
        "pvt",
        "ltda",
        "lda",
        "kg",
        "sl",
        "kft",
        "jsc",
        "ooo",
        "sae",
        "kaisha",
        "kabushiki",
        # filler
        "group",
        "holdings",
        "holding",
        "international",
        "intl",
        "global",
        "services",
        "service",
        "solutions",
        "systems",
        "industries",
        "industrial",
        "enterprises",
        "partners",
        "capital",
        "trading",
        "traders",
        "bank",
        "of",
        "the",
        "and",
        "et",
        "und",
        "de",
        "del",
        "des",
        "der",
        "do",
        "da",
        "van",
        "von",
        "brothers",
        "bros",
        "technologies",
        "technology",
        "tech",
        "consulting",
        "products",
        "logistics",
        "trust",
        "national",
        "first",
        "general",
        "united",
        "american",
        "european",
        "pacific",
        "atlantic",
        "worldwide",
        "associates",
        "assoc",
        "sons",
        "fils",
        "zonen",
    }
)


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalize(text: str) -> str:
    """Accent-stripped, case-folded, whitespace-collapsed. The key for all matching."""
    return " ".join(strip_accents(text).casefold().split())


def words(text: str) -> list[str]:
    """Normalized alphanumeric words, punctuation removed."""
    out = []
    for raw in normalize(text).split():
        w = "".join(c for c in raw if c.isalnum() or c == "&")
        if w:
            out.append(w)
    return out


def distinctive_words(text: str, *, min_len: int = 3) -> list[str]:
    """Words worth searching for on their own: not generic, not tiny, order preserved."""
    seen: set[str] = set()
    out = []
    for w in words(text):
        if w in GENERIC_WORDS or len(w) < min_len or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out


def trigrams(text: str) -> set[str]:
    padded = f"  {normalize(text)} "
    return {padded[i : i + 3] for i in range(len(padded) - 2)}


def similarity(a: str, b: str) -> float:
    """Trigram Jaccard similarity in [0, 1]. Cheap, symmetric, accent and case insensitive."""
    ta, tb = trigrams(a), trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def edit_distance(a: str, b: str, *, max_distance: int | None = None) -> int:
    """Levenshtein distance. Returns max_distance + 1 early when the bound is exceeded."""
    if a == b:
        return 0
    if max_distance is not None and abs(len(a) - len(b)) > max_distance:
        return max_distance + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            cost = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            cur.append(cost)
            row_min = min(row_min, cost)
        if max_distance is not None and row_min > max_distance:
            return max_distance + 1
        prev = cur
    return prev[-1]


def fuzzy_word_distance(term: str, name: str, *, max_distance: int) -> int | None:
    """Smallest edit distance between `term` and any word of `name`, or None if over the bound."""
    best: int | None = None
    for w in words(name):
        d = edit_distance(term, w, max_distance=max_distance)
        if d <= max_distance and (best is None or d < best):
            best = d
            if d == 0:
                break
    return best


def default_max_distance(term: str) -> int:
    """Typo tolerance by length: 0 for very short terms, 1 up to 5 letters, else 2."""
    n = len(term)
    return 0 if n <= 3 else 1 if n <= 5 else 2

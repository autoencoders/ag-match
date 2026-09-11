"""Synthetic dataset: a list to match against plus perturbed queries with known answers.

Everything is seeded, so the same seed always yields the same records and cases.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from ag_match import Record

from .seed_names import TARGETS

GENERIC_WORDS = {
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
    "de",
    "cv",
    "kg",
    "sl",
    "oü",
    "kft",
    "aş",
    "jsc",
    "ooo",
    "sp",
    "z",
    "o",
    "group",
    "holdings",
    "holding",
    "international",
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
    "&",
    "et",
    "fils",
    "und",
    "do",
    "del",
    "des",
    "der",
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
}

ABBREVIATIONS = {
    "International": "Intl",
    "Corporation": "Corp",
    "Company": "Co",
    "Limited": "Ltd",
    "Incorporated": "Inc",
    "Brothers": "Bros",
    "Manufacturing": "Mfg",
    "Technologies": "Tech",
    "Laboratories": "Labs",
    "Associates": "Assoc",
    "Industries": "Ind",
    "Engineering": "Eng",
    "Services": "Svcs",
    "Holdings": "Hldgs",
    "and": "&",
}
EXPANSIONS = {v: k for k, v in ABBREVIATIONS.items() if v != "&"}

LEGAL_FORMS = {
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
    "ltda",
    "lda",
    "oü",
    "kft",
    "aş",
    "jsc",
    "ooo",
    "sae",
    "kaisha",
    "a/s",
}

SUFFIX_SWAPS = {
    "Inc": ["Incorporated", "Inc.", ", Inc", ""],
    "Ltd": ["Limited", "Ltd.", "LTD", ""],
    "Corp": ["Corporation", "Corp.", ""],
    "Corporation": ["Corp", "Corp.", ""],
    "LLC": ["L.L.C.", "LLC.", ""],
    "plc": ["PLC", "Plc", ""],
    "GmbH": ["GmbH & Co. KG", "gmbh", ""],
    "AG": ["A.G.", ""],
    "SA": ["S.A.", "SA.", ""],
    "SpA": ["S.p.A.", ""],
    "BV": ["B.V.", ""],
    "NV": ["N.V.", ""],
    "AB": ["Aktiebolag", ""],
    "Co": ["Company", "Co.", ""],
    "Company": ["Co", "Co.", ""],
}

# Word pools for filler records. Distinct from target words so target searches stay
# predictable, and from the ABSENT pool so negatives are truly absent.
FILLER_DISTINCTIVE = [
    "Veltrix",
    "Corvano",
    "Halbrook",
    "Ostermann",
    "Quillen",
    "Brannock",
    "Sandoval",
    "Whitcombe",
    "Ravensworth",
    "Ibarra",
    "Lindqvist",
    "Marchetti",
    "Nakamura",
    "Oyelaran",
    "Petrakis",
    "Radovan",
    "Sørensen",
    "Tremblay",
    "Uribe",
    "Vasquez",
    "Wexler",
    "Yamamoto",
    "Zielinski",
    "Ashcroft",
    "Beaumont",
    "Castellano",
    "Dubrovnik",
    "Escalante",
    "Fairweather",
    "Galloway",
    "Hartigan",
    "Ivanova",
    "Jorgensen",
    "Kensington",
    "Larkspur",
    "Montclair",
    "Norwood",
    "Oakridge",
    "Pemberton",
    "Quintero",
    "Redfern",
    "Stonebridge",
    "Thornbury",
    "Underhill",
    "Valmont",
    "Westbrook",
    "Yardley",
    "Zamora",
    "Aldana",
    "Birchwood",
    "Caldera",
    "Dalloway",
    "Eastgate",
    "Falconer",
    "Greystone",
    "Hollister",
    "Ironside",
    "Juniper",
    "Kingsley",
    "Lockhart",
    "Maddox",
    "Newhaven",
    "Oberon",
    "Pinecrest",
    "Quarry",
    "Rutherford",
    "Sablewood",
    "Tanner",
    "Upton",
    "Vickers",
    "Wainwright",
    "Yeoman",
    "Zephyrine",
    "Ambrose",
    "Bellweather",
    "Crestline",
    "Dunmore",
    "Elmhurst",
    "Foxglove",
    "Glenmoor",
    "Harlow",
    "Inglewood",
    "Jasper",
    "Kestrel",
    "Lyndhurst",
    "Marlow",
    "Nightingale",
    "Orchard",
    "Prescott",
    "Quinlan",
    "Ridgeway",
    "Selwyn",
    "Tillman",
    "Umber",
    "Vale",
    "Whitfield",
]
ABSENT_DISTINCTIVE = [
    "Xanthippe",
    "Brumlow",
    "Kreznik",
    "Oduya",
    "Pfalzgraf",
    "Szczepan",
    "Voskuijlen",
    "Mbeki-Roux",
    "Tarquinio",
    "Hjalmarsson",
    "Quenneville",
    "Ybarrola",
    "Zsolnay",
    "Achterberg",
    "Bexley-Hume",
    "Cwmbran",
    "Drăgulescu",
    "Eyjafjalla",
    "Fitzwilliam",
]
FILLER_DESCRIPTOR = [
    "Industries",
    "Logistics",
    "Pharma",
    "Energy",
    "Capital",
    "Partners",
    "Foods",
    "Motors",
    "Textiles",
    "Shipping",
    "Consulting",
    "Engineering",
    "Mining",
    "Media",
    "Software",
    "Chemicals",
    "Aviation",
    "Construction",
    "Insurance",
    "Retail",
    "Packaging",
    "Robotics",
    "Biotech",
    "Timber",
    "Dairy",
    "Brewing",
    "Hospitality",
    "Analytics",
    "Security",
    "Freight",
]
FILLER_GENERIC = ["International", "Global", "Group", "Holdings", "Services", "Systems", ""]
FILLER_SUFFIX_BY_COUNTRY = {
    "US": ["Inc", "Corp", "LLC", "Co"],
    "GB": ["Ltd", "plc", "Limited"],
    "DE": ["GmbH", "AG", "GmbH & Co. KG"],
    "FR": ["SA", "SAS", "SARL"],
    "NL": ["BV", "NV"],
    "SE": ["AB"],
    "IT": ["SpA", "Srl"],
    "ES": ["SA", "SL"],
    "JP": ["KK", "Co Ltd"],
    "IN": ["Pvt Ltd", "Limited"],
    "AU": ["Pty Ltd"],
    "CA": ["Inc", "Ltd"],
    "BR": ["SA", "Ltda"],
}

POSITIVE_KINDS = [
    "exact",
    "typo",
    "two_typos",
    "suffix",
    "abbrev",
    "expand",
    "drop_word",
    "reorder",
    "case_punct",
    "accent",
    "combo",
    "with_context",
]
NEGATIVE_KINDS = ["absent", "decoy", "generic_only"]
ALL_KINDS = POSITIVE_KINDS + NEGATIVE_KINDS


@dataclass
class Case:
    id: str
    query: str
    kind: str
    expected_id: str | None
    context: dict[str, Any] | None = None
    source_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class Dataset:
    records: list[Record]
    cases: list[Case]
    seed: int
    by_id: dict[str, Record] = field(init=False)

    def __post_init__(self) -> None:
        self.by_id = {r.id: r for r in self.records}


# --- text helpers -------------------------------------------------------------------


def tokens(name: str) -> list[str]:
    return name.split()


def is_generic(token: str) -> bool:
    bare = re.sub(r"[^\w&]", "", token).casefold()
    return bare in GENERIC_WORDS or len(bare) < 4


def distinctive_tokens(name: str) -> list[str]:
    return [t for t in tokens(name) if not is_generic(t)]


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


ACCENT_MAP = {"u": "ü", "o": "ö", "a": "ä", "e": "é"}


def add_accent(word: str, rng: random.Random) -> str:
    positions = [i for i, c in enumerate(word) if c in ACCENT_MAP]
    if not positions:
        return word
    i = rng.choice(positions)
    return word[:i] + ACCENT_MAP[word[i]] + word[i + 1 :]


def typo(word: str, rng: random.Random) -> str:
    """One edit in a word: swap, drop, double, or replace a letter. Keeps first letter."""
    letters = [i for i, c in enumerate(word) if c.isalpha() and i > 0]
    if len(letters) < 2:
        return word + "e"
    op = rng.choice(["swap", "drop", "double", "replace"])
    i = rng.choice(letters[:-1] if op == "swap" else letters)
    if op == "swap":
        j = i + 1
        return word[:i] + word[j] + word[i] + word[j + 1 :]
    if op == "drop":
        return word[:i] + word[i + 1 :]
    if op == "double":
        return word[:i] + word[i] + word[i:]
    vowels = "aeiou"
    c = word[i]
    repl = rng.choice([v for v in vowels if v != c.lower()]) if c.lower() in vowels else "n"
    return word[:i] + (repl.upper() if c.isupper() else repl) + word[i + 1 :]


def replace_token(name: str, old: str, new: str) -> str:
    out = " ".join(new if t == old else t for t in tokens(name))
    return tidy(out)


def tidy(name: str) -> str:
    name = " ".join(name.split())
    name = re.sub(r"\s+,", ",", name)
    name = re.sub(r",\s*,+", ",", name)
    return name.strip(" ,")


# --- perturbations ------------------------------------------------------------------


def perturb(name: str, kind: str, rng: random.Random) -> str:
    toks = tokens(name)
    dist = distinctive_tokens(name)
    if kind == "exact":
        return name
    if kind == "typo":
        target = rng.choice(dist) if dist else toks[0]
        return replace_token(name, target, typo(target, rng))
    if kind == "two_typos":
        out = perturb(name, "typo", rng)
        return perturb(out, "typo", rng)
    if kind == "suffix":
        for t in reversed(toks):
            if t.strip(".,") in SUFFIX_SWAPS:
                new = rng.choice(SUFFIX_SWAPS[t.strip(".,")])
                out = replace_token(name, t, new)
                return out if out != name else tidy(name + " Inc")
        last = toks[-1].strip(".,").casefold()
        if last in LEGAL_FORMS:
            return tidy(" ".join(toks[:-1]))
        return name + " " + rng.choice(["Inc", "Ltd", "Corp", "Co"])
    if kind == "abbrev":
        out = name
        for full, short in ABBREVIATIONS.items():
            out = re.sub(rf"\b{re.escape(full)}\b", short, out)
        return out if out != name else perturb(name, "suffix", rng)
    if kind == "expand":
        out = name
        for short, full in EXPANSIONS.items():
            out = re.sub(rf"\b{re.escape(short)}\b\.?", full, out)
        return out if out != name else perturb(name, "abbrev", rng)
    if kind == "drop_word":
        generic = [t for t in toks if is_generic(t) and t != toks[0]]
        if not generic:
            return perturb(name, "suffix", rng)
        return replace_token(name, rng.choice(generic), "")
    if kind == "reorder":
        if len(dist) >= 2:
            a, b = dist[0], dist[1]
            swapped = [b if t == a else a if t == b else t for t in toks]
            return " ".join(swapped)
        if len(toks) >= 2:
            return " ".join(toks[1:2] + toks[:1] + toks[2:])
        return name
    if kind == "case_punct":
        choice = rng.choice(["upper", "lower", "strip", "dots"])
        if choice == "upper":
            return name.upper()
        if choice == "lower":
            return name.lower()
        if choice == "strip":
            return re.sub(r"[^\w\s]", "", name)
        return " ".join(t + "." if len(t) <= 4 and t.isalpha() else t for t in toks)
    if kind == "accent":
        stripped = strip_accents(name)
        if stripped != name:
            return stripped
        target = rng.choice(dist) if dist else toks[0]
        return replace_token(name, target, add_accent(target, rng))
    if kind == "combo":
        out = perturb(name, "typo", rng)
        out = perturb(out, "abbrev", rng)
        return perturb(out, "suffix", rng)
    if kind == "with_context":
        return perturb(name, rng.choice(["exact", "typo", "suffix"]), rng)
    raise ValueError(f"unknown kind {kind}")


# --- dataset construction -----------------------------------------------------------


def build_filler(n: int, rng: random.Random) -> list[tuple[str, str]]:
    out = []
    countries = list(FILLER_SUFFIX_BY_COUNTRY)
    for _ in range(n):
        country = rng.choice(countries)
        parts = [rng.choice(FILLER_DISTINCTIVE)]
        if rng.random() < 0.7:
            parts.append(rng.choice(FILLER_DESCRIPTOR))
        generic = rng.choice(FILLER_GENERIC)
        if generic:
            parts.append(generic)
        parts.append(rng.choice(FILLER_SUFFIX_BY_COUNTRY[country]))
        out.append((" ".join(parts), country))
    return out


def build_dataset(
    *,
    seed: int = 0,
    filler: int = 1500,
    kinds: list[str] | None = None,
    per_target: int = 3,
) -> Dataset:
    """Build records and cases.

    Each target gets `per_target` positive cases with distinct kinds (cycling through
    `kinds` so every kind gets roughly equal coverage), plus negatives sized to about a
    quarter of the positives.
    """
    rng = random.Random(seed)
    kinds = kinds or ALL_KINDS
    pos_kinds = [k for k in kinds if k in POSITIVE_KINDS]
    neg_kinds = [k for k in kinds if k in NEGATIVE_KINDS]

    records: list[Record] = []
    for i, (name, country) in enumerate(TARGETS):
        records.append(Record(id=f"t{i + 1}", name=name, extra={"country": country}))
    for i, (name, country) in enumerate(build_filler(filler, rng)):
        records.append(Record(id=f"f{i + 1}", name=name, extra={"country": country}))
    rng.shuffle(records)

    name_counts: dict[str, int] = {}
    for r in records:
        for t in distinctive_tokens(r.name):
            key = strip_accents(t).casefold()
            name_counts[key] = name_counts.get(key, 0) + 1

    def namesake(r: Record) -> bool:
        keys = [strip_accents(t).casefold() for t in distinctive_tokens(r.name)]
        return bool(keys) and any(name_counts.get(k, 0) > 1 for k in keys)

    cases: list[Case] = []
    targets = [r for r in records if r.id.startswith("t")]
    kind_cursor = 0
    for r in targets:
        chosen: list[str] = []
        while len(chosen) < min(per_target, len(pos_kinds)):
            k = pos_kinds[kind_cursor % len(pos_kinds)]
            kind_cursor += 1
            if k not in chosen:
                chosen.append(k)
        for k in chosen:
            query = perturb(r.name, k, rng)
            context = None
            # Namesakes (same distinctive word elsewhere in the list) always get context,
            # otherwise the expected answer would not be determinable.
            if k == "with_context" or namesake(r) or rng.random() < 0.25:
                context = {"country": r.extra["country"]}
            cases.append(
                Case(
                    id=f"{r.id}-{k}",
                    query=query,
                    kind=k,
                    expected_id=r.id,
                    context=context,
                    source_name=r.name,
                )
            )

    n_neg = max(1, len(cases) // 4) if neg_kinds else 0
    for i in range(n_neg):
        k = neg_kinds[i % len(neg_kinds)]
        if k == "absent":
            name = " ".join(
                [
                    rng.choice(ABSENT_DISTINCTIVE),
                    rng.choice(FILLER_DESCRIPTOR),
                    rng.choice([g for g in FILLER_GENERIC if g]),
                    rng.choice(["Inc", "Ltd", "GmbH", "SA"]),
                ]
            )
            cases.append(Case(id=f"neg{i + 1}-absent", query=name, kind=k, expected_id=None))
        elif k == "decoy":
            # A distinctive word that exists exactly once, attached to a clearly different
            # business in a different country. Correct answer: no match.
            src = rng.choice(targets)
            dist = distinctive_tokens(src.name)
            if not dist:
                continue
            word = dist[0]
            if name_counts.get(strip_accents(word).casefold(), 0) != 1:
                continue
            other_country = rng.choice(
                [c for c in FILLER_SUFFIX_BY_COUNTRY if c != src.extra["country"]]
            )
            descriptor = rng.choice(
                [d for d in FILLER_DESCRIPTOR if d.casefold() not in src.name.casefold()]
            )
            suffix = rng.choice(FILLER_SUFFIX_BY_COUNTRY[other_country])
            name = f"{word} {descriptor} {suffix}"
            cases.append(
                Case(
                    id=f"neg{i + 1}-decoy",
                    query=name,
                    kind=k,
                    expected_id=None,
                    context={"country": other_country},
                    source_name=src.name,
                )
            )
        else:  # generic_only
            words = rng.sample(["International", "Global", "Group", "Holdings", "Services"], 3)
            name = " ".join(words + [rng.choice(["Inc", "Ltd", "SA"])])
            cases.append(Case(id=f"neg{i + 1}-generic", query=name, kind=k, expected_id=None))

    return Dataset(records=records, cases=cases, seed=seed)

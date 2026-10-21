"""Compare a ScheMatiQ CSV export of NES proteins with the NESdb (LRNes) ground truth.

Each CSV row is matched to an NESdb protein by its full-name column (also tried
against NESdb's short and alternative names). Every other column is then mapped
to an NESdb field (from its header, from --map, or by picking the field whose
values agree with it most often) and each cell is scored as recovered or not.
Cells whose NESdb value is Unknown/Undetermined are not counted.

Writes, next to the CSV (or to --out-dir):
  <name>_vs_nesdb.csv          the table + per-row "Recovered" columns and a footer
                               row with how many values each column recovered
  <name>_vs_nesdb_per_nes.csv  per NES ID: how many NESdb fields any row recovered
  <name>_vs_nesdb_cells.csv    every compared cell, side by side with NESdb

NESdb pages are scraped once and cached in --cache.

Usage (from the repo root):
    .venv\\Scripts\\python.exe backend\\scripts\\compare_with_nesdb.py <export.csv>
        [--name-col "Full Name"] [--map "CSV column=field" ...] [--skip "CSV column" ...]
        [--llm-judge] [--include-uncited] [--include-doubt] [--refresh]

--llm-judge sends free-text cells the rules call a miss (evidence, secondary structure, other
residues, and mutations written in words) to Gemini (GEMINI_API_KEY from backend/.env) and saves each verdict, so re-runs are free.

Fields for --map: name, alt_names, organism, uniprot, evidence, mutations,
mutations_export, mutations_crm1, signals, secondary_structure, other_residues,
sequence, pmids.
"""

import argparse
import csv
import html
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "http://prodata.swmed.edu/LRNes/IndexFiles/"
PAGES = {"good": "namesGood.php", "doubt": "namesDoubt.php"}
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}
DEFAULT_CACHE = Path.home() / "Downloads" / "NES" / "nesdb_entries.json"

ID_COLUMNS = {"doc name", "row name", "observation unit", "unit name", "source document"}
NO_VALUE = {"", "unknown", "undetermined", "none", "n/a", "na", "-", "not reported", "not mentioned", "not specified", "null"}
HEAD_TO_FIELD = {
    "alternative names": "alt_names",
    "organism": "organism",
    "experimental evidence for crm1-mediated export": "evidence",
    "mutations that affect nuclear export": "mutations_export",
    "mutations that affect crm1 binding": "mutations_crm1",
    "functional export signals": "signals",
    "secondary structure of export signal": "secondary_structure",
    "other residues important for export": "other_residues",
}
# Header keyword rules, checked in order: the first rule whose keywords all appear wins.
HEADER_RULES = [
    (("uniprot",), "uniprot"), (("accession",), "uniprot"),
    (("organism",), "organism"), (("species",), "organism"),
    (("alternative",), "alt_names"), (("alias",), "alt_names"), (("synonym",), "alt_names"),
    (("gene",), "alt_names"), (("short", "name"), "alt_names"), (("abbrev",), "alt_names"),
    (("mutation", "crm1"), "mutations_crm1"), (("mutation", "binding"), "mutations_crm1"),
    (("mutation",), "mutations"), (("mutant",), "mutations"),
    (("secondary",), "secondary_structure"), (("helix",), "secondary_structure"),
    (("other", "residue"), "other_residues"), (("contact",), "other_residues"),
    (("full", "sequence"), "sequence"), (("protein", "sequence"), "sequence"),
    (("nes", "sequence"), "signals"), (("export", "signal"), "signals"), (("nes", "motif"), "signals"),
    (("nes", "position"), "signals"), (("evidence",), "evidence"),
    (("pmid",), "pmids"), (("pubmed",), "pmids"),
]
# Columns named exactly like an NESdb field (as ScheMatiQ exports a schema seeded from NESdb).
EXACT_HEADERS = {"alternative names": "alt_names", "sequence": "sequence", "uniprot": "uniprot",
                 **{" ".join(re.sub(r"[^a-z0-9]+", " ", h).split()): f for h, f in HEAD_TO_FIELD.items()}}
GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "κ": "kappa", "λ": "lambda", "μ": "mu", "σ": "sigma"}
STOPWORDS = {"the", "a", "an", "of", "and", "or", "in", "to", "with", "by", "for", "is", "was", "on", "at", "as", "protein"}
AA_RE = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]{6,}")
THREE_LETTER = {"ala": "A", "arg": "R", "asn": "N", "asp": "D", "cys": "C", "gln": "Q", "glu": "E", "gly": "G",
                "his": "H", "ile": "I", "leu": "L", "lys": "K", "met": "M", "phe": "F", "pro": "P", "ser": "S",
                "thr": "T", "trp": "W", "tyr": "Y", "val": "V"}
_AA3 = "|".join(THREE_LETTER)
# Leu308Ala, Leu-544 to Ala, L619-A, L300->R, L172A
MUTATION3_RE = re.compile(rf"\b({_AA3})[\s\-]?(\d{{1,4}})\s*(?:to|->|→|>|-)?\s*({_AA3})\b", re.I)
MUTATION_RE = re.compile(r"\b([A-Z])-?(\d{1,4})\s*(?:->|→|>|-)?\s*([A-Z])\b")
# L348/352A, L348,352A and 14,16L->A: one substitution applied at several positions
MUTATION_LIST_RE = re.compile(r"\b([A-Z])?(\d{1,4}(?:\s*[/,]\s*\d{1,4})+)\s*([A-Z])?\s*(?:->|→|>|to)?\s*([A-Z])\b")
ORGANISM_ADJECTIVES = {"canine": "dog", "murine": "mouse", "porcine": "pig", "equine": "horse",
                       "feline": "cat", "ovine": "sheep", "simian": "monkey"}
GENERIC_NAME_WORDS = {"kinase", "factor", "receptor", "domain", "subunit", "homolog", "transcription factor",
                      "protein kinase", "nes", "complex", "enzyme", "helicase", "phosphatase"}
GENERIC_ORGANISM_WORDS = {"virus", "cells", "cell", "strain", "mammal", "mammalian", "in vitro", "yeast cells"}
RANGE_RE = re.compile(r"(\d{1,4})\s*(?:-|–|—|to|\.\.)\s*(\d{1,4})")
UNIPROT_RE = re.compile(r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\b")


# ---------------------------------------------------------------- NESdb scraping

def parse_details(page_html: str) -> dict:
    soup = BeautifulSoup(page_html, "html.parser")
    entry: dict = {"pmids": sorted(set(re.findall(r"list_uids=(\d+)", page_html)))}
    for head in soup.select("div.head"):
        content = head.find_next_sibling("div", class_="content")
        if content is None:
            continue
        key = " ".join(head.get_text(" ", strip=True).split()).lower()
        if key == "full name":
            link = content.find("a", class_="swiss")
            entry["uniprot"] = link["href"].strip().rstrip("/").split("/")[-1] if link else ""
        if key == "sequence":
            fasta = content.find("pre", id="toggleFasta")
            entry["sequence"] = "".join(fasta.get_text().splitlines()[1:]).strip() if fasta else ""
            continue
        for ref in content.select("sup:has(a.ref)"):
            ref.decompose()
        for note in content.select("div.notes"):
            note.decompose()
        if key == "functional export signals":
            entry["signals"] = [
                {"start": int(s), "seq": seq, "end": int(e)}
                for s, seq, e in re.findall(r"<sup>(\d+)</sup>\s*([A-Za-z]+)\s*<sup>(\d+)</sup>", str(content))
            ]
        text = content.get_text(" ", strip=True).replace("\xa0", " ")
        text = re.sub(r"\s*UniProt\s*$", "", " ".join(text.split()))
        if key == "full name":
            entry["full_name"] = text
        elif key in HEAD_TO_FIELD:
            entry[HEAD_TO_FIELD[key] + ("_text" if key == "functional export signals" else "")] = text
    return entry


def scrape_nesdb(include_doubt: bool) -> list[dict]:
    entries = []
    for kind in ["good", "doubt"] if include_doubt else ["good"]:
        soup = BeautifulSoup(requests.get(urljoin(BASE_URL, PAGES[kind]), headers=HEADERS, timeout=60).text, "html.parser")
        rows = {}
        for tr in soup.select("tr"):
            link = tr.select_one("td:nth-of-type(1) a")
            if link and "details.php" in link.get("href", ""):
                rows.setdefault(urljoin(BASE_URL, link["href"]), html.unescape(link.get_text(strip=True)))
        print(f"NESdb {kind}: {len(rows)} entries")
        for n, (url, name) in enumerate(rows.items(), 1):
            for attempt in range(3):
                try:
                    page = requests.get(url, headers=HEADERS, timeout=60).text
                    break
                except requests.RequestException as e:
                    if attempt == 2:
                        print(f"  skipped {url}: {e}")
                        page = None
                    time.sleep(2)
            if page is not None:
                entry = parse_details(page)
                entry.update(nes_id=url.split("=")[-1], name=name, set=kind, url=url)
                entries.append(entry)
            if n % 50 == 0:
                print(f"  {n}/{len(rows)}")
            time.sleep(0.2)
    return entries


def load_nesdb(cache: Path, include_doubt: bool, refresh: bool) -> list[dict]:
    if cache.exists() and not refresh:
        entries = json.loads(cache.read_text(encoding="utf-8"))
        if include_doubt and not any(e["set"] == "doubt" for e in entries):
            print("Cache has no 'in doubt' entries - re-scraping (or drop --include-doubt)")
        else:
            return [e for e in entries if include_doubt or e["set"] == "good"]
    entries = scrape_nesdb(include_doubt)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(entries, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Cached {len(entries)} NESdb entries in {cache}")
    return entries


# ---------------------------------------------------------------- normalising

def norm(text) -> str:
    text = html.unescape(str(text or "")).lower()
    for g, word in GREEK.items():
        text = text.replace(g, f" {word} ")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def tokens(text) -> set[str]:
    return {t for t in norm(text).split() if t not in STOPWORDS}


def has_value(value) -> bool:
    n = norm(value)  # "unknown (NES not included in PDB)" records nothing to recover either
    return n not in NO_VALUE and not n.startswith("unknown ")


def entry_names(entry: dict) -> list[str]:
    names = [entry.get("full_name", ""), entry.get("name", "")]
    names += re.split(r"[,;]", entry.get("alt_names", ""))
    return [n for n in (norm(x) for x in names) if n and n not in NO_VALUE]


def name_score(csv_name: str, entry: dict) -> float:
    target = norm(csv_name)
    if not target:
        return 0.0
    best = 0.0
    for n in entry_names(entry):
        if n == target:
            return 1.0
        best = max(best, SequenceMatcher(None, target, n).ratio())
    return best


# ---------------------------------------------------------------- comparators
# Each returns (truth_text, result) where result is True/False, or None when NESdb has no value.

def text_match(csv_value: str, truth: str, threshold: float) -> bool:
    a, b = norm(csv_value), norm(truth)
    if not a:
        return False
    if (len(a) >= 3 and a in b) or (len(b) >= 3 and b in a):
        return True
    ta, tb = tokens(a), tokens(b)
    if tb and len(ta & tb) / len(tb) >= threshold:
        return True
    return SequenceMatcher(None, a, b).ratio() >= threshold


def cmp_text(field):
    def compare(value, entry, args):
        truth = entry.get(field, "")
        if not has_value(truth):
            return truth, None
        return truth, text_match(value, truth, args.text_threshold)
    return compare


def organism_text(text) -> str:
    v = norm(text)
    v = re.sub(r"\btype (\d)", r"\1", v)  # "herpes simplex virus type 1" = "... virus 1"
    return ORGANISM_ADJECTIVES.get(v, v)


def cmp_organism(value, entry, args):
    truth = entry.get("organism", "")
    if not has_value(truth):
        return truth, None
    parts = [p for p in (organism_text(x) for x in re.split(r"[()]", truth)) if p]
    v = organism_text(value)
    # The value may itself be "Name (abbrev)" or "Mouse / Human"; any substantial piece may carry the species.
    pieces = [v] + [p for p in (organism_text(x) for x in re.split(r"[()/,;]", value))
                    if len(p) >= 4 and p not in GENERIC_ORGANISM_WORDS]
    return truth, bool(v) and any(p in q or q in p for p in parts for q in pieces)


def compact_name(text) -> set[str]:
    """Spelling variants of a name with spaces dropped: Greek letters as words and as letters (IκBα = IkBa)."""
    raw = html.unescape(str(text or "")).lower()
    short = raw
    for g, word in GREEK.items():
        short = short.replace(g, word[0])
    return {re.sub(r"[^a-z0-9]+", "", norm(raw)), re.sub(r"[^a-z0-9]+", "", short)}


def cmp_names(value, entry, args):
    truth = "; ".join([entry.get("name", ""), entry.get("alt_names", "")]).strip("; ")
    raw_names = [entry.get("full_name", "")] + re.split(r"[,;/]", f"{entry.get('name', '')};{entry.get('alt_names', '')}")
    names = [n for n in raw_names if norm(n) and norm(n) not in NO_VALUE]
    variants = set().union(*(compact_name(n) for n in names)) if names else set()
    # NESdb runs several alternative names together ("Protein kinase B (PKB) Protein kinase B alpha ..."),
    # so a CSV name also counts when it appears as a whole phrase inside one of them - a single word only
    # when the rest of that name is generic ("APC" in "APC Protein"), so "ERK" does not match "ERK kinase 1".
    padded = [f" {norm(n)} " for n in names]
    generic = STOPWORDS | GENERIC_NAME_WORDS | {"protein", "transcription"}
    items = [x for x in re.split(r"[,;/|]", value) if norm(x)] + [value]
    for item in items:
        n = norm(item)
        if n in NO_VALUE or n in STOPWORDS or n in GENERIC_NAME_WORDS:
            continue
        if compact_name(item) & variants:
            return truth, True
        if " " in n and any(f" {n} " in p for p in padded):
            return truth, True
        if len(n) >= 3 and any(f" {n} " in p and set(p.replace(f" {n} ", " ", 1).split()) <= generic for p in padded):
            return truth, True
        # "Mek1 kinase" vs "MEK1": a one-word NESdb name inside the CSV name.
        if any(len(m) >= 3 and " " not in m and f" {m} " in f" {n} " for m in (norm(x) for x in names)):
            return truth, True
    return truth, False


def mutation_set(text: str) -> set[tuple[str, str]]:
    """(position, new residue) of each substitution in any common notation. The original residue is left out
    so that a residue NESdb or the paper names differently at the same position still counts."""
    text = text or ""
    found = {(pos, THREE_LETTER[new.lower()]) for _, pos, new in MUTATION3_RE.findall(text)}
    found |= {(pos, new) for _, pos, new in MUTATION_RE.findall(text)}
    for _, positions, _, new in MUTATION_LIST_RE.findall(text):
        found |= {(pos, new) for pos in re.findall(r"\d+", positions)}
    return found


def residues(value: str) -> str:
    """The value as one run of residues when it is nothing but a peptide - spaced ("L V V L"), numbered
    ("L165AGKM169SNLVL174"), starred or in three-letter code ("Cys-Gly-Gln") - else ''."""
    words = re.findall(r"[A-Za-z]+", value)
    if words and re.fullmatch(rf"(?:(?:{_AA3})[\s\-]*)+", value.strip(), re.I):
        return "".join(THREE_LETTER[w.lower()] for w in re.findall(_AA3, value, re.I))
    compact = re.sub(r"[\s\-\*\d]+", "", value)
    return compact if len(compact) >= 5 and re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", compact) else ""


def cmp_mutations(fields):
    def compare(value, entry, args):
        truth = ", ".join(dict.fromkeys(entry.get(f, "") for f in fields if has_value(entry.get(f, ""))))
        expected = mutation_set(truth)
        if not expected:
            return truth, None
        return truth, bool(expected & mutation_set(value))
    return compare


def longest_common(a: str, b: str) -> int:
    return SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b)).size


def cmp_signals(value, entry, args):
    signals = entry.get("signals") or []
    truth = ", ".join(f"{s['start']}-{s['end']} {s['seq']}" for s in signals) or entry.get("signals_text", "")
    if not signals:
        return truth, None
    whole = residues(value)
    seqs = [whole] if whole else AA_RE.findall(value.upper())
    for s in signals:
        for seq in seqs:
            short = min(len(seq), len(s["seq"]))
            if longest_common(seq, s["seq"].upper()) >= min(8, short):
                return truth, True
        for start, end in RANGE_RE.findall(value):
            if int(start) <= s["end"] and int(end) >= s["start"]:
                return truth, True
    return truth, False


def cmp_sequence(value, entry, args):
    truth = entry.get("sequence", "")
    if not truth:
        return "", None
    seq = residues(value) or "".join(AA_RE.findall(value.upper()))
    if seq and len(seq) < 50:  # an NES, not the whole protein
        return cmp_signals(value, entry, args)
    return truth[:30] + "...", bool(seq) and (seq in truth or truth in seq)


def cmp_uniprot(value, entry, args):
    truth = entry.get("uniprot", "").strip()  # older caches kept the link's trailing newline
    if not truth:
        return truth, None
    return truth, truth.split("-")[0] in {u.split("-")[0] for u in UNIPROT_RE.findall(value.upper())}


def cmp_pmids(value, entry, args):
    truth = set(entry.get("pmids") or [])
    if not truth:
        return "", None
    return ", ".join(sorted(truth)), bool(truth & set(re.findall(r"\b\d{6,9}\b", value)))


COMPARATORS = {
    "name": cmp_names,
    "alt_names": cmp_names,
    "organism": cmp_organism,
    "uniprot": cmp_uniprot,
    "evidence": cmp_text("evidence"),
    "mutations": cmp_mutations(["mutations_export", "mutations_crm1"]),
    "mutations_export": cmp_mutations(["mutations_export"]),
    "mutations_crm1": cmp_mutations(["mutations_crm1"]),
    "signals": cmp_signals,
    "secondary_structure": cmp_text("secondary_structure"),
    "other_residues": cmp_text("other_residues"),
    "sequence": cmp_sequence,
    "pmids": cmp_pmids,
}


# ---------------------------------------------------------------- LLM judge

LLM_FIELDS = ("evidence", "secondary_structure", "other_residues", "mutations", "mutations_export", "mutations_crm1")
# Field-specific rules appended to the prompt.
LLM_HINTS = {
    "evidence": " A bare \"yes\", a list of numbers, or a value that only says the protein or signal is exported, "
                "without naming the kind of evidence NESdb records (e.g. LMB sensitivity, CRM1 binding, crm1/Xpo1 "
                "mutant, competition with other CRM1 cargos), is too vague.",
    "mutations": " It is a match only if the extracted value describes at least one of the same substitutions (same "
                 "residue position and the same replacement residue), in any notation or in words "
                 "(\"leucines 237 and 240 to alanine\" = L237A/L240A). Mutating \"the hydrophobic residues\" without "
                 "positions, a deletion, or other positions is not a match.",
}
LLM_HINTS["mutations_export"] = LLM_HINTS["mutations_crm1"] = LLM_HINTS["mutations"]
LLM_MODEL = "gemini-3.5-flash-lite"  # the model the backend uses for its own verification calls
LLM_PROMPT = """You are checking whether an automatic extraction from a scientific paper recovered the value that \
the curated NESdb database (nuclear export signals) records for the same protein.

Protein: {protein}
Field: {field}
NESdb value: {truth}
Extracted value: {value}

Answer "match": true if the extracted value states the same fact(s) as the NESdb value, or its main fact when the \
NESdb value lists several; wording, synonyms and abbreviations don't matter (e.g. "LMB sensitive" = "export blocked \
by leptomycin B", "alpha-helix" = "helical"). Extra information in the extracted value is fine. Answer false if it \
contradicts NESdb, is about something else, or is too vague to show the NESdb fact. Give a one-sentence reason."""
LLM_SCHEMA = {"type": "OBJECT", "properties": {"match": {"type": "BOOLEAN"}, "reason": {"type": "STRING"}},
              "required": ["match", "reason"]}


def llm_judge_cells(pending, rows, row_results, mapping, args) -> dict:
    """Return {(row index, column): (match, reason)} for each pending cell, using saved verdicts when present."""
    from concurrent.futures import ThreadPoolExecutor
    import hashlib
    import os

    from dotenv import load_dotenv
    from google import genai
    from google.genai import types

    saved = json.loads(args.llm_cache.read_text(encoding="utf-8")) if args.llm_cache.exists() else {}
    jobs = {}
    for i, col in pending:
        entry = row_results[i][2]
        prompt = LLM_PROMPT.format(protein=row_results[i][1].get("full_name") or row_results[i][1].get("name"),
                                   field=mapping[col].replace("_", " "), truth=entry[col][0], value=rows[i][col])
        prompt += LLM_HINTS.get(mapping[col], "")
        jobs[(i, col)] = (hashlib.sha1(f"{args.llm_model}\n{prompt}".encode()).hexdigest(), prompt)
    todo = {key: prompt for key, prompt in jobs.values() if key not in saved}
    print(f"\nLLM judge: {len(pending)} free-text cells the rules missed, {len(todo)} new calls to {args.llm_model}")

    if todo:
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        if not os.environ.get("GEMINI_API_KEY"):
            sys.exit("GEMINI_API_KEY not found in backend/.env")
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        config = types.GenerateContentConfig(temperature=0, response_mime_type="application/json", response_schema=LLM_SCHEMA)

        def ask(item):
            key, prompt = item
            for attempt in range(4):
                try:
                    reply = json.loads(client.models.generate_content(model=args.llm_model, contents=prompt, config=config).text)
                    return key, [bool(reply["match"]), reply.get("reason", "")]
                except Exception as e:  # rate limits and malformed replies: back off and retry
                    if attempt == 3:
                        print(f"  LLM call failed, cell left as missed: {e}")
                        return key, None
                    time.sleep(2 ** attempt * 2)

        with ThreadPoolExecutor(max_workers=8) as pool:
            for n, (key, verdict) in enumerate(pool.map(ask, todo.items()), 1):
                if verdict:
                    saved[key] = verdict
                if n % 25 == 0:
                    print(f"  {n}/{len(todo)}")
        args.llm_cache.parent.mkdir(parents=True, exist_ok=True)
        args.llm_cache.write_text(json.dumps(saved, indent=1, ensure_ascii=False), encoding="utf-8")

    return {cell: tuple(saved[key]) for cell, (key, _) in jobs.items() if key in saved}


# ---------------------------------------------------------------- matching

def find_entries(csv_name: str, by_name: dict, entries: list[dict], threshold: float) -> tuple[list[dict], float]:
    """Return every NESdb entry the name matches (one protein can have several NES IDs) and the score."""
    exact = by_name.get(norm(csv_name))
    if exact:
        return exact, 1.0
    scored = [(name_score(csv_name, e), e) for e in entries]
    best = max((s for s, _ in scored), default=0.0)
    if best < threshold:
        return [], best
    return [e for s, e in scored if s == best], best


def same_name(a: str, b: str) -> bool:
    """True if two protein names are equal ignoring spacing, or one appears as whole words in the other."""
    a, b = norm(a), norm(b)
    if not a or not b:
        return False
    return a.replace(" ", "") == b.replace(" ", "") or f" {a} " in f" {b} " or f" {b} " in f" {a} "


def match_row(row: dict, cols: dict, by_pmid: dict, by_name: dict, entries: list[dict], args) -> tuple[list[dict], float, str]:
    """Match a row to NESdb entries. Returns (entries, score, how).

    'cited': NESdb cites this row's paper (from '<name>_<pmid>_full' in Source Document) for the
    protein the row names - the paper is ground truth for it. 'name only': the protein is in NESdb
    but NESdb does not cite this paper for it (e.g. a control NES mentioned in passing).
    """
    names = [row.get(cols["name"], "")]
    if cols.get("alt"):
        names += re.split(r"[,;]", row[cols["alt"]])
    if cols.get("unit"):
        names.append(re.sub(r"\bnes\b.*$", "", row[cols["unit"]], flags=re.I))
    names = [n.strip() for n in names if has_value(n)]

    m = re.match(r"(.*)_(\d{5,9})_full", row.get(cols.get("source") or "", ""))
    if m:
        prefix, pmid = m.groups()
        scored = []
        for entry in by_pmid.get(pmid, []):
            prefix_hit = same_name(prefix.replace("_", " "), entry.get("name", "")) or \
                any(same_name(prefix, n) for n in entry_names(entry))
            for n in names:
                score = name_score(n, entry)
                if score >= args.name_threshold - 0.1 or any(same_name(n, en) for en in entry_names(entry)) \
                        or (prefix_hit and same_name(n, prefix)):
                    scored.append((max(score, 0.9), entry))
                    break
        if scored:
            best = max(s for s, _ in scored)
            return [e for s, e in scored if s == best], best, "cited"

    cands, score = find_entries(names[0] if names else "", by_name, entries, args.name_threshold)
    for n in names[1:]:
        if not cands and by_name.get(norm(n)):
            cands, score = by_name[norm(n)], 1.0
    if not cands and names:
        cands = [e for e in entries if any(same_name(n, en) for n in names for en in entry_names(e)[:2])]
        score = 0.9 if cands else score
    return cands, score, "name only" if cands else ""


def guess_field(header: str) -> str | None:
    h = norm(header)
    if h in EXACT_HEADERS:
        return EXACT_HEADERS[h]
    words = h.split()
    for keywords, field in HEADER_RULES:
        if all(any(w.startswith(k) for w in words) for k in keywords):
            return field
    return None


def best_field(column: str, rows: list[dict], matches: list[list[dict]], args) -> tuple[str | None, float]:
    """Pick the NESdb field whose values agree with this column most often."""
    best, best_rate = None, 0.0
    for field, compare in COMPARATORS.items():
        hits = total = 0
        for row, cands in zip(rows, matches):
            if not cands or not has_value(row[column]):
                continue
            results = [compare(row[column], e, args)[1] for e in cands]
            if all(r is None for r in results):
                continue
            total += 1
            hits += any(results)
        rate = hits / total if total else 0.0
        if total >= 3 and rate > best_rate:
            best, best_rate = field, rate
    return (best, best_rate) if best_rate >= args.auto_threshold else (None, best_rate)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="CSV exported from ScheMatiQ")
    parser.add_argument("--name-col", help="Column holding the protein full name (default: auto-detect 'Full Name')")
    parser.add_argument("--map", action="append", default=[], metavar="COLUMN=FIELD", help="Force a column to an NESdb field")
    parser.add_argument("--skip", action="append", default=[], metavar="COLUMN", help="Column not to compare")
    parser.add_argument("--include-doubt", action="store_true", help="Also use NESdb's 'NESs in doubt' entries")
    parser.add_argument("--include-uncited", action="store_true", help="Also score rows whose paper NESdb does not cite for that NES")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="NESdb cache file (default: %(default)s)")
    parser.add_argument("--refresh", action="store_true", help="Re-scrape NESdb instead of using the cache")
    parser.add_argument("--out-dir", type=Path, help="Output folder (default: next to the CSV)")
    parser.add_argument("--name-threshold", type=float, default=0.85, help="Min name similarity to accept a protein match (default: %(default)s)")
    parser.add_argument("--text-threshold", type=float, default=0.6, help="Min similarity for free-text fields (default: %(default)s)")
    parser.add_argument("--llm-judge", action="store_true", help="Ask Gemini about free-text cells the rules call a miss")
    parser.add_argument("--llm-fields", nargs="+", default=list(LLM_FIELDS), choices=list(COMPARATORS),
                        help="NESdb fields the LLM judges (default: %(default)s)")
    parser.add_argument("--llm-model", default=LLM_MODEL, help="Gemini model (default: %(default)s)")
    parser.add_argument("--llm-cache", type=Path, default=DEFAULT_CACHE.with_name("nesdb_llm_judgments.json"),
                        help="Saved verdicts, so re-runs don't repeat calls (default: %(default)s)")
    parser.add_argument("--auto-map", action="store_true", help="Map unrecognised columns to the NESdb field they agree with most")
    parser.add_argument("--auto-threshold", type=float, default=0.3, help="Min agreement rate for --auto-map (default: %(default)s)")
    args = parser.parse_args()

    with args.csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        rows = [{c: (r.get(c) or "").strip() for c in columns} for r in reader]
    if not rows:
        sys.exit("The CSV has no rows")

    name_col = args.name_col or next((c for c in columns if norm(c) == "full name"), None) \
        or next((c for c in columns if "full name" in norm(c)), None) \
        or next((c for c in columns if "protein" in norm(c) and "name" in norm(c)), None)
    if name_col not in columns:
        sys.exit(f"No full-name column found; pass --name-col. Columns: {columns}")

    entries = load_nesdb(args.cache, args.include_doubt, args.refresh)
    by_name, by_pmid = defaultdict(list), defaultdict(list)
    for e in entries:
        for n in entry_names(e)[:2]:  # full name and short name are unique enough for an exact hit
            by_name[n].append(e)
        for pmid in e.get("pmids") or []:
            by_pmid[pmid].append(e)

    def find_col(*names):
        return next((c for c in columns if norm(c) in names), None)
    cols = {"name": name_col, "alt": find_col("alternative names"), "unit": find_col("observation unit"),
            "source": find_col("source document")}
    found = [match_row(r, cols, by_pmid, by_name, entries, args) for r in rows]
    # Rows matched by name only are listed but not scored: NESdb's values come from other papers.
    matches = [cands if how == "cited" or args.include_uncited else [] for cands, _, how in found]
    hows = [how for _, _, how in found]
    print(f"\n{len(rows)} rows: {hows.count('cited')} matched to an NES that NESdb cites the row's paper for, "
          f"{hows.count('name only')} to an NES NESdb cites other papers for"
          f"{' (scored too)' if args.include_uncited else ' (not scored; --include-uncited to score)'}, "
          f"{hows.count('')} not in NESdb")

    # Decide which NESdb field each column is compared with.
    forced = {}
    for item in args.map:
        col, _, field = item.partition("=")
        if col.strip() not in columns or field.strip() not in COMPARATORS:
            sys.exit(f"Bad --map '{item}'. Columns: {columns}; fields: {list(COMPARATORS)}")
        forced[col.strip()] = field.strip()
    mapping = {}
    print("\nColumn -> NESdb field")
    for col in columns:
        if col == name_col or col in args.skip or norm(col) in ID_COLUMNS:
            continue
        if col in forced:
            mapping[col], how = forced[col], "--map"
        elif guess_field(col):
            mapping[col], how = guess_field(col), "header"
        elif not args.auto_map:
            print(f"  {col!r}: not compared (no NESdb field by that name - use --map or --auto-map)")
            continue
        else:
            field, rate = best_field(col, rows, matches, args)
            if not field:
                print(f"  {col!r}: not compared (no field agrees with it; best rate {rate:.0%} - use --map)")
                continue
            mapping[col], how = field, f"auto, {rate:.0%} agreement"
        print(f"  {col!r} -> {mapping[col]} ({how})")

    # Score every cell. A row matching several NES IDs is scored against the one it recovers most of.
    cells, row_results = [], []
    for i, (row, cands) in enumerate(zip(rows, matches)):
        best = None
        for entry in cands:
            res = {col: COMPARATORS[field](row[col], entry, args) for col, field in mapping.items()}
            score = (sum(r is True for _, r in res.values()), -sum(r is False for _, r in res.values()))
            if best is None or score > best[0]:
                best = (score, entry, res)
        row_results.append(best)

    # Free-text cells the rules missed go to the LLM judge.
    judged = {}
    if args.llm_judge:
        pending = [(i, col) for i, best in enumerate(row_results) if best
                   for col, (_, r) in best[2].items()
                   if r is False and mapping[col] in args.llm_fields and has_value(rows[i][col])]
        judged = llm_judge_cells(pending, rows, row_results, mapping, args)
        for (i, col), (match, reason) in judged.items():
            truth, _ = row_results[i][2][col]
            row_results[i][2][col] = (truth, match)

    for i, (row, best) in enumerate(zip(rows, row_results)):
        if best:
            _, entry, res = best
            for col, (truth, result) in res.items():
                cells.append({
                    "row": i + 1, "protein": row[name_col], "nes_id": entry["nes_id"], "column": col,
                    "nesdb_field": mapping[col], "csv_value": row[col], "nesdb_value": truth,
                    "result": {True: "recovered", False: "missed", None: "no NESdb value"}[result],
                    "judged_by": "llm" if (i, col) in judged else "rules",
                    "llm_reason": judged.get((i, col), (None, ""))[1],
                })

    out_dir = args.out_dir or args.csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.csv.stem + "_vs_nesdb"
    extra = ["NESdb ID", "NESdb Full Name", "Match", "Name Match", "Recovered", "Recovered %"]

    # Main table: original columns, per-row recovery, footer with per-column recovery.
    col_hits = defaultdict(lambda: [0, 0])
    with (out_dir / f"{stem}.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns + extra)
        for row, best, (cands, score, how) in zip(rows, row_results, found):
            if not best:
                ids = ", ".join(e["nes_id"] for e in cands)
                names = ", ".join(dict.fromkeys(e.get("full_name") or e.get("name", "") for e in cands)) or "not in NESdb"
                note = "name only - paper not cited for it, not scored" if how else "not in NESdb"
                w.writerow([row[c] for c in columns] + [ids, names, note, f"{score:.2f}", "", ""])
                continue
            _, entry, res = best
            judged = [r for _, r in res.values() if r is not None]
            for col, (_, r) in res.items():
                if r is not None:
                    col_hits[col][0] += r
                    col_hits[col][1] += 1
            pct = f"{sum(judged) / len(judged):.0%}" if judged else ""
            w.writerow([row[c] for c in columns] + [
                entry["nes_id"], entry.get("full_name", ""), how, f"{score:.2f}", f"{sum(judged)}/{len(judged)}", pct])
        all_hits = sum(h for h, _ in col_hits.values())
        all_total = sum(t for _, t in col_hits.values())
        field_row = [f"NESdb field: {mapping[c]}" if c in mapping else ("(name key)" if c == name_col else "") for c in columns]
        recovered_row = [f"{col_hits[c][0]}/{col_hits[c][1]} ({col_hits[c][0] / col_hits[c][1]:.0%})" if col_hits[c][1]
                         else ("0/0" if c in mapping else "") for c in columns]
        if columns[0] not in mapping and columns[0] != name_col:  # label the footer rows in the first (ID) column
            field_row[0], recovered_row[0] = "NESdb FIELD", "RECOVERED"
        total = f"{all_hits}/{all_total} ({all_hits / all_total:.0%})" if all_total else ""
        w.writerow([])
        w.writerow(field_row)
        w.writerow(recovered_row + ["", "", "", "", total, ""])

    # Per NES: a field counts as recovered if any row of that NES recovered it.
    per_nes = defaultdict(lambda: {"rows": 0, "fields": defaultdict(lambda: None), "entry": None})
    for c in cells:
        nes = per_nes[c["nes_id"]]
        r = {"recovered": True, "missed": False}.get(c["result"])
        if r is not None:
            nes["fields"][c["nesdb_field"]] = bool(nes["fields"][c["nesdb_field"]]) or r
    for best in row_results:
        if best:
            per_nes[best[1]["nes_id"]]["rows"] += 1
            per_nes[best[1]["nes_id"]]["entry"] = best[1]
    with (out_dir / f"{stem}_per_nes.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["NES ID", "Name", "Full Name", "CSV Rows", "Recovered", "Recovered %", "Recovered Fields", "Missed Fields"])
        nes_rates, nes_got, nes_total = [], 0, 0
        for nes_id, nes in sorted(per_nes.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0):
            got = sorted(k for k, v in nes["fields"].items() if v)
            missed = sorted(k for k, v in nes["fields"].items() if v is False)
            total = len(got) + len(missed)
            if total:
                nes_rates.append(len(got) / total)
                nes_got, nes_total = nes_got + len(got), nes_total + total
            w.writerow([nes_id, nes["entry"].get("name", ""), nes["entry"].get("full_name", ""), nes["rows"],
                        f"{len(got)}/{total}", f"{len(got) / total:.0%}" if total else "", "; ".join(got), "; ".join(missed)])
        if nes_rates:
            # Average: every NES weighs the same. Pooled: every NESdb field weighs the same.
            mean, median = sum(nes_rates) / len(nes_rates), statistics.median(nes_rates)
            w.writerow([])
            w.writerow(["AVERAGE per NES", "", f"over {len(nes_rates)} NES", "", "", f"{mean:.0%}", "", ""])
            w.writerow(["MEDIAN per NES", "", "", "", "", f"{median:.0%}", "", ""])
            w.writerow(["POOLED (all fields)", "", "", "", f"{nes_got}/{nes_total}", f"{nes_got / nes_total:.0%}", "", ""])

    with (out_dir / f"{stem}_cells.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["row", "protein", "nes_id", "column", "nesdb_field", "csv_value", "nesdb_value",
                                          "result", "judged_by", "llm_reason"])
        w.writeheader()
        w.writerows(cells)

    print("\nRecovered per column")
    for col in mapping:
        hits, total = col_hits[col]
        print(f"  {col}: {hits}/{total}" + (f" ({hits / total:.0%})" if total else ""))
    if nes_rates:
        print(f"\nPer NES ({len(nes_rates)} NES): average {mean:.0%} of fields recovered, median {median:.0%}, "
              f"pooled {nes_got}/{nes_total} ({nes_got / nes_total:.0%})")
    print(f"\n{len(per_nes)} NES IDs covered. Wrote {stem}.csv, {stem}_per_nes.csv, {stem}_cells.csv to {out_dir}")


if __name__ == "__main__":
    main()

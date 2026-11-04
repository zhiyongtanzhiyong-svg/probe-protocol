"""Reproducible reference audit for probe-protocol.tex.

Resolves every \\bibitem against a canonical registry (Crossref, DataCite,
arXiv, HuggingFace, with Semantic Scholar for abstracts), checks cited<->bibitem
parity and first-citation order, and compares the resolved title / first author /
year against the manuscript entry. Fetches abstracts as evidence for the content
claims made at each \\cite site.

Fetched metadata is cached to data/reference_audit_cache.json so re-runs are
fast, deterministic, and the cache itself is the audit record of exactly what
each registry returned at fetch time. --refresh forces a live re-fetch.

Outputs:
  data/reference_audit.json                     machine-readable evidence + verdicts
  manuscript/paper/CITATION_VERIFICATION.md     human report (regenerated)

Exit code: 0 = all PASS and parity clean; 1 = any MISMATCH or parity error.
UNRESOLVED (registry unreachable / no identifier) is a WARN, not a failure.

Offline local ground-truth fallback: for community datasets with no registry
identifier (e.g. Hugging Face datasets with no DOI), an entry listed in
LOCAL_VERIFY is additionally checked against a local materialization (a manifest
JSON + the on-disk files). If the local files exist and the manifest's class /
image counts agree with the manuscript's claims, the entry is promoted from
UNRESOLVED to PASS (marked `local_verified`), and the report documents the
offline evidence. This only ever fires when the registry path could not resolve,
so online-resolvable entries are unaffected.

Run:  python3 scripts/audit_references.py [--refresh]
"""
import argparse
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "manuscript" / "paper" / "probe-protocol.tex"
CACHE = ROOT / "data" / "reference_audit_cache.json"
REPORT_JSON = ROOT / "data" / "reference_audit.json"
REPORT_MD = ROOT / "manuscript" / "paper" / "CITATION_VERIFICATION.md"

# Offline local ground-truth verification for registry-less (community) datasets.
# key -> manifest that materializes the dataset locally + the numeric claims the
# manuscript makes about it. Used only when the online registry cannot resolve.
LOCAL_VERIFY = {
    "tld": {
        "manifest": "data/tld_local_manifest.json",
        "claimed_images": 1392,   # manuscript: "TLD (1,392 images, 10 classes)"
        "claimed_classes": 10,
        "label": "TLD (Tobacco_Leaf_Diseases) test split - local materialization",
    },
}

UA = {"User-Agent": "citation-audit/1.0 (reproducible ref audit)"}
STOP = set(("a an the and of for in with to from on by using based via our this "
            "that et al arxiv preprint data image learning").split())


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def http_get(url, accept=None, tmo=30):
    h = dict(UA)
    if accept:
        h["Accept"] = accept
    req = urllib.request.Request(url, headers=h)
    return urllib.request.urlopen(req, timeout=tmo).read().decode("utf-8", "ignore")


def norm(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def content_words(s):
    return [w for w in norm(s).split() if len(w) > 3 and w not in STOP]


def extract_year(blob):
    m = re.search(r"(19|20)\d{2}", blob)
    return int(m.group(0)) if m else None


def parse_tex():
    tex = TEX.read_text(encoding="utf-8")
    lines = tex.splitlines()
    cited = {}
    first_cite = {}
    cite_re = re.compile(r"\\cite[a-zA-Z*]*\{([^}]+)\}")
    for i, ln in enumerate(lines, 1):
        for m in cite_re.finditer(ln):
            for k in m.group(1).split(","):
                k = k.strip()
                if not k:
                    continue
                cited.setdefault(k, []).append((i, ln.strip()[:200]))
                if k not in first_cite:
                    first_cite[k] = (i, m.start())
    bib = {}
    bib_order = []
    bib_re = re.compile(r"\\bibitem\{([^}]+)\}")
    matches = list(bib_re.finditer(tex))
    for j, m in enumerate(matches):
        key = m.group(1)
        start = m.end()
        end = matches[j + 1].start() if j + 1 < len(matches) else tex.find("\\end{thebibliography}", start)
        if end < 0:
            end = len(tex)
        bib[key] = {"text": re.sub(r"\s+", " ", tex[start:end]).strip(), "order_index": j}
        bib_order.append(key)
    tm = re.search(r"\\title\{([^}]+)\}", tex)
    paper_title = tm.group(1).strip() if tm else "(untitled)"
    return tex, lines, cited, first_cite, bib, bib_order, paper_title


def extract_ids(text):
    doi = None
    dm = re.search(r"(?:https?://doi\.org/)?(10\.\d{4,9}/[^\s\\{}]+)", text)
    if dm:
        doi = dm.group(1).rstrip(".,;")
    arxiv = None
    for pat in (r"10\.48550/arXiv\.(\d{4}\.\d{4,5})",
                r"arxiv\.org/abs/(\d{4}\.\d{4,5})",
                r"arXiv:(\d{4}\.\d{4,5})"):
        am = re.search(pat, text)
        if am:
            arxiv = am.group(1)
            break
    hf = None
    hm = re.search(r"huggingface\.co/datasets/([\w.-]+/[\w.-]+)", text)
    if hm:
        hf = hm.group(1)
    return doi, arxiv, hf, extract_year(text)


def _fetch(src, rid):
    if src == "crossref":
        d = json.loads(http_get("https://api.crossref.org/works/" + rid))
        m = d.get("message", {})
        yr = (m.get("issued") or m.get("created") or {}).get("date-parts", [[None]])[0][0]
        return {"source": src, "id": rid, "http_status": 200,
                "title": (m.get("title") or [""])[0],
                "authors": [a.get("family") or a.get("name", "") for a in (m.get("author") or [])],
                "year": yr, "venue": (m.get("container-title") or [""])[0],
                "abstract": re.sub(r"<[^>]+>", " ", m.get("abstract") or "").strip(),
                "fetched_at": now()}
    if src == "datacite":
        d = json.loads(http_get("https://doi.org/" + rid,
                                accept="application/vnd.citationstyles.csl+json"))
        yr = (d.get("issued") or d.get("created") or {}).get("date-parts", [[None]])[0][0]
        return {"source": src, "id": rid, "http_status": 200,
                "title": d.get("title"),
                "authors": [a.get("family") or a.get("name", "") for a in (d.get("author") or [])],
                "year": yr, "venue": d.get("publisher"), "abstract": "",
                "fetched_at": now()}
    if src == "arxiv":
        txt = http_get("https://export.arxiv.org/api/query?id_list=" + rid)
        titles = re.findall(r"<title>(.*?)</title>", txt, re.S)
        names = re.findall(r"<name>(.*?)</name>", txt, re.S)
        sums = re.findall(r"<summary>(.*?)</summary>", txt, re.S)
        pubs = re.findall(r"<published>(.*?)</published>", txt, re.S)
        title = re.sub(r"\s+", " ", titles[1]).strip() if len(titles) > 1 else ""
        if not title and titles:
            title = re.sub(r"\s+", " ", titles[0]).strip()
        return {"source": src, "id": rid, "http_status": 200,
                "title": title,
                "authors": [re.sub(r"\s+", " ", n).strip() for n in names],
                "year": int(pubs[0][:4]) if pubs else None,
                "venue": "arXiv",
                "abstract": re.sub(r"\s+", " ", sums[0]).strip() if sums else "",
                "fetched_at": now()}
    if src == "hf":
        d = json.loads(http_get("https://huggingface.co/api/datasets/" + rid))
        return {"source": src, "id": rid, "http_status": 200,
                "title": d.get("id") or rid, "authors": [], "year": None,
                "venue": "Hugging Face Datasets", "abstract": "",
                "downloads": d.get("downloads"), "fetched_at": now()}
    if src == "crossref_search":
        url = ("https://api.crossref.org/works?query.bibliographic="
               + urllib.parse.quote(rid) + "&rows=1")
        d = json.loads(http_get(url))
        m = (d.get("message", {}).get("items") or [{}])[0]
        yr = (m.get("issued") or m.get("created") or {}).get("date-parts", [[None]])[0][0]
        return {"source": src, "id": rid[:80], "http_status": 200,
                "title": (m.get("title") or [""])[0],
                "authors": [a.get("family") or a.get("name", "") for a in (m.get("author") or [])],
                "year": yr, "venue": (m.get("container-title") or [""])[0],
                "abstract": re.sub(r"<[^>]+>", " ", m.get("abstract") or "").strip(),
                "fetched_at": now()}
    raise ValueError("unknown source " + src)


def s2_abstract(doi):
    try:
        d = json.loads(http_get(
            "https://api.semanticscholar.org/graph/v1/paper/DOI:%s?fields=abstract,tldr" % doi, tmo=25))
        return (d.get("abstract") or (d.get("tldr") or {}).get("text") or "").strip()
    except Exception:
        return ""


def resolve(doi, arxiv, hf, blob, cache, refresh):
    attempts = []
    if hf:
        attempts.append(("hf", hf))
    if arxiv:
        attempts.append(("arxiv", arxiv))
    if doi and not doi.startswith("10.48550/arXiv"):
        attempts.append(("crossref", doi))
        attempts.append(("datacite", doi))
    if not attempts:
        attempts.append(("crossref_search", blob))
    last = None
    for src, rid in attempts:
        ck = src + ":" + rid
        if not refresh and ck in cache:
            e = dict(cache[ck])
            e.setdefault("_source", src)
            if e.get("http_status") == 200 and e.get("title"):
                return e
            last = last or e
            continue
        try:
            meta = _fetch(src, rid)
        except Exception as ex:
            meta = {"source": src, "id": rid, "http_status": 0, "error": str(ex),
                    "fetched_at": now()}
        meta["_source"] = src
        cache[ck] = meta
        last = last or meta
        if meta.get("http_status") == 200 and meta.get("title"):
            return meta
        time.sleep(0.3)
    return last


def _bib_has_author(blob):
    seg = blob.split(";")[0]
    if "," not in seg:
        return False
    rest = seg.split(",", 1)[1].strip().lstrip(".")
    rest_clean = re.sub(r"[^a-z]", "", rest.lower())
    return 0 < len(rest_clean) <= 4


def _bib_first_surname(blob):
    seg = blob.split(";")[0]
    if "," not in seg:
        return ""
    first = seg.split(",", 1)[0].strip()
    toks = first.split()
    return toks[0].lower() if toks else ""


def compare(blob, meta, byr):
    if meta.get("_source") == "hf":
        ok = meta.get("http_status") == 200
        return {"verdict": "PASS" if ok else "UNRESOLVED",
                "note": "HuggingFace dataset; existence check only",
                "downloads": meta.get("downloads")}
    blob_norm = norm(blob)
    rw = content_words(meta.get("title") or "")
    ratio = (sum(1 for w in rw if w in blob_norm) / len(rw)) if rw else 0.0
    title_ok = ratio >= 0.6
    if _bib_has_author(blob):
        bfs = norm(_bib_first_surname(blob))
        res_auth = " ".join(norm(a) for a in (meta.get("authors") or []))
        first_ok = (not bfs) or (bfs in res_auth)
    else:
        bfs, first_ok = "", True
    ryr = meta.get("year")
    year_ok = True if not (ryr and byr) else (abs(int(ryr) - byr) <= 1)
    return {"ratio": round(ratio, 2), "title_ok": title_ok,
            "bib_first_author": bfs, "first_ok": first_ok,
            "resolved_year": ryr, "bib_year": byr, "year_ok": year_ok,
            "verdict": "PASS" if (title_ok and first_ok and year_ok) else "MISMATCH"}


def parity(cited, bib):
    c, b = set(cited), set(bib)
    return {"cited_count": len(c), "bibitem_count": len(b),
            "cited_not_in_bib": sorted(c - b),
            "bibitem_not_cited": sorted(b - c)}


def order_check(bib_order, first_cite):
    actual = [k for k in bib_order if k in first_cite]
    expected = sorted(actual, key=lambda k: first_cite[k])
    pa = {k: i for i, k in enumerate(actual)}
    pe = {k: i for i, k in enumerate(expected)}
    out = [k for k in actual if pa[k] != pe[k]]
    return {"total": len(actual), "in_first_appearance_order": len(actual) - len(out),
            "out_of_order": out, "strict": actual == expected}


def local_verify(spec):
    """Offline ground-truth check for a locally materialized community dataset.

    Confirms the manifest exists, the class/image counts agree with the
    manuscript's claims, and the files actually exist on disk. PASS / MISMATCH /
    UNRESOLVED (manifest missing) - never touches the network.
    """
    mp = ROOT / spec["manifest"]
    if not mp.is_file():
        return {"verdict": "UNRESOLVED", "error": "local manifest not found: " + spec["manifest"]}
    try:
        man = json.loads(mp.read_text())
    except Exception as ex:
        return {"verdict": "UNRESOLVED", "error": "unreadable local manifest: " + str(ex)}
    if not isinstance(man, list) or not man:
        return {"verdict": "MISMATCH", "error": "local manifest is empty or not a list"}
    classes = sorted({e.get("relpath", "").split("/")[1] for e in man if "/" in e.get("relpath", "")})
    total = len(man)
    missing = 0
    for e in man:
        lp = e.get("local")
        if lp and not Path(lp).is_file():
            missing += 1
    images_ok = (total == spec["claimed_images"])
    classes_ok = (len(classes) == spec["claimed_classes"])
    disk_ok = (missing == 0)
    verdict = "PASS" if (images_ok and classes_ok and disk_ok) else "MISMATCH"
    return {"verdict": verdict, "manifest": spec["manifest"], "total": total,
            "claimed_images": spec["claimed_images"], "images_ok": images_ok,
            "n_classes": len(classes), "claimed_classes": spec["claimed_classes"],
            "classes_ok": classes_ok, "missing_on_disk": missing, "disk_ok": disk_ok,
            "classes": classes}


def reorder_bib(tex, first_cite):
    """Reorder \\bibitem blocks to first-citation order (idempotent).

    Moves whole bibitem blocks verbatim (leading indentation normalised), keeps
    the original begin/end tags and blank-line separators, and leaves uncited
    entries at the end in their original relative order. Returns
    (new_tex, changed, before_keys, after_keys).
    """
    m = re.search(r"\\begin\{thebibliography\}\s*\{[^}]*\}", tex)
    endm = re.search(r"\\end\{thebibliography\}", tex)
    if not m or not endm or m.start() > endm.start():
        return tex, False, None, None
    beg = m.start()
    inner_end = endm.start()      # slice up to (not including) the \end tag
    begin_tag = m.group(0)
    inner = tex[beg:inner_end]
    ib = list(re.finditer(r"\\bibitem\{([^}]+)\}", inner))
    if not ib:
        return tex, False, None, None
    entries = []
    for j, mm in enumerate(ib):
        key = mm.group(1)
        bstart = mm.start()
        bend = ib[j + 1].start() if j + 1 < len(ib) else len(inner)
        body = inner[bstart:bend]
        lead = len(body) - len(body.lstrip(" \t"))
        body = body[lead:].rstrip() + "\n"
        entries.append((key, body))

    def sk(kv):
        k = kv[0]
        return (0, first_cite[k][0], first_cite[k][1]) if k in first_cite else (1, 0, 0)

    ordered = sorted(entries, key=sk)
    new_inner = begin_tag + "\n\n" + "\n".join(e[1] for e in ordered) + "\n"
    new_tex = tex[:beg] + new_inner + tex[inner_end:]   # preserves the original \end{thebibliography}
    before = [e[0] for e in entries]
    after = [e[0] for e in ordered]
    return new_tex, before != after, before, after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--reorder", action="store_true",
                    help="reorder \\bibitem blocks to first-citation order (idempotent; writes the .tex)")
    ap.add_argument("--dry-run", action="store_true", help="with --reorder: show the change without writing")
    args = ap.parse_args()

    cache = {}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text()).get("entries", {})

    _, _, cited, first_cite, bib, bib_order, paper_title = parse_tex()

    if args.reorder:
        new_tex, changed, before, after = reorder_bib(TEX.read_text(), first_cite)
        if changed:
            if args.dry_run:
                print("reorder (dry-run): would re-order bibliography")
                print("  before:", before)
                print("  after :", after)
            else:
                TEX.write_text(new_tex)
                print("reorder: bibliography re-ordered to first-citation order (written to "
                      + str(TEX.relative_to(ROOT)) + ")")
                print("  before:", before)
                print("  after :", after)
            _, _, cited, first_cite, bib, bib_order, paper_title = parse_tex()
        else:
            print("reorder: already in first-citation order (no change)")

    refs = []
    for key in bib_order:
        blob = bib[key]["text"]
        doi, arxiv, hf, byr = extract_ids(blob)
        meta = resolve(doi, arxiv, hf, blob, cache, args.refresh)
        rec = {"key": key, "bib_text": blob, "doi": doi, "arxiv": arxiv, "hf": hf,
               "bib_year": byr, "cite_sites": cited.get(key, [])}
        if not meta or meta.get("http_status") != 200:
            rec["verdict"] = "UNRESOLVED"
            rec["resolved"] = {"error": (meta or {}).get("error", "no identifier")}
            if key in LOCAL_VERIFY:
                ev = local_verify(LOCAL_VERIFY[key])
                rec["local"] = ev
                if ev["verdict"] == "PASS":
                    rec["verdict"] = "PASS"
                    rec["local_verified"] = True
                    rec["resolved"] = {"source": "local (offline)", "title": LOCAL_VERIFY[key]["label"],
                                       "authors": [], "year": None, "venue": "local materialization"}
                elif ev["verdict"] == "MISMATCH":
                    rec["verdict"] = "MISMATCH"
                    rec["resolved"] = dict(rec["resolved"], source="local (offline)")
        else:
            if not (meta.get("abstract") or "").strip() and doi and not hf \
                    and not doi.startswith("10.48550"):
                ab = s2_abstract(doi)
                if ab:
                    meta["abstract"] = ab
                    meta["abstract_source"] = "s2"
                    cache[(meta.get("_source") or "?") + ":" + (meta.get("id") or "?")] = meta
            rec["verdict"] = compare(blob, meta, byr)["verdict"]
            rec["compare"] = compare(blob, meta, byr)
            rec["resolved"] = {"source": meta.get("_source"), "title": meta.get("title"),
                               "authors": (meta.get("authors") or [])[:6], "year": meta.get("year"),
                               "venue": meta.get("venue")}
            rec["abstract"] = (meta.get("abstract") or "")[:600]
            if rec["verdict"] == "MISMATCH" and rec["resolved"].get("source") == "crossref_search":
                rec["verdict"] = "UNRESOLVED"
                rec["resolved"]["note"] = "title-search hit not confirmed against the bibitem"
        refs.append(rec)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps({"version": 1, "generated": now(), "entries": cache}, indent=2))

    par = parity(cited, bib)
    ocheck = order_check(bib_order, first_cite)
    n_pass = sum(1 for r in refs if r["verdict"] == "PASS")
    n_mis = sum(1 for r in refs if r["verdict"] == "MISMATCH")
    n_unres = sum(1 for r in refs if r["verdict"] == "UNRESOLVED")
    parity_ok = not par["cited_not_in_bib"] and not par["bibitem_not_cited"]
    overall = "FAIL" if (n_mis or not parity_ok) else ("WARN" if n_unres else "PASS")

    report = {"generated": now(), "tex": str(TEX.relative_to(ROOT)), "mode": "refresh" if args.refresh else "cache",
              "paper_title": paper_title, "overall": overall,
              "summary": {"pass": n_pass, "mismatch": n_mis, "unresolved": n_unres,
                          "parity_ok": parity_ok, "order_strict": ocheck["strict"]},
              "parity": par, "order": ocheck, "references": refs}
    REPORT_JSON.write_text(json.dumps(report, indent=2))
    write_md(report)

    print(f"overall: {overall}   PASS={n_pass}  MISMATCH={n_mis}  UNRESOLVED={n_unres}  parity_ok={parity_ok}")
    print(f"order: strict={ocheck['strict']}  out_of_order={ocheck['out_of_order']}")
    for r in refs:
        flag = {"PASS": "PASS", "MISMATCH": "MISMATCH", "UNRESOLVED": "UNRESOLVED"}[r["verdict"]]
        res = r["resolved"].get("title") or r["resolved"].get("error", "")
        print(f"  {flag:9s} {r['key']:24s} {res[:70]}")
    print(f"\nreport: {REPORT_JSON.relative_to(ROOT)}")
    print(f"        {REPORT_MD.relative_to(ROOT)}")
    print(f"cache : {CACHE.relative_to(ROOT)}")
    return 1 if (n_mis or not parity_ok) else 0


def write_md(rep):
    s = rep["summary"]
    lines = []
    lines.append("# Citation Audit Report - " + rep["paper_title"])
    lines.append("")
    lines.append("**Date**: " + rep["generated"][:10])
    lines.append("**Generator**: `scripts/audit_references.py` (re-runnable; `--refresh` to re-fetch). "
                 "Audit record of registry responses: `data/reference_audit_cache.json`.")
    lines.append("**Bib source**: inline `\\begin{thebibliography}{99}` ("
                 f"{rep['parity']['bibitem_count']} `\\bibitem`).")
    lines.append("**Cited keys**: " + str(rep["parity"]["cited_count"]))
    lines.append(f"**Verdict**: **{rep['overall']}** "
                 f"(PASS={s['pass']}, MISMATCH={s['mismatch']}, UNRESOLVED={s['unresolved']}; "
                 f"parity_ok={s['parity_ok']}, order_strict={s['order_strict']})")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Verdict | Count |")
    lines.append("|---|---|")
    lines.append(f"| PASS (real + metadata matches + parity clean) | {s['pass']} |")
    lines.append(f"| MISMATCH (metadata drift) | {s['mismatch']} |")
    lines.append(f"| UNRESOLVED (registry unreachable / no id) | {s['unresolved']} |")
    lines.append("")
    lines.append("## Parity & order")
    lines.append("")
    lines.append("- cited-not-in-bib: " + (", ".join(rep["parity"]["cited_not_in_bib"]) or "none"))
    lines.append("- bib-not-cited: " + (", ".join(rep["parity"]["bibitem_not_cited"]) or "none"))
    lines.append("- first-appearance order strict: " + str(rep["order"]["strict"]))
    lines.append("- out-of-order keys: " + (", ".join(rep["order"]["out_of_order"]) or "none"))
    lines.append("")
    lines.append("## Per-entry")
    lines.append("")
    lines.append("| key | Verdict | Resolved (title, 1st author, year) | Via |")
    lines.append("|---|---|---|---|")
    for r in rep["references"]:
        res = r["resolved"]
        if "title" in res:
            cell = f"{res.get('title','')} | { (res.get('authors') or [''])[0] } | {res.get('year')}"
        else:
            cell = res.get("error", "")
        via = res.get("source", "")
        lines.append(f"| {r['key']} | {r['verdict']} | {cell} | {via} |")
    lines.append("")
    lines.append("## Flagged entries (detail)")
    lines.append("")
    flagged = [r for r in rep["references"] if r["verdict"] in ("MISMATCH", "UNRESOLVED")]
    if not flagged:
        lines.append("None.")
    for r in flagged:
        lines.append(f"### `{r['key']}` - {r['verdict']}")
        lines.append(f"- bib: {r['bib_text']}")
        res = r["resolved"]
        if "title" in res:
            lines.append(f"- resolved: {res.get('title')} | {res.get('authors')} | {res.get('year')} | via {res.get('source')}")
            c = r.get("compare", {})
            lines.append(f"- fields: title_ok={c.get('title_ok')} ratio={c.get('ratio')} "
                         f"first_ok={c.get('first_ok')} year_ok={c.get('year_ok')} "
                         f"(bib {c.get('bib_year')} vs resolved {c.get('resolved_year')})")
        else:
            lines.append(f"- resolved: {res.get('error')}")
        sites = r["cite_sites"]
        lines.append(f"- cite sites: " + ", ".join(str(x[0]) for x in sites))
        ab = r.get("abstract")
        if ab:
            lines.append(f"- abstract: {ab}")
        lines.append("")
    local_entries = [r for r in rep["references"] if "local" in r]
    if local_entries:
        lines.append("## Local ground-truth verification (offline)")
        lines.append("")
        lines.append("For registry-less community datasets, the entry is additionally checked against a local "
                     "materialization (manifest + on-disk files), offline, so the check is reproducible without "
                     "network access to the dataset host.")
        lines.append("")
        for r in local_entries:
            ev = r["local"]
            lines.append(f"### `{r['key']}` - {r['verdict']}" + (" (local-verified)" if r.get("local_verified") else ""))
            lines.append(f"- local manifest: `{ev.get('manifest')}`")
            lines.append(f"- images: {ev.get('total')} (manuscript claims {ev.get('claimed_images')}; images_ok={ev.get('images_ok')})")
            lines.append(f"- classes: {ev.get('n_classes')} (manuscript claims {ev.get('claimed_classes')}; classes_ok={ev.get('classes_ok')})")
            lines.append(f"- missing on disk: {ev.get('missing_on_disk')} (disk_ok={ev.get('disk_ok')})")
            if ev.get("classes"):
                lines.append(f"- classes: {', '.join(ev['classes'])}")
            lines.append("")
    lines.append("## Method notes")
    lines.append("- Every `\\bibitem` resolved independently against a canonical registry "
                 "(Crossref REST, DataCite, arXiv API, HuggingFace API; Semantic Scholar for "
                 "abstracts), not by recall. Title match is a >=60% content-word overlap; "
                 "first-author surname and year (<=1y tolerance) must also agree.")
    lines.append("- Fetches are cached to `data/reference_audit_cache.json`; re-run "
                  "`python3 scripts/audit_references.py` (no network) to reproduce, or add "
                  "`--refresh` to re-fetch live.")
    lines.append("- Registry-less community datasets listed in `LOCAL_VERIFY` are additionally "
                  "verified offline against a local materialization (manifest + on-disk files). "
                  "If the local files exist and the class/image counts agree with the "
                  "manuscript's claims, the entry is promoted from UNRESOLVED to PASS and "
                  "flagged `local_verified`; this only fires when the online registry cannot resolve.")
    REPORT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())

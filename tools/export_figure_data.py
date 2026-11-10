#!/usr/bin/env python3
"""Derive docs/data/figure-data.json from the archived analyses (maintainers only; the archive is not in this repository).

    python tools/export_figure_data.py ../../work

tools/figures/make_figures.py draws every figure from that file.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data" / "figure-data.json"


def export_data(work: Path) -> None:
    sys.path.insert(0, str(ROOT / "tools"))
    sys.path.insert(0, str(ROOT))
    import build_report as br  # noqa: E402

    d = br.collect(work)
    gm = statistics.geometric_mean
    groups = {}
    for r in d["fresh"]:
        groups.setdefault((r["kit"], r["mode"]), []).append(r)
    fresh = []
    for (kit, mode), rs in groups.items():
        fresh.append(dict(kit=kit, mode=mode, n=len(rs), anthropic_over_stock_time=gm([r["anthropic"] / r["stock"] for r in rs]),
                          ours_over_stock_time=gm([r["ours"] / r["stock"] for r in rs])))
    fresh.sort(key=lambda x: x["ours_over_stock_time"])
    allr = d["fresh"]
    geo = dict(n=len(allr), anthropic_speed=1 / gm([r["anthropic"] / r["stock"] for r in allr]), ours_speed=1 / gm([r["ours"] / r["stock"] for r in allr]),
               ours_over_anthropic=gm([r["ratio"] for r in allr]))
    ex = [r for r in allr if r["mode"] == "exact"]
    geo.update(exact_n=len(ex), exact_anthropic_speed=1 / gm([r["anthropic"] / r["stock"] for r in ex]), exact_ours_speed=1 / gm([r["ours"] / r["stock"] for r in ex]))
    chains = []
    for r in d["resident"]:
        if r["kit"] == "atlasfold":
            h = next((x for x in d["holdout"] if x["kit"] == "atlasfold" and x["input"] == r["tokens"]), None)
            if h:
                chains.append(dict(kind="atlasfold", size=r["tokens"], stock=h["stock"], anthropic=h["anthropic"], ours=h["ours"], resident=r["resident"]))
    b = next(x for x in d["b300"] if x["tokens"] == 760)
    res = next(r for r in d["resident"] if r["label"] == "B300 · resident" and r["tokens"] == 760)
    resa = next(r for r in d["resident"] if r["label"].startswith("B300 · resident + our") and r["tokens"] == 760)
    chains.append(dict(kind="openfold3_b300", size=760, stock=b["stock"], anthropic=b["anthropic"], ours=b["ours"], resident=res["resident"], resident_adapter=resa["resident"]))
    series = {}
    leak = work / "holdout" / "openfold3-memleak" / "raw"
    for name, rel in (("unguarded", "armB2-serve-exact"), ("guarded", "armB2-fix-packs"), ("upstream", "armA-upstream-loop")):
        rows = [json.loads(line) for line in (leak / rel / "extracted" / rel / "probe.jsonl").read_text().splitlines() if line.strip()]
        post = sorted((r for r in rows if r["tag"] == "post"), key=lambda r: r["request"])
        series[name] = [[r["request"], r["smi_used_mib"]] for r in post]
    ml = d["memleak"]
    fast = {}
    fl = work / "holdout" / "openfold3-fastleak" / "raw" / "pulled" / "cand30b" / "probe.jsonl"
    gl = work / "holdout" / "openfold3-memleak" / "raw" / "armC-serve-fast" / "extracted" / "armC-serve-fast" / "probe.jsonl"
    for name, path in (("unguarded", gl), ("released", fl)):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        post = sorted((r for r in rows if r["tag"] == "post"), key=lambda r: r["request"])
        fast[name] = [[r["request"], r["smi_used_mib"]] for r in post]
    memory = dict(series=series, fast=fast, unguarded_growth_pct=ml["before"]["smi_used_mib"]["growth_pct"], guarded_growth_pct=ml["after"]["smi_used_mib"]["growth_pct"])
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(dict(provenance="derived by tools/export_figure_data.py from the archived analyses (see HYBRID.md); policy digest " + d["policy_digest"],
                                    fresh=fresh, geo=geo, chains=chains, memory=memory), indent=1))
    print("wrote", DATA)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("work", type=Path)
    export_data(ap.parse_args().work)

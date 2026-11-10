"""Normalise the per-kit quick-look summaries (written by different forks) into one table: work/holdout/wide/ROWS.json."""
import json
import sys
from pathlib import Path

W = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[3] / "work" / "holdout" / "wide"
rows = []


def load(name):
    try:
        return json.loads((W / name / "summary.json").read_text())
    except (OSError, ValueError):
        return None


c = load("colabfold")
if c:
    for key, label in (("barnase_barstar", "1BRS 608 res"), ("ubiquitin", "ubiquitin 76 res")):
        i = c["inputs"].get(key)
        if i:
            rows.append(dict(kit="colabfold", gpu="H100", mode="exact", input=label, stock=i["stock_median"], anthropic=i["anthropic_median"], resident=i["resident_mean"],
                             a_over_resident=i["ratio_anthropic_fresh_over_resident"], a_over_stock=1 / i["ratio_stock_over_anthropic_fresh"] if i.get("ratio_stock_over_anthropic_fresh") else None,
                             outputs="byte-identical (stock = Anthropic = resident)", verdict="no meaningful gain (inside run-to-run spread)", method="3 fresh runs per arm (median), 4-7 resident requests"))
c = load("caliby")
if c:
    m = c["fresh_median_seconds"]
    for k, label in (("A8", "8 sequences, workload A"), ("B8", "8 sequences, workload B"), ("T1", "1-sequence example")):
        rows.append(dict(kit="caliby", gpu="H100", mode="fast", input=label, stock=m[f"{k}/off"], anthropic=m[f"{k}/fast"], resident=None, a_over_resident=None,
                         a_over_stock=m[f"{k}/off"] / m[f"{k}/fast"], outputs="fresh repeats byte-identical (17/17, timestamp header removed)",
                         verdict="Anthropic fast is faster than stock" if m[f"{k}/off"] / m[f"{k}/fast"] > 1.05 else "no difference (cleaning dominates)", method="3 fresh runs per arm (median); resident not run"))
g = load("gpnstar")
if g:
    f = g["fresh_seconds"]
    rows.append(dict(kit="gpnstar", gpu="H100", mode="exact", input="shipped 8-variant example", stock=f["stock"]["median"], anthropic=f["anthropic_exact"]["median"], resident=None,
                     a_over_resident=None, a_over_stock=f["stock"]["median"] / f["anthropic_exact"]["median"], outputs="parquet bytes identical",
                     verdict="Anthropic exact is SLOWER than stock; resident process unusable (latency 8.8 -> 46.8 s, +840 MiB per request)", method="3 fresh runs per arm (median), 14 resident requests"))
for name, kit, gpu in (("genie3-resident", "genie3", "H100"), ("openfold3_ob0-b300", "openfold3_ob0", "B300")):
    s = load(name)
    if s and s.get("inputs"):
        for key, i in s["inputs"].items():
            a, r = i.get("anthropic"), i.get("resident")
            st = i.get("stock")
            rows.append(dict(kit=kit, gpu=gpu, mode=s["mode"], input=(f"length {key}" if kit == "genie3" else f"{int(key) * 76} tokens"), stock=st["trimmed"] if st else None, anthropic=a["trimmed"],
                             resident=r["trimmed"] if r else None, a_over_resident=i.get("ratio_anthropic_fresh_over_resident_trimmed"),
                             a_over_stock=(st["trimmed"] / a["trimmed"]) if st else None,
                             outputs="identical to fresh" if s["gates"].get("all_resident_outputs_identical_to_fresh") else "see summary",
                             verdict=("resident gain is large" if (i.get("ratio_anthropic_fresh_over_resident_trimmed") or 0) > 1.5 else "resident gain is small: the kit starts a subprocess per request"),
                             method="quick-look, trimmed means"))
p = load("proteinmpnn-resident")
if p:
    for key, i in p["inputs"].items():
        rows.append(dict(kit="proteinmpnn", gpu="H100", mode="exact", input=f"{key[1:]} sequences", stock=i["stock"]["trimmed"], anthropic=i["anthropic"]["trimmed"], resident=None, a_over_resident=None,
                         a_over_stock=i["stock"]["trimmed"] / i["anthropic"]["trimmed"], outputs="stock = Anthropic bytes",
                         verdict="resident not supported: second request exits 3 ('late activation refused')", method="quick-look, trimmed means"))
a = load("atlasfold-b300")
if a:
    rows.append(dict(kit="atlasfold", gpu="B300", mode="exact", input="ubiquitin 228", stock=None, anthropic=None, resident=None, a_over_resident=None, a_over_stock=None, outputs="-",
                     verdict="cannot run: pinned Triton 3.3.1 does not know sm_103a (stock and Anthropic both fail)", method="install + first run"))

r1 = load("rfdiffusion1")
if r1:
    a = r1["fresh_input_A"]
    for mode, label in (("exact", "insulin target, 80-res binder"), ("fast", "insulin target, 80-res binder")):
        rows.append(dict(kit="rfdiffusion1", gpu="H100", mode=mode, input=label, stock=a["off"]["median"], anthropic=a[mode]["median"], resident=60.7 if mode == "exact" else None,
                         a_over_resident=(a["exact"]["median"] / 60.7) if mode == "exact" else None, a_over_stock=a["off"]["median"] / a[mode]["median"],
                         outputs="exact = stock bytes; fast differs from stock by design; fresh repeats identical" if mode == "fast" else "exact = stock bytes; resident outputs identical to fresh",
                         verdict=("Anthropic exact is 1.60x faster than stock; resident gives no gain (the kit starts a driver subprocess per design)" if mode == "exact" else "Anthropic fast is 2.24x faster than stock (outputs differ from stock by design)"),
                         method="3 fresh runs per arm (median), 14 resident requests"))

bg = load("boltzgen")
if bg and isinstance(bg.get("fresh_seconds"), dict):
    f = bg["fresh_seconds"]
    rows.append(dict(kit="boltzgen", gpu="H100", mode="exact", input="PD-L1 spec, 2 designs", stock=f["stock_off"]["median"], anthropic=f["anthropic_exact"]["median"],
                     resident=bg["resident_seconds"]["mean_excluding_first"], a_over_resident=bg["ratios"]["resident_vs_anthropic_fresh (fresh/resident, >1 = resident faster)"],
                     a_over_stock=f["stock_off"]["median"] / f["anthropic_exact"]["median"], outputs="17 structure files byte-identical in every arm (metrics files differ even between fresh repeats)",
                     verdict="Anthropic exact is 1.97x faster than stock; resident gives no gain (the kit runs its steps as child processes)", method="3 fresh runs per arm (median), 13 resident requests"))
rf = load("rosettafold3")
if rf and isinstance(rf.get("fresh_seconds"), dict):
    f = rf["fresh_seconds"]
    for mode, key in (("exact", "exact"), ("fast", "fast")):
        rows.append(dict(kit="rosettafold3", gpu="H100", mode=mode, input="1BRS tiles", stock=f["stock"]["median"], anthropic=f[key]["median"], resident=115.3 if mode == "exact" else None,
                         a_over_resident=(f["exact"]["median"] / 115.3) if mode == "exact" else None, a_over_stock=f["stock"]["median"] / f[key]["median"],
                         outputs="not comparable: repeated fresh runs differ in structure bytes in every mode, stock included",
                         verdict="small gain over stock (exact 1.05x, fast 1.14x); resident gives none (the kit starts the model with subprocess.run)", method="3 fresh runs per mode (median), 5 resident requests"))
(W / "ROWS.json").write_text(json.dumps(rows, indent=1) + "\n")
print(len(rows), "rows")

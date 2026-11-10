"""Add family scopes to the policy: widen from exact recorded inputs to the tested input-size hull for kits whose
unseen-input holdout passed. A family exists only if (a) every tested input (discovery + holdout) leads the three-way rule to the
same provider and (b) the holdout verdict, under the owner's trimmed-mean rule, passed for every holdout input.

    python tools/build_policy_v2.py --work ../../work --out foldboost/data/policy-v1.json
(the file name stays policy-v1.json for the exact scopes; `families` and `schema` v2 are added in place)
"""
import argparse
import hashlib
import json
from pathlib import Path

# kit/contract -> sizes tested in discovery (tokens or residues, same measure as sizes.py), holdout dir + sizes, free options
FAMILIES = [
    dict(kit="genie3", contract="exact", free=["seed"], discovery=[224, 256], holdout="genie3", holdout_sizes=[192, 240, 288], size_note="design length"),
    dict(kit="openfold3", contract="fast", free=["runner_yaml_sha256"], require=["runner_yaml_template_seed"], discovery=[608, 684], holdout="openfold3-fast", holdout_sizes=[304, 912], size_note="polymer tokens"),
    dict(kit="openfold3", contract="exact", free=["runner_yaml_sha256"], require=["runner_yaml_template_seed"], discovery=[760, 836], holdout="openfold3-exact", holdout_sizes=[304, 912], size_note="polymer tokens"),
    dict(kit="openfold3_ob0", contract="fast", free=["runner_yaml_sha256"], require=["runner_yaml_template_seed"], discovery=[456, 532], holdout="openfold3-ob0", holdout_sizes=[228, 684], size_note="polymer tokens"),
    dict(kit="atlasfold", contract="exact", free=["seed"], discovery=[228, 304], holdout="atlasfold", holdout_sizes=[152, 380, 456], size_note="residues"),
    dict(kit="esm_if1", contract="fast", free=["seed", "chain"], discovery=[248, 395], holdout="esm_if1", holdout_sizes=[76, 164, 247], size_note="chain residues"),
    dict(kit="protenix_v1", contract="exact", free=["seeds"], discovery=[398, 796], holdout="protenix_v1", holdout_sizes=[199, 597, 995], size_note="polymer tokens"),
]


# Resident-process support per kit and mode, only where outputs were byte-identical to a fresh CLI over a multi-request run
# and latency did not creep. "unsupported" entries carry the measured reason; kits not listed are never served by a resident process.
RESIDENT = {
    "openfold3": {
        "exact": {"status": "supported", "beats_stock": True, "gpu_names": ["NVIDIA H100 80GB HBM3", "NVIDIA B300 SXM6 AC"],
                  "evidence": "work/holdout/b300/serve-study-b300 (B300, 1.45-1.58x), serve-b300-adapter, serve-h100-v2 (H100, 1.39-1.49x): 30 requests each, outputs identical to fresh, rc 0; H100 memory creep handled by the server's recycle limit (stress run: restart + fresh fallback, outputs identical)"},
        "fast": {"status": "supported", "gpu_names": ["NVIDIA H100 80GB HBM3"],
                 "evidence": "work/holdout/openfold3-resident/fast (H100): 30 requests, outputs identical to fresh, 1.60-1.82x vs fresh Anthropic fast; reserved GPU memory grew ~8 GB over 30 requests, so the server's recycle limit is required. Fast mode is refused by the kit on B300."},
    },
    "openfold3_ob0": {
        "exact": {"status": "supported", "gpu_names": ["NVIDIA B300 SXM6 AC"],
                  "evidence": "work/holdout/wide/openfold3_ob0-b300: quick-look, 1.86x (456 tok) and 1.99x (684 tok) vs fresh Anthropic exact, 16/16 outputs identical; GPU memory rose 4.7->12.6 GB over 16 requests, so the recycle limit applies"},
    },
    "genie3": {"exact": {"status": "unsupported", "evidence": "work/holdout/wide/genie3-resident: only 1.03-1.05x; the kit starts a driver subprocess per request (genie3_opt/design.py:485), so the resident parent never touches the GPU"}},
    "proteinmpnn": {"exact": {"status": "unsupported", "evidence": "work/holdout/wide/proteinmpnn-resident: second request exits 3 'late activation refused ... one mode per process'"}},
    "gpnstar": {"exact": {"status": "unsupported", "evidence": "work/holdout/wide/gpnstar: outputs identical but latency rose from 8.8 s to 46.8 s over 14 requests and GPU memory grew ~840 MiB per request"}},
    "colabfold": {"exact": {"status": "no_benefit", "evidence": "work/holdout/wide/colabfold: resident 1.05-1.06x vs Anthropic fresh on tiny inputs, inside run-to-run spread; Anthropic exact itself is 0.97-0.99x of stock"}},
    "enformer": {"exact": {"status": "unsupported", "evidence": "run.sh has no `exec python -m <pkg>` line to derive a resident launcher from"}},
    "enformer_deepmind": {"exact": {"status": "unsupported", "evidence": "run.sh has no `exec python -m <pkg>` line to derive a resident launcher from"}},
    "borzoi": {"exact": {"status": "unsupported", "evidence": "borzoi_opt.cli.cmd_sad runs each job in a subprocess, so a resident process saves nothing"}},
    "evo2": {"exact": {"status": "unsupported", "evidence": "run.sh ends in `python route/evo2_route.py`, not `exec python -m <pkg>`"}},
    "atlasfold": {"exact": {"status": "supported", "beats_stock": True, "gpu_names": ["NVIDIA H100 80GB HBM3"],
                            "evidence": "work/holdout/atlasfold-resident: 30 requests, outputs identical to fresh, GPU memory constant, 1.96x (152 tokens), 1.73x (380), 1.62x (456) vs fresh Anthropic exact; launcher via serve_wrapper ENTRY_OVERRIDES (atlasfold_opt.cli:main)"}},
    "boltz2": {"fast": {"status": "supported", "gpu_names": ["NVIDIA H100 80GB HBM3"],
                        "evidence": "work/holdout/boltz2-resident: outputs identical, all gates passed; small gain (1.08-1.12x) because boltz2 fast runs the model in a short-lived worker per request"}},
    "protenix_v1": {"exact": {"status": "unsupported", "evidence": "work/holdout/protenix_v1-resident: the first request works, every later one exits 3: 'a second InferenceRunner in one process: the kit binds one model per process'"}},
    "rfdiffusion1": {"exact": {"status": "no_benefit", "evidence": "work/holdout/wide/rfdiffusion1: resident 0.99-1.00x vs fresh exact; the kit starts a driver subprocess per design call. (Anthropic exact is 1.60x and fast 2.24x faster than stock here.)"}},
    "boltzgen": {"exact": {"status": "no_benefit", "evidence": "work/holdout/wide/boltzgen: resident 0.97x vs fresh Anthropic; the kit runs its pipeline steps as child processes. (Anthropic exact is 1.97x faster than stock.)"}},
    "rosettafold3": {"exact": {"status": "no_benefit", "evidence": "work/holdout/wide/rosettafold3: resident 1.00x; fold.py starts the model with subprocess.run. Outputs are not reproducible even between fresh runs of stock."}},
    "esm_if1": {"fast": {"status": "no_benefit", "evidence": "work/holdout/esm_if1-resident: outputs identical but only 1.02-1.05x; the kit spawns a worker process per request (2.2-2.4 s model load each time)"}},
}


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def holdout_verdict(work, name):
    for fname in ("analysis-trimmed.json", "analysis-with-trimmed.json", "analysis.json"):
        a = load(work / "holdout" / name / fname)
        if a and "inputs" in a:
            ins = [i for i in a["inputs"] if "pairwise" in i]
            if ins and all("passed_trimmed" in i for i in ins):
                return a, ins
    return None, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    policy = load(a.out)
    policy.pop("policy_sha256", None)
    policy.pop("families", None)
    families, skipped = [], []
    for spec in FAMILIES:
        scopes = [s for s in policy["scopes"] if s["kit"] == spec["kit"] and s["contract"] == spec["contract"]]
        analysis, hins = holdout_verdict(a.work, spec["holdout"])
        if not scopes:
            skipped.append((spec["kit"], spec["contract"], "no discovery scopes"))
            continue
        if analysis is None:
            skipped.append((spec["kit"], spec["contract"], "holdout analysis missing"))
            continue
        if not all(i["passed_trimmed"] for i in hins):
            skipped.append((spec["kit"], spec["contract"], "a holdout input failed the trimmed rule"))
            continue
        choices = {s["decision"]["provider"] for s in scopes} | {i.get("rule_choice_trimmed", i["rule_choice"]) for i in hins}
        if len(choices) != 1:
            skipped.append((spec["kit"], spec["contract"], f"providers differ across tested inputs: {sorted(choices)}"))
            continue
        provider = choices.pop()
        first = scopes[0]
        fixed = {k: v for k, v in first["match"]["options"].items() if k not in spec["free"]}
        sizes = sorted(spec["discovery"] + spec["holdout_sizes"])
        families.append({
            "family_id": f"{spec['kit']}.h100.{spec['contract']}.family",
            "kit": spec["kit"], "gpu": "H100", "contract": spec["contract"], "adapter": first["adapter"],
            "environment": first["environment"], "requires": first.get("requires", []), "weights": first.get("weights", []),
            "fixed_options": fixed, "free_options": spec["free"], "require_present": spec.get("require", []),
            "size_hull": [sizes[0], sizes[-1]], "size_measure": spec["size_note"], "tested_sizes": sizes,
            "decision": {"provider": provider, "basis": f"same three-way choice at all {len(sizes)} tested inputs; holdout passed the trimmed-mean rule"},
            "evidence": {"discovery_scopes": [s["scope_id"] for s in scopes], "holdout": spec["holdout"],
                         "holdout_rule": "trimmed mean ratio > 1 (owner's rule); original rule: " + ("passed" if all(i["passed"] for i in hins) else "failed on at least one input")},
        })
    policy["families"] = families
    policy["resident"] = RESIDENT
    policy["family_skipped"] = [{"kit": k, "contract": c, "reason": r} for k, c, r in skipped]
    policy["schema"] = "flashpair-hybrid-policy/2"
    policy["status"] = ("exact recorded inputs: discovery-frozen; family scopes: widened to the tested input-size hull after unseen-input holdouts (trimmed-mean rule); "
                        "everything else routes to anthropic")
    body = json.dumps(policy, indent=1, sort_keys=True, ensure_ascii=False)
    policy["policy_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    a.out.write_text(json.dumps(policy, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    for f in families:
        print("FAMILY", f["family_id"], f["decision"]["provider"], f["size_hull"], f["free_options"])
    for k, c, r in skipped:
        print("SKIPPED", k, c, "-", r)


if __name__ == "__main__":
    main()

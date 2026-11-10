"""Build foldboost/data/policy-v1.json from the archived H100 whole-CLI benchmark records.

For every tested input of the nine qualified experiments this records the three engines'
timings, the correctness group they share, pairwise paired-bootstrap bounds and the provider the
frozen selection rule picks. The rule is applied to these same measurements, so the table is a
*discovery* policy: it is not validated until it passes a holdout run on unused inputs
(see HOLDOUT_PLAN.json). The builder never edits an archived record.

    python tools/build_policy.py --work /path/to/x20/work --out foldboost/data/policy-v1.json
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foldboost.requests import directory_digest, file_sha256, parse_request  # noqa: E402

BOOTSTRAP_SEED = 20261109
BOOTSTRAP_SAMPLES = 10000
PRIORITY = ["anthropic", "stock", "ours"]  # tie-break among statistically undominated providers

# experiment id -> everything the builder needs. `cond` names the record field that identifies
# a tested input; `adapter` is the vendored candidate (flag value 1 enables it).
SPECS = [
    dict(exp="022", kit="openfold3_ob0", contract="fast", results="ob022-results/openfold3-ob0-h100-022", cond="copies", adapter="ob0-inline-021",
         analysis="ob022-analysis.json", manifest="ob022-raw-manifest.json", bundle="bundle-022", python=None),
    dict(exp="027", kit="openfold3", contract="fast", results="of3027-results/openfold3-h100-027", cond="copies", adapter="of3-inline-023",
         analysis="of3027-analysis.json", manifest="of3027-raw-manifest.json", bundle="bundle-027", python=None),
    dict(exp="030", kit="openfold3", contract="exact", results="of3030-results/openfold3-h100-030", cond="copies", adapter="of3-inline-023",
         analysis="of3030-analysis.json", manifest="of3030-raw-manifest.json", bundle="bundle-030", python=None),
    dict(exp="031", kit="boltz2", contract="fast", results="boltz031-results/boltz2-h100-031", cond="copies", adapter="boltz-mmap-031",
         analysis="boltz031-analysis.json", manifest="boltz031-raw-manifest.json", bundle="bundle-031", python=None),
    dict(exp="036", kit="proteinmpnn", contract="exact", results="mpnn036-results/proteinmpnn-h100-036", cond="condition", adapter="mpnn-triton-key-036",
         analysis="mpnn036-analysis.json", manifest="mpnn036-raw-manifest.json", bundle="bundle-036", python="3.11.5",
         requires=["warm_fingerprint_cache"]),
    dict(exp="037", kit="atlasfold", contract="exact", results="atlas037-results/atlasfold-h100-037", cond="tokens", adapter="atlas-mmap-037",
         analysis="atlas037-analysis.json", manifest="atlas037-raw-manifest.json", bundle="bundle-037", python="3.11.5",
         env_fixed={"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}),
    dict(exp="039", kit="esm_if1", contract="fast", results="esm039-results/esm-if1-h100-039", cond="condition", adapter="esmif1-mmap-039",
         analysis="esm039-analysis.json", manifest="esm039-raw-manifest.json", bundle="bundle-039", python="3.11.5"),
    dict(exp="040", kit="protenix_v1", contract="exact", results="ptx040-results/protenix-v1-h100-040", cond="condition", adapter="ptx-cif-040",
         analysis="ptx040-analysis.json", manifest="ptx040-raw-manifest.json", bundle="bundle-040", python=None),
    dict(exp="050", kit="genie3", contract="exact", results="genie050-results/genie3-h100-050", cond="condition", adapter="genie3-dispatch-050",
         analysis="genie050-analysis.json", manifest="genie050-raw-manifest.json", bundle="bundle-050", python="3.10.13"),
]

# Candidates measured and rejected (or never run): these scopes may never select "ours".
REJECTED = [
    ("011/013", "protenix_v1/atlasfold", "K2B candidate inactive: other providers served the attention calls"),
    ("014", "atlasfold", "transition operation slower after batch change"),
    ("024/026/028", "boltz2", "operation-level gain did not become a whole-CLI gain (0.974/0.985)"),
    ("032/035", "proteinmpnn", "lower bound below 1"),
    ("034", "protenix_v1", "checkpoint mmap lost at 597 tokens (lower bound 0.915)"),
    ("041/044", "rfdiffusion1", "lower bound below 1 after two tries; Anthropic stays"),
    ("042/046", "genie3", "lower bound below 1; superseded by 050 for the tested exact scope only"),
]


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_records(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pair_stats(slow: np.ndarray, fast: np.ndarray, idx: np.ndarray) -> dict:
    """Geometric-mean time ratio slow/fast with a one-sided 95% bootstrap lower bound (paired by repetition)."""
    ratios = slow / fast
    boot = np.exp(np.mean(np.log(ratios[idx]), axis=1))
    return {"ratio": float(np.exp(np.mean(np.log(ratios)))), "lower95": float(np.quantile(boot, 0.05))}


def build_scope(spec: dict, work: Path) -> list:
    runpod = work / "runpod"
    rows = load_records(runpod / spec["results"] / "records.jsonl")
    plan_row = next(r for r in rows if r["type"] == "plan")
    plan = plan_row.get("plan", plan_row)
    stack = json.loads((runpod / spec["results"].split("/")[0] / "installed-stack.json").read_text())
    runs = [r for r in rows if r["type"] == "run"]
    trios = [r for r in rows if r["type"] == "trio"]
    input_rows = {r["condition"]: r for r in rows if r["type"] == "input" and "condition" in r}
    keys = sorted({r[spec["cond"]] for r in runs})
    exp_dir = runpod / spec["results"]

    def hasher_for(condition_hint):
        def hasher(path: str) -> str:
            base = Path(path).name
            if base.endswith("-input.json") and condition_hint in input_rows:
                return input_rows[condition_hint]["input_sha256"]
            for table in ("input_sha256", "inputs_sha256"):
                mapping = plan.get(table)
                if isinstance(mapping, dict) and base in mapping:
                    return mapping[base]
            if base == "pdbs":
                return directory_digest(plan["inputs_sha256"])
            local = exp_dir / base
            if local.is_file():
                return file_sha256(local)
            raise OSError(f"no recorded hash for {path}")
        return hasher

    scopes = []
    for key in keys:
        arms = {}
        for arm in ("stock", "anthropic", "ours"):
            ls = sorted((r for r in runs if r[spec["cond"]] == key and r["arm"] == arm), key=lambda r: r["rep"])
            assert [r["rep"] for r in ls] == list(range(6)), (spec["exp"], key, arm)
            assert all(r["exit_code"] == 0 for r in ls)
            arms[arm] = ls
        assert all(r["candidate_active"] for r in arms["ours"]), (spec["exp"], key, "ours inactive")
        cond_trios = [t for t in trios if t.get(spec["cond"], key) == key]
        stock_eq, ours_eq = _equalities(cond_trios)
        ours_cmd = arms["ours"][1]["command"]
        anth_cmd = arms["anthropic"][1]["command"]
        hasher = hasher_for(arms["ours"][1].get("condition", key if spec["cond"] == "condition" else None))
        seen_paths: dict = {}

        def recording(path, _inner=hasher):
            seen_paths[path] = _inner(path)
            return seen_paths[path]
        req_ours = parse_request(spec["kit"], ours_cmd, recording)
        req_anth = parse_request(spec["kit"], anth_cmd, hasher_for(arms["anthropic"][1].get("condition", key if spec["cond"] == "condition" else None)))
        assert not req_ours.problems and not req_ours.unrecognized, (spec["exp"], key, req_ours)
        assert req_ours.match_view() == req_anth.match_view(), (spec["exp"], key, req_ours.match_view(), req_anth.match_view())
        times = {a: np.array([_seconds(r) for r in ls[1:]]) for a, ls in arms.items()}  # repetition 0 is the excluded warmup
        members = ["stock", "anthropic", "ours"] if (spec["contract"] == "exact" and stock_eq and ours_eq) else ["anthropic", "ours"]
        assert ours_eq, (spec["exp"], key, "ours output differs from Anthropic")
        idx = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 5, size=(BOOTSTRAP_SAMPLES, 5))
        pairs = {}
        for a in members:
            for b in members:
                if a != b:
                    pairs[f"{a}_over_{b}"] = pair_stats(times[b], times[a], idx)  # >1 means a is faster than b
        dominated = {b: sorted(a for a in members if a != b and pairs[f"{a}_over_{b}"]["lower95"] >= 1.0) for b in members}
        undominated = [p for p in PRIORITY if p in members and not dominated[p]]
        choice = undominated[0]
        scope_id = f"{spec['kit']}.h100.{spec['contract']}.{spec['exp']}.{key}"
        scopes.append({
            "scope_id": scope_id,
            "kit": spec["kit"], "gpu": "H100", "contract": spec["contract"], "experiment": spec["exp"],
            "adapter": spec["adapter"],
            "match": req_ours.match_view(),
            "environment": {
                "torch": stack.get("torch"), "triton": stack.get("triton"), "python": spec["python"],
                "gpu_name": "NVIDIA H100 80GB HBM3", "env_fixed": spec.get("env_fixed", {}),
            },
            "requires": spec.get("requires", []),
            "weights": _weights(plan),
            "providers": {
                a: {"mode": "off" if a == "stock" else spec["contract"], "in_correctness_group": a in members,
                    "timed_seconds": [round(float(x), 6) for x in times[a]], "warmup_seconds": round(float(_seconds(arms[a][0])), 6)}
                for a in ("stock", "anthropic", "ours")
            },
            "correctness": {
                "ours_output_equals_anthropic_every_repetition": ours_eq,
                "stock_output_equals_anthropic_every_repetition": stock_eq,
                "group_members": members,
                "note": "exact scopes: all three native outputs identical" if len(members) == 3
                else "fast scope: stock output differs from the fast modes (or is unseeded); stock is not an eligible provider",
            },
            "pairwise_ratio_slower_over_faster": pairs,
            "dominated_by": dominated,
            "decision": {"provider": choice, "undominated": undominated,
                         "rule": "provider not beaten with 95% one-sided confidence by any other group member; ties go to anthropic, then stock, then ours"},
            "example": {"argv": [t for t in ours_cmd[2:]], "input_paths": dict(sorted(seen_paths.items())),
                        "note": "recorded command of the 'ours' arm (mode as measured); used by tests and as a request template"},
            "validation": {"status": "discovery-only", "holdout": None},
            "evidence": {
                "analysis": spec["analysis"], "analysis_sha256": sha_bytes((runpod / spec["analysis"]).read_bytes()),
                "raw_manifest": spec["manifest"], "raw_manifest_sha256": sha_bytes((runpod / spec["manifest"]).read_bytes()),
                "plan_sha256": sha_bytes((work / "three-engine-72" / spec["bundle"] / _plan_name(work, spec)).read_bytes()),
            },
        })
    return scopes


def _weights(plan: dict) -> list:
    """Weight files the plan pins by digest (empty when the kit pins them elsewhere, e.g. stock/PINS.json)."""
    out = []
    weights = plan.get("weights")
    if isinstance(weights, dict):
        for name, info in sorted(weights.items()):
            if isinstance(info, dict) and "sha256" in info:
                out.append({"name": info.get("file", name), "sha256": info["sha256"], "size": info.get("size_bytes", info.get("bytes"))})
            elif name == "file" and "sha256" in weights:
                out.append({"name": weights["file"], "sha256": weights["sha256"], "size": weights.get("bytes")})
                break
    if plan.get("weight_config"):
        out.append({"name": "config.yaml", "sha256": plan["weight_config"]["sha256"], "size": plan["weight_config"].get("size_bytes")})
    if plan.get("checkpoint_sha256"):
        out.append({"name": "protenix_v1 checkpoint", "sha256": plan["checkpoint_sha256"], "size": None})
    return out


def _plan_name(work: Path, spec: dict) -> str:
    base = work / "three-engine-72" / spec["bundle"]
    return "PLAN.json" if (base / "PLAN.json").exists() else "BENCHMARK_PLAN.json"


def _seconds(row: dict) -> float:
    return row["cli_seconds"] if "cli_seconds" in row else row["seconds"]


def _equalities(trios: list):
    def collect(*names):
        vals = [t[n] for t in trios for n in names if n in t]
        return bool(vals) and all(v is True for v in vals) and len(vals) == len(trios)
    if any("all_three_native_equal" in t or "all_three_native_scientific_equal" in t for t in trios):
        both = collect("all_three_native_equal", "all_three_native_scientific_equal")
        return both, both
    stock = collect("stock_anthropic_equal", "stock_vs_anthropic_bytes_equal")
    ours = collect("ours_anthropic_equal", "ours_vs_anthropic_bytes_equal")
    return stock, ours


def coverage(upstream: Path, scoped: set, extra: dict) -> list:
    out = []
    for kit in sorted(p.name for p in upstream.iterdir() if p.is_dir() and p.name != "common" and not p.name.startswith(".")):
        for gpu in ("H100", "B300"):
            status = "not_measured"
            note = "no archived three-engine whole-CLI measurement; Anthropic is the default"
            if (kit, gpu) in scoped:
                status, note = "scoped_evidence", "tested inputs only; see scopes"
            elif (kit, gpu) in extra:
                status, note = extra[(kit, gpu)]
            out.append({"kit": kit, "gpu": gpu, "status": status, "note": note})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    work = args.work
    scopes = []
    for spec in SPECS:
        scopes.extend(build_scope(spec, work))
    scoped_pairs = {(s["kit"], "H100") for s in scopes}
    extra = {
        ("rfdiffusion1", "H100"): ("candidate_rejected", "two candidates failed the whole-CLI gate (041, 044); Anthropic stays"),
        ("protenix_v2", "H100"): ("weights_access_blocked", "official checkpoint request returned 403; not a driver or kernel limit"),
        ("protenix_v2", "B300"): ("weights_access_blocked", "official checkpoint request returned 403; not a driver or kernel limit"),
    }
    policy = {
        "schema": "flashpair-hybrid-policy/1",
        "status": "discovery-frozen; NOT validated on a holdout; routing outside the listed scopes always selects anthropic",
        "upstream_commit": "f4f62fa6592ae4938d49b1757bea0cfeff9f468e",
        "default_provider": "anthropic",
        "bootstrap": {"seed": BOOTSTRAP_SEED, "samples": BOOTSTRAP_SAMPLES, "alpha": 0.05, "pairing": "by timed repetition, geometric mean of per-repetition time ratios"},
        "selection_rule": [
            "A request is in scope only if kit, GPU, requested mode, input digest, every matched option, torch/triton versions, fixed environment variables and listed preconditions equal an evidence scope.",
            "Within a scope the candidates are the providers whose outputs were identical to Anthropic's on every repetition (exact scopes: stock, anthropic, ours; fast scopes: anthropic, ours).",
            "A candidate is dominated when another candidate is faster with 95% one-sided paired-bootstrap confidence (lower bound >= 1.0).",
            "The chosen provider is the first undominated one in the order anthropic, stock, ours. A statistical tie therefore never selects our code.",
            "Anything else -- unknown input, other seed, other GPU, other mode, unmatched environment -- selects anthropic with the requested mode unchanged.",
        ],
        "scopes": scopes,
        "rejected_candidates": [{"experiments": e, "kit": k, "reason": r} for e, k, r in REJECTED],
        "coverage": coverage(work / "upstream", scoped_pairs, extra),
    }
    body = json.dumps(policy, indent=1, sort_keys=True, ensure_ascii=False)
    policy["policy_sha256"] = sha_bytes(body.encode())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(policy, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    for s in scopes:
        d = s["decision"]
        print(f"{s['scope_id']:44s} -> {d['provider']:9s} undominated={d['undominated']}")


if __name__ == "__main__":
    main()

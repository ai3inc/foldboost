"""Apply the pre-registered holdout gates (HOLDOUT_PLAN.json) to one tool's records.jsonl.

    python tools/analyze_holdout.py --records RESULTS/records.jsonl --kit genie3 --contract exact \
        --cond condition --bootstrap-seed 7010043 --out analysis.json

`--cond` names the record field that identifies an input (condition, copies or tokens). The same
record format as the discovery harnesses is expected. The analyzer never edits its inputs.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_policy import PRIORITY, _equalities, _seconds, pair_stats  # noqa: E402


def trim_mean(x):
    """Mean without the fastest and slowest value: one host stall cannot decide a verdict (owner's rule, applied after the genie3 stall)."""
    x = np.sort(np.asarray(x, dtype=float))
    return float(x[1:-1].mean()) if len(x) > 2 else float(x.mean())


def trimmed_pair(slow, fast, idx):
    """Ratio of trimmed mean times slow/fast with a one-sided 95% bootstrap lower bound over resampled repetitions (paired)."""
    ratio = trim_mean(slow) / trim_mean(fast)
    boot = np.array([trim_mean(slow[i]) / trim_mean(fast[i]) for i in idx])
    return {"ratio": ratio, "lower95": float(np.quantile(boot, 0.05))}


def analyze(rows, kit, contract, cond, seed):
    runs = [r for r in rows if r["type"] == "run"]
    trios = [r for r in rows if r["type"] == "trio"]
    complete = [r for r in rows if r["type"] in ("complete", "stopped")]
    out = {"kit": kit, "contract": contract, "terminal_record": complete[-1] if complete else None, "inputs": []}
    for key in sorted({r[cond] for r in runs}):
        arms = {a: sorted((r for r in runs if r[cond] == key and r["arm"] == a), key=lambda r: r["rep"]) for a in ("stock", "anthropic", "ours")}
        item = {"input": key, "gates": {}}
        complete_reps = all([r["rep"] for r in v] == list(range(6)) and all(r["exit_code"] == 0 for r in v) for v in arms.values())
        item["gates"]["all_runs_present_and_exit_zero"] = complete_reps
        if not complete_reps:
            item["passed"] = False
            out["inputs"].append(item)
            continue
        stock_eq, ours_eq = _equalities([t for t in trios if t.get(cond, key) == key])
        item["gates"]["outputs_equal_to_anthropic"] = bool(ours_eq) and (stock_eq if contract == "exact" else True)
        item["gates"]["ours_active_every_rep"] = all(r.get("candidate_active") for r in arms["ours"])
        item["gates"]["other_arms_show_no_candidate_activity"] = not any(r.get("candidate_active") for r in arms["anthropic"] + arms["stock"])
        t = {a: np.array([_seconds(r) for r in v[1:]]) for a, v in arms.items()}
        idx = np.random.default_rng(seed).integers(0, 5, size=(10000, 5))
        members = ["stock", "anthropic", "ours"] if (contract == "exact" and stock_eq and ours_eq) else ["anthropic", "ours"]
        pairs = {f"{a}_over_{b}": pair_stats(t[b], t[a], idx) for a in members for b in members if a != b}
        item["gates"]["speed_ours_vs_anthropic_lower95_ge_1"] = pairs["ours_over_anthropic"]["lower95"] >= 1.0
        dominated = {b: sorted(a for a in members if a != b and pairs[f"{a}_over_{b}"]["lower95"] >= 1.0) for b in members}
        undominated = [p for p in PRIORITY if p in members and not dominated[p]]
        item.update(mean_seconds={a: round(float(v.mean()), 4) for a, v in t.items()}, timed_seconds={a: [round(float(x), 4) for x in v] for a, v in t.items()},
                    pairwise=pairs, group_members=members, rule_choice=undominated[0], undominated=undominated)
        item["pairwise_trimmed"] = {f"{a}_over_{b}": trimmed_pair(t[b], t[a], idx[:2000]) for a in members for b in members if a != b}
        item["trimmed_mean_seconds"] = {a: round(trim_mean(v), 4) for a, v in t.items()}
        # Three-way choice under the trimmed rule: X dominates Y when trimmed mean(Y)/trimmed mean(X) >= 1.02 (a 2% margin so noise never selects our code).
        pt = item["pairwise_trimmed"]
        dom_t = {b: sorted(a for a in members if a != b and pt[f"{a}_over_{b}"]["ratio"] >= 1.02) for b in members}
        und_t = [q for q in PRIORITY if q in members and not dom_t[q]]
        item["rule_choice_trimmed"] = und_t[0]
        item["undominated_trimmed"] = und_t
        item["passed_original"] = all(item["gates"].values())
        # Owner's rule (set after the genie3 stall): headline verdict = trimmed-mean ratio > 1; the bootstrap bound over 5 trimmed reps is shown but not decisive.
        trimmed_ok = {**item["gates"], "speed_ours_vs_anthropic_lower95_ge_1": item["pairwise_trimmed"]["ours_over_anthropic"]["ratio"] > 1.0}
        item["passed_trimmed"] = all(trimmed_ok.values())
        item["passed"] = item["passed_original"]
        out["inputs"].append(item)
    out["kit_accepted"] = bool(out["inputs"]) and all(i["passed"] for i in out["inputs"])
    out["kit_accepted_trimmed"] = bool(out["inputs"]) and all(i.get("passed_trimmed", False) for i in out["inputs"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--kit", required=True)
    ap.add_argument("--contract", choices=["exact", "fast"], required=True)
    ap.add_argument("--cond", required=True)
    ap.add_argument("--bootstrap-seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = [json.loads(line) for line in a.records.read_text().splitlines() if line.strip()]
    result = analyze(rows, a.kit, a.contract, a.cond, a.bootstrap_seed)
    a.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    for i in result["inputs"]:
        print(i["input"], "PASS" if i["passed"] else "FAIL", "| trimmed:", "PASS" if i.get("passed_trimmed") else "FAIL", i.get("mean_seconds"), i["gates"])
    print("KIT", a.kit, "ACCEPTED" if result["kit_accepted"] else "REJECTED", "| trimmed rule:", "ACCEPTED" if result["kit_accepted_trimmed"] else "REJECTED")


if __name__ == "__main__":
    main()

"""Apply the pre-registered resident-process gates (HOLDOUT_PLAN.json amendment resident-process-openfold3).

    python tools/analyze_serve.py --records RESULTS/records.jsonl --bootstrap-seed 7910043 --out analysis.json
"""
import argparse
import json
from pathlib import Path

import numpy as np


def trim_mean(x):
    x = np.sort(np.asarray(x, dtype=float))
    return float(x[1:-1].mean()) if len(x) > 2 else float(x.mean())


def analyze(rows, seed, resamples=10000):
    fresh = [r for r in rows if r["type"] == "fresh"]
    res = [r for r in rows if r["type"] == "resident"]
    ready = next((r for r in rows if r["type"] == "server_ready"), None)
    complete = any(r["type"] == "complete" for r in rows)
    rng = np.random.default_rng(seed)
    out = {"complete": complete, "inputs": [], "gates": {}}
    out["gates"]["server_first_request_matches_fresh"] = bool(ready and ready.get("first_request_matches_fresh"))
    out["gates"]["all_resident_outputs_identical_to_fresh"] = bool(res) and all(r.get("matches_fresh") is True for r in res)
    out["gates"]["all_resident_requests_rc0_none_refused"] = bool(res) and all(r["rc"] == 0 for r in res)
    mem = [r["gpu_mib"] for r in sorted(res, key=lambda r: r["i"])]
    if len(mem) >= 10:
        growth = mem[-1] / mem[9] - 1.0
        out["memory"] = {"after_10th_mib": mem[9], "after_last_mib": mem[-1], "growth": growth}
        out["gates"]["gpu_memory_growth_le_5pct"] = growth <= 0.05
    else:
        out["gates"]["gpu_memory_growth_le_5pct"] = False
    for c in sorted({r["copies"] for r in fresh}):
        f = np.array([r["seconds"] for r in fresh if r["copies"] == c and not r["warmup"]])
        q = [r for r in res if r["copies"] == c]
        s = np.array([r["seconds"] for r in q])
        if len(f) < 5 or len(s) == 0:
            out["inputs"].append({"copies": c, "passed": False, "reason": "not enough timings"})
            continue
        ratios = np.array([f[rng.integers(0, len(f), len(f))].mean() / s[rng.integers(0, len(s), len(s))].mean() for _ in range(resamples)])
        lower = float(np.quantile(ratios, 0.05))
        tratios = np.array([trim_mean(f[rng.integers(0, len(f), len(f))]) / trim_mean(s[rng.integers(0, len(s), len(s))]) for _ in range(2000)])
        trimmed = {"ratio": round(trim_mean(f) / trim_mean(s), 4), "lower95": round(float(np.quantile(tratios, 0.05)), 4)}
        first_of_shape = min(q, key=lambda r: r["i"])
        later = np.array([r["seconds"] for r in q if r is not first_of_shape])
        out["inputs"].append({
            "copies": c, "tokens": c * 76, "fresh_mean_seconds": round(float(f.mean()), 3), "fresh_timed": len(f),
            "resident_mean_seconds": round(float(s.mean()), 3), "resident_requests": len(s),
            "resident_mean_excluding_first_of_this_shape": round(float(later.mean()), 3) if len(later) else None,
            "first_request_of_this_shape_seconds": round(first_of_shape["seconds"], 3),
            "ratio_fresh_over_resident": round(float(f.mean() / s.mean()), 4), "lower95": round(lower, 4), "passed": lower >= 1.0,
            "trimmed": trimmed, "passed_trimmed": trimmed["ratio"] > 1.0,
        })
    out["gates"]["speed_lower95_ge_1_every_input"] = bool(out["inputs"]) and all(i["passed"] for i in out["inputs"])
    out["passed"] = complete and all(out["gates"].values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--bootstrap-seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = [json.loads(line) for line in a.records.read_text().splitlines() if line.strip()]
    result = analyze(rows, a.bootstrap_seed)
    a.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()

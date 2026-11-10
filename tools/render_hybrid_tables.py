"""Render the per-input three-engine table and the 72-row coverage summary from policy-v1.json.

    python tools/render_hybrid_tables.py > HYBRID_TABLES.md
Every number is read from the policy, which was built from the archived raw records.
"""
import collections
import json
import sys
from pathlib import Path

policy = json.loads((Path(__file__).resolve().parents[1] / "foldboost" / "data" / "policy-v1.json").read_text())
w = sys.stdout.write
w("# Three-engine table (H100, whole fresh CLI, discovery inputs)\n\n")
w(f"Policy digest `{policy['policy_sha256'][:16]}`. Status: {policy['status']}.\n\n")
w("Mean seconds of 5 timed fresh CLI processes per provider (1 excluded warmup). `A/ours` is Anthropic's mode time divided by ours; the bound is the one-sided 95% paired-bootstrap lower bound. "
  "`stock` is the unmodified release (`--mode off`); it is in the correctness group only where its outputs were identical to Anthropic's exact mode.\n\n")
w("| tool | mode | input | stock s | Anthropic s | ours s | A/ours | lower95 | stock in group | router choice |\n|---|---|---|---:|---:|---:|---:|---:|:--:|---|\n")
for s in policy["scopes"]:
    m = {a: sum(v["timed_seconds"]) / len(v["timed_seconds"]) for a, v in s["providers"].items()}
    p = s["pairwise_ratio_slower_over_faster"]["ours_over_anthropic"]
    label = s["scope_id"].split(".")[-1]
    in_group = "yes" if "stock" in s["correctness"]["group_members"] else "no"
    w(f"| {s['kit']} | {s['contract']} | {label} | {m['stock']:.2f} | {m['anthropic']:.2f} | {m['ours']:.2f} | {p['ratio']:.3f} | {p['lower95']:.3f} | {in_group} | **{s['decision']['provider']}** |\n")
w("\nFindings that need the three-way view:\n\n")
for s in policy["scopes"]:
    if s["decision"]["provider"] == "stock":
        o = s["pairwise_ratio_slower_over_faster"]
        w(f"- `{s['scope_id']}`: Anthropic's exact mode is slower than unmodified stock here (stock/Anthropic bound {o['stock_over_anthropic']['lower95']:.3f}); "
          f"our adapter beats Anthropic but is statistically tied with stock (ours/stock bound {o['ours_over_stock']['lower95']:.3f}), so the rule picks stock.\n")
w("\n# Coverage of the 36 kits x 2 GPUs\n\n| status | rows |\n|---|---:|\n")
c = collections.Counter(r["status"] for r in policy["coverage"])
for k, v in sorted(c.items()):
    w(f"| {k} | {v} |\n")
w(f"\nTotal {sum(c.values())}. `scoped_evidence` means the listed inputs only; every other row routes to Anthropic.\n")

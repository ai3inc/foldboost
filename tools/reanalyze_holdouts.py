"""Re-run tools/analyze_holdout.py over every holdout result folder and write analysis-trimmed.json beside the fork's own analysis.json."""
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

W = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[3] / "work" / "holdout"
SEEDS = {"genie3": 7010043, "proteinmpnn": 7110043, "esm_if1": 7310043, "openfold3": 7210043, "openfold3_ob0": 7220043, "boltz2": 7410043, "protenix_v1": 7510043, "atlasfold": 7610043}
HERE = Path(__file__).resolve().parent
for d in sorted(os.listdir(W)):
    a_path = W / d / "analysis.json"
    if not a_path.exists() or "resident" in d or d.startswith(("serve", "b300", "wide")):
        continue
    a = json.loads(a_path.read_text())
    if "kit" not in a:
        continue
    recs = [r for r in glob.glob(f"{W}/{d}/**/records.jsonl", recursive=True) if "attempt" not in r]
    if not recs:
        continue
    rows = [json.loads(line) for line in open(recs[0]) if line.strip()]
    run = next((r for r in rows if r["type"] == "run"), None)
    if run is None:
        continue
    cond = "condition" if "condition" in run else ("copies" if "copies" in run else "tokens")
    out = W / d / "analysis-trimmed.json"
    r = subprocess.run([sys.executable, str(HERE / "analyze_holdout.py"), "--records", recs[0], "--kit", a["kit"], "--contract", a["contract"], "--cond", cond,
                        "--bootstrap-seed", str(SEEDS[a["kit"]]), "--out", str(out)], capture_output=True, text=True)
    print(d, "->", (r.stdout.strip().splitlines() or [r.stderr[-120:]])[-1])

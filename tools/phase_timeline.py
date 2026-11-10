"""Run a command and stamp every output line with the seconds since start; report where the wall time goes.

    python3 phase_timeline.py --out run.timeline [--bash-trace] -- bash run.sh pred --config b300 --mode fast ...

--bash-trace sets PS4 so `bash -x` lines carry a clock, which exposes time spent inside run.sh before the
program starts. The report lists the largest gaps between consecutive output lines: a gap is time during
which the process printed nothing, attributed to the line that ended it. Standard library only.
"""
import argparse
import os
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bash-trace", action="store_true")
    ap.add_argument("--top", type=int, default=18)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if a.bash_trace and cmd[:2] == ["bash", "run.sh"]:
        cmd = ["bash", "-x"] + cmd[1:]
        env["PS4"] = "+ "
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, bufsize=0)
    rows, buf = [], b""
    while True:
        chunk = os.read(p.stdout.fileno(), 65536)
        if not chunk:
            break
        now = time.perf_counter() - t0
        buf += chunk.replace(b"\r", b"\n")
        *lines, buf = buf.split(b"\n")
        rows.extend((now, ln.decode("utf-8", "replace")) for ln in lines if ln.strip())
    rc = p.wait()
    total = time.perf_counter() - t0
    with open(a.out, "w") as f:
        for t, ln in rows:
            f.write(f"{t:8.3f}  {ln[:400]}\n")
        f.write(f"{total:8.3f}  [exit {rc}]\n")
    gaps = sorted(((rows[i][0] - rows[i - 1][0], rows[i - 1][0], rows[i][1]) for i in range(1, len(rows))), reverse=True)[: a.top]
    print(f"total {total:.2f}s exit {rc}; {len(rows)} lines; largest silent gaps (seconds, started at, line that ended the gap):")
    for g, start, ln in gaps:
        print(f"  {g:6.2f}  @{start:7.2f}  {ln[:150]}")
    sys.exit(rc)


if __name__ == "__main__":
    main()

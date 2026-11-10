"""A stand-in kit entry for the resident-runner tests."""
import os
import sys

CALLS = []


def main(argv):
    CALLS.append(list(argv))
    flags = {f: v for f, v in __import__('foldboost.serve_kit', fromlist=['x']).flag_items(argv) if f}
    print("fakekit stdout", len(CALLS), flush=True)
    print("fakekit stderr", len(CALLS), file=sys.stderr, flush=True)
    os.write(1, b"fakekit raw fd1\n")
    if flags.get("--input") == "fail":
        raise SystemExit(3)
    if flags.get("--input") == "boom":
        raise RuntimeError("kit crash")
    out = flags.get("--out")
    if out:
        with open(out, "w") as f:
            f.write(f"{flags.get('--mode')}:{flags.get('--input')}:{os.getpid()}\n")
    return 0

"""python -m foldboost {run,select,install-gate,verify,coverage} ... [-- kit arguments]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import adapters
from .selector import load_policy


def split_argv(argv: list) -> tuple[list, list]:
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def add_route_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--kit", required=True, help="kit name, e.g. genie3, proteinmpnn, openfold3")
    p.add_argument("--kit-dir", required=True, help="directory holding the kit's run.sh")
    p.add_argument("--python", default=sys.executable, help="interpreter of the kit's environment (versions are read from its metadata)")
    p.add_argument("--weights", action="append", default=[], help="file or directory to search for policy-listed weight files (sha256 verified, cached)")
    p.add_argument("--trust-weights", action="store_true", help="skip weight verification (recorded in the log as trusted, not verified)")
    p.add_argument("--allow-cold", action="store_true", help="treat warm-cache preconditions as met; the first run pays an unmeasured preparation cost")
    p.add_argument("--cache-dir", default="~/.cache/foldboost")
    p.add_argument("--log-dir", default="foldboost-logs")
    p.add_argument("--site-packages", default=None, help="site-packages that holds the gate (default: this interpreter's)")
    p.add_argument("--server", default=None, help="Unix socket of a resident serve_kit process; used when it holds the same Anthropic code path, else a fresh CLI runs")


def main(argv: list | None = None) -> int:
    own, kit_argv = split_argv(list(sys.argv[1:] if argv is None else argv))
    ap = argparse.ArgumentParser(prog="foldboost", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "select"):
        p = sub.add_parser(name, help="route and run" if name == "run" else "route only; print the decision")
        add_route_flags(p)
        p.add_argument("--allow-inactive", action="store_true", help="exit with the kit's status even if our adapter did not fire")
        p.add_argument("--dry-run", action="store_true")
    g = sub.add_parser("install-gate", help="write the interpreter gate into a Python environment's site-packages")
    g.add_argument("--python", required=True)
    g.add_argument("--dry-run", action="store_true")
    sv = sub.add_parser("serve", help="manage a resident kit process: start | status | stop")
    sv.add_argument("action", choices=["start", "status", "stop"])
    sv.add_argument("--socket", required=True)
    sv.add_argument("--kit")
    sv.add_argument("--kit-dir")
    sv.add_argument("--python", default=None, help="interpreter of the kit's environment (its directory goes first on PATH)")
    sv.add_argument("--max-requests", type=int, default=500)
    sv.add_argument("--wait", type=float, default=900)
    sub.add_parser("verify", help="check policy digest and vendored adapter integrity")
    sub.add_parser("coverage", help="print the kit x GPU coverage table")
    args = ap.parse_args(own)

    if args.cmd in ("run", "select"):
        if args.cmd == "select":
            args.dry_run = True
        if not kit_argv:
            ap.error("give the kit command after --, e.g. -- design --config h100 --mode exact ...")
        from .launch import run
        return run(args, kit_argv)
    if args.cmd == "install-gate":
        import subprocess
        out = subprocess.run([args.python, "-I", "-c", "import site;print(site.getsitepackages()[0])"], capture_output=True, text=True, check=True)
        target = adapters.install_gate(Path(out.stdout.strip()), dry_run=args.dry_run)
        print(("would write " if args.dry_run else "wrote ") + str(target))
        return 0
    if args.cmd == "serve":
        from . import serve_ctl
        path = str(Path(args.socket).expanduser())
        if args.action == "status":
            info = serve_ctl.status(path)
            print(json.dumps(info) if info else "not running")
            return 0 if info else 1
        if args.action == "stop":
            ok = serve_ctl.stop(path)
            print("stopped" if ok else "no server answered; stop file written")
            return 0
        if not (args.kit and args.kit_dir and kit_argv):
            ap.error("serve start needs --kit, --kit-dir and the kit's first request after --")
        return serve_ctl.start(args.kit, Path(args.kit_dir), path, kit_argv, args.python, args.max_requests, args.wait)
    if args.cmd == "verify":
        policy = load_policy()
        for adapter_id in adapters.registry():
            adapters.verify_adapter(adapter_id)
        used = {s["adapter"] for s in policy["scopes"]}
        missing = used - set(adapters.registry())
        if missing:
            print("scopes reference unknown adapters:", sorted(missing), file=sys.stderr)
            return 1
        print(f"ok: policy {policy['policy_sha256'][:16]}… {len(policy['scopes'])} scopes, {len(adapters.registry())} adapters; status: {policy['status']}")
        return 0
    if args.cmd == "coverage":
        for row in load_policy()["coverage"]:
            print(f"{row['kit']:18s} {row['gpu']:5s} {row['status']:24s} {row['note']}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

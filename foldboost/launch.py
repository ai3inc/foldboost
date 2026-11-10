"""Route one kit command through the frozen policy and run it, recording why.

The launcher changes only two things in the command it was given: the value after ``--mode``
(never between exact and fast; ``off`` is used only when the policy proved stock's output
identical to the exact modes for that input) and the environment of the child process (adapter
directory and enable flag). Nothing falls back silently: a selected adapter that did not fire is
reported and turns the exit status into 70 unless --allow-inactive is given.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import adapters
from .requests import file_sha256, parse_request, with_mode
from .selector import DEFAULT, Decision, Facts, load_policy, select

AUDIT_PATTERNS = (re.compile(r"\[flashpair-model-audit\] (\{.*\})"), re.compile(r"^FLASHP_[A-Z0-9_]+ (\{.*\})$", re.M))
PROBE = ("import sys,json,importlib.metadata as m\n"
         "def v(n):\n"
         "    try: return m.version(n)\n"
         "    except Exception: return None\n"
         "print(json.dumps({'python':'.'.join(map(str,sys.version_info[:3])),'torch':v('torch'),'triton':v('triton')}))\n")
INACTIVE_EXIT = 70


def probe_gpu() -> str | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=20, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    names = {line.strip() for line in out.stdout.splitlines() if line.strip()}
    return names.pop() if len(names) == 1 else None  # mixed GPUs: do not claim a match


def probe_python(python: str) -> dict:
    try:
        out = subprocess.run([python, "-I", "-c", PROBE], capture_output=True, text=True, timeout=60, check=True)
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}


def _receipts_path(cache_dir: Path) -> Path:
    return cache_dir / "weights-receipts.json"


def verify_weights(policy: dict, kit: str, locations: list, cache_dir: Path) -> set:
    """Hash weight files whose name and size match a policy weight; remember results by (path, size, mtime)."""
    wanted = {w["name"]: w for s in policy["scopes"] if s["kit"] == kit for w in (s.get("weights") or [])}
    if not wanted:
        return set()
    receipts_file = _receipts_path(cache_dir)
    try:
        receipts = json.loads(receipts_file.read_text())
    except (OSError, ValueError):
        receipts = {}
    files = []
    for loc in locations:
        p = Path(loc)
        files.extend(q for q in ([p] if p.is_file() else sorted(p.rglob("*"))[:5000]) if q.is_file())
    verified = set()
    for f in files:
        w = wanted.get(f.name)
        if not w or (w.get("size") and f.stat().st_size != w["size"]):
            continue
        st = f.stat()
        key = f"{f.resolve()}|{st.st_size}|{st.st_mtime_ns}"
        digest = receipts.get(key) or file_sha256(f)
        receipts[key] = digest
        if digest == w["sha256"]:
            verified.add(digest)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        receipts_file.write_text(json.dumps(receipts, sort_keys=True))
    except OSError:
        pass
    return verified


def collect_audits(log: str) -> list:
    audits = []
    for pattern in AUDIT_PATTERNS:
        for match in pattern.finditer(log):
            try:
                audits.append(json.loads(match.group(1)))
            except ValueError:
                continue
    return audits


def activation(audits: list) -> dict:
    calls = sum(int(a.get("candidate_calls", 0)) for a in audits)
    fallbacks = sum(int(a.get("fallback_calls", 0)) for a in audits)
    return {"audit_lines": len(audits), "candidate_calls": calls, "fallback_calls": fallbacks, "active": calls > 0}


def build_facts(args, policy: dict, kit: str) -> Facts:
    probe = probe_python(args.python)
    env = dict(os.environ)
    cache_dir = Path(args.cache_dir).expanduser()
    satisfied = set()
    fp_cache = cache_dir / "mpnn-fingerprint"
    if args.allow_cold or (fp_cache.is_dir() and any(fp_cache.glob("*.json"))):
        satisfied.add("warm_fingerprint_cache")
    verified = set()
    if args.weights:
        verified = verify_weights(policy, kit, args.weights, cache_dir)
    if args.trust_weights:
        verified = {w["sha256"] for s in policy["scopes"] if s["kit"] == kit for w in (s.get("weights") or [])}
    return Facts(gpu_name=probe_gpu(), torch=probe.get("torch"), triton=probe.get("triton"), python=probe.get("python"),
                 env=env, satisfied=satisfied, verified_weights=verified)


def route(args, kit_argv: list) -> tuple[Decision, object, Facts, dict]:
    policy = load_policy()
    request = parse_request(args.kit, kit_argv)
    facts = build_facts(args, policy, args.kit)
    decision = select(policy, request, facts)
    return decision, request, facts, policy


def child_environment(decision: Decision, args) -> dict:
    env = dict(os.environ)
    env.update(decision.fixed_env)
    if decision.adapter:
        extra = {}
        if decision.adapter_enabled and decision.adapter == "mpnn-triton-key-036":
            extra["FLASHP_FINGERPRINT_CACHE"] = str(Path(args.cache_dir).expanduser() / "mpnn-fingerprint")
        env.update(adapters.adapter_environment(decision.adapter, decision.adapter_enabled, extra))
    return env


def strip_flag(argv: list, flag: str) -> list:
    out, i = [], 0
    while i < len(argv):
        if argv[i] == flag:
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def try_resident(args, kit_argv: list, decision: Decision, log_path: str, policy: dict, gpu_name: str | None = None, requested_mode: str | None = None) -> tuple[int | None, dict]:
    """Send the request to a resident server when one is listening and its code path equals the chosen one.

    Returns (exit status, info) or (None, info) when the request must run as a fresh CLI. The server only ever
    runs Anthropic's mode without our adapter (the stage-two adapter result is not routed), and it refuses any
    request whose configuration differs from the process it holds (rc 75)."""
    import socket as _socket
    info: dict = {"socket": args.server}
    # A stock decision (chosen inside an exact scope because stock's output was byte-identical to the exact mode) may be served by the
    # resident process when the policy records that the resident exact mode beat fresh stock; the request then runs in its requested exact mode.
    mode = requested_mode if decision.provider == "stock" else decision.mode
    if decision.provider not in ("anthropic", "ours", "stock") or mode not in ("exact", "fast"):
        info["skipped"] = f"provider {decision.provider} mode {decision.mode} is not served by the resident process"
        return None, info
    support = (policy.get("resident") or {}).get(args.kit, {}).get(mode)
    if decision.provider == "stock" and not (support and support.get("beats_stock")):
        info["skipped"] = "stock was chosen and the resident process is not recorded as faster than fresh stock"
        return None, info
    if not support or support.get("status") != "supported":
        info["skipped"] = f"resident mode is not validated for {args.kit} {decision.mode}" + (f" ({support['status']})" if support else "")
        return None, info
    if support.get("gpu_names") and gpu_name not in support["gpu_names"]:
        info["skipped"] = f"resident mode was validated for {support['gpu_names']}, not {gpu_name!r}"
        return None, info
    argv = strip_flag(list(kit_argv), "--config")
    argv = with_mode(argv, mode)
    try:
        with _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM) as s:
            s.settimeout(2)
            s.connect(args.server)
            s.settimeout(None)
            with s.makefile("rwb") as stream:
                stream.write((json.dumps({"ping": True}) + "\n").encode())
                stream.flush()
                hello = json.loads(stream.readline())
                info["server"] = hello
                if hello.get("adapter") and decision.provider != "ours":
                    info["skipped"] = "the resident process runs with an adapter enabled, but the policy chose Anthropic's unmodified path for this request"
                    return None, info
                info["adapter_in_resident_process"] = bool(hello.get("adapter"))
                stream.write((json.dumps({"argv": argv, "log": log_path, "cwd": os.getcwd()}) + "\n").encode())
                stream.flush()
                reply = json.loads(stream.readline())
    except (OSError, ValueError) as error:
        info["skipped"] = f"resident server unavailable: {error}"
        return None, info
    info["reply"] = reply
    if reply.get("rc") == 75:
        info["skipped"] = "refused by the resident process: " + str(reply.get("refused"))
        return None, info
    return int(reply["rc"]), info


def run(args, kit_argv: list) -> int:
    decision, request, facts, policy = route(args, kit_argv)
    command = ["bash", "run.sh"] + (with_mode(kit_argv, decision.mode) if decision.mode and request.mode else list(kit_argv))
    record = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "policy_sha256": policy["policy_sha256"], "policy_status": policy["status"],
        "request": {"kit": request.kit, "config": request.config, "mode": request.mode, "input_sha256": request.input_sha256, "options": request.options},
        "facts": {"gpu_name": facts.gpu_name, "torch": facts.torch, "triton": facts.triton, "python": facts.python},
        "decision": decision.as_dict(), "command": command, "kit_dir": str(args.kit_dir),
    }
    if args.dry_run:
        print(json.dumps(record, indent=1, sort_keys=True))
        return 0
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if getattr(args, "server", None):
        started = time.perf_counter()
        code, info = try_resident(args, kit_argv, decision, str(log_dir / "last-run.log"), policy, facts.gpu_name, request.mode)
        record["resident"] = info
        if code not in (None, 0):
            info["resident_failed_rc"] = code
            record["resident"] = info
            print(f"foldboost: the resident process returned {code}; re-running as a fresh CLI", file=sys.stderr)
            code = None
        if code is not None:
            seconds = time.perf_counter() - started
            record.update(route="resident", exit_code=code, wall_seconds=round(seconds, 3),
                          executed_provider="ours" if info.get("adapter_in_resident_process") else "anthropic")  # the resident server runs the kit's own exact/fast path; "decision" is what the policy chose
            log_file = log_dir / "last-run.log"
            if log_file.exists():
                sys.stdout.write(log_file.read_text(errors="replace"))
            with (log_dir / "decisions.jsonl").open("a") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            return code
        if info.get("skipped"):
            print("foldboost: resident server not used (" + str(info.get("skipped")) + "); running a fresh CLI", file=sys.stderr)
    if decision.adapter_enabled and not adapters.gate_installed(Path(args.site_packages) if args.site_packages else None):
        if not getattr(args, "server", None):
            print("foldboost: interpreter gate missing; run `python -m foldboost install-gate --python ...` first", file=sys.stderr)
            return 2
        # A resident deployment: the server itself runs Anthropic's unmodified path, so when it refuses or restarts, the fresh CLI that takes over
        # must not turn a missing optional gate into a failed request. Same mode, Anthropic's code, recorded and announced.
        decision = dataclasses.replace(decision, provider=DEFAULT, adapter=None, adapter_enabled=False, fixed_env={}, non_default=False,
                                       reason=decision.reason + "; downgraded to Anthropic's path: interpreter gate not installed")
        record["decision"] = decision.as_dict()
        record["downgraded"] = "interpreter gate missing: ran Anthropic's unmodified path (run `install-gate` to let the policy's adapter choice apply)"
        print("foldboost: WARNING " + record["downgraded"], file=sys.stderr)
    record["route"] = "fresh"
    env = child_environment(decision, args)
    started = time.perf_counter()
    chunks = []
    with subprocess.Popen(command, cwd=args.kit_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace") as proc:
        for line in proc.stdout:
            chunks.append(line)
            sys.stdout.write(line)
        code = proc.wait()
    seconds = time.perf_counter() - started
    log = "".join(chunks)
    act = activation(collect_audits(log))
    record.update(exit_code=code, wall_seconds=round(seconds, 3), activation=act,
                  log_sha256=hashlib.sha256(log.encode()).hexdigest())
    inactive = decision.adapter_enabled and not act["active"]
    if inactive:
        record["warning"] = "our adapter was selected but reported no candidate calls; the run used the original code path"
        print("foldboost: WARNING " + record["warning"], file=sys.stderr)
    (log_dir / "last-run.log").write_text(log)
    with (log_dir / "decisions.jsonl").open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    if code == 0 and inactive and not args.allow_inactive:
        return INACTIVE_EXIT
    return code

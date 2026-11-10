"""Start, stop and inspect a resident kit process.

    python -m foldboost serve start --kit genie3 --kit-dir /kit/genie3 --socket ~/.fp/g3.sock \
        --python /opt/venv/bin/python -- design --config h100 --mode exact --input warm.yaml --out_dir warm-out ...
    python -m foldboost serve status --socket ~/.fp/g3.sock
    python -m foldboost serve stop   --socket ~/.fp/g3.sock

The arguments after `--` are the kit's own first request: it fixes the process configuration (mode, GPU config,
precision flags) and doubles as the warm-up. A detached supervisor restarts the process when it reaches its request
limit (clients meanwhile get refusals and run fresh CLIs) and exits when `stop` is called.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from . import serve_wrapper

_DETACHED: list = []  # keep the supervisor's Popen alive so it is not reported as an orphan


def _rpc(path: str, message: dict, timeout: float = 5.0) -> dict | None:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(path)
            s.settimeout(None)
            with s.makefile("rwb") as stream:
                stream.write((json.dumps(message) + "\n").encode())
                stream.flush()
                line = stream.readline()
                return json.loads(line) if line else None
    except (OSError, ValueError):
        return None


def status(path: str) -> dict | None:
    return _rpc(path, {"ping": True})


def stop(path: str) -> bool:
    control = Path(path + ".stop")
    control.write_text("stop\n")
    reply = _rpc(path, {"quit": True}, timeout=10)
    pid_file = Path(path + ".pid")
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text()), signal.SIGTERM)
        except (OSError, ValueError):
            pass
    return reply is not None


def supervisor(kit_dir: Path, socket_path: str, kit: str, python_dir: str | None, first_argv: list, max_requests: int, log: Path) -> None:
    """Detached loop: run the derived launcher; restart after it exits unless the stop file exists."""
    wrapper = kit_dir / ".fp_run_serve.sh"
    env = dict(os.environ, FP_SERVE_SOCKET=socket_path, FP_SERVE_COMPAT=kit, FP_SERVE_MAX_REQUESTS=str(max_requests),
               PYTHONPATH=os.pathsep.join(filter(None, [str(Path(__file__).resolve().parents[1]), os.environ.get("PYTHONPATH")])))
    if python_dir:
        env["PATH"] = python_dir + os.pathsep + env["PATH"]
    stop_file = Path(socket_path + ".stop")
    while not stop_file.exists():
        with open(log, "ab") as out:
            proc = subprocess.Popen(["bash", str(wrapper), *first_argv], cwd=kit_dir, env=env, stdout=out, stderr=subprocess.STDOUT)
            Path(socket_path + ".pid").write_text(str(proc.pid))
            proc.wait()
        time.sleep(1)


def start(kit: str, kit_dir: Path, socket_path: str, first_argv: list, python: str | None = None, max_requests: int = 500, wait: float = 900) -> int:
    socket_path = os.path.abspath(os.path.expanduser(socket_path))
    Path(socket_path).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    for leftover in (socket_path + ".stop", socket_path):
        if os.path.exists(leftover):
            if leftover == socket_path and status(socket_path):
                print(f"already serving on {socket_path}", file=sys.stderr)
                return 0
            os.unlink(leftover)
    text, pkg = serve_wrapper.derive((kit_dir / "run.sh").read_text())
    wrapper = kit_dir / ".fp_run_serve.sh"
    wrapper.write_text(text)
    wrapper.chmod(0o755)
    log = Path(socket_path + ".log")
    code = ("import sys,json; sys.path.insert(0, %r); from foldboost.serve_ctl import supervisor; from pathlib import Path; "
            "supervisor(Path(%r), %r, %r, %r, json.loads(sys.argv[1]), %d, Path(%r))") % (
        str(Path(__file__).resolve().parents[1]), str(kit_dir), socket_path, kit, str(Path(python).parent) if python else None, max_requests, str(log))
    _DETACHED.append(subprocess.Popen([sys.executable, "-c", code, json.dumps(first_argv)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL, start_new_session=True))
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait:
        if status(socket_path):
            print(f"serving {kit} on {socket_path} (log {log})", file=sys.stderr)
            return 0
        time.sleep(1)
    print(f"server did not come up within {wait:.0f}s; see {log}", file=sys.stderr)
    return 1

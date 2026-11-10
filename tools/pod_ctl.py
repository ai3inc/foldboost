"""Minimal RunPod controller for the FoldBoost holdout runs.

Touches only pods whose name starts with ``fp-hybrid-``; never lists or modifies anything else.
The API key is read from the research key file (work/runpod/.api-key, else ~/.runpod/api_key; override with FP_RUNPOD_KEY_FILE) and travels to curl on stdin, never argv or logs.

    pod_ctl.py create NAME            # H100 80GB HBM3, SECURE cloud, waits until SSH answers
    pod_ctl.py status NAME
    pod_ctl.py ssh NAME [--timeout S] -- COMMAND...       (stdout streams; exit status is the remote status)
    pod_ctl.py put NAME LOCAL REMOTE
    pod_ctl.py get NAME REMOTE LOCAL
    pod_ctl.py delete NAME
    pod_ctl.py list                    # fp-hybrid-* pods only
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

PREFIX = "fp-hybrid-"
BASE = "https://rest.runpod.io/v1"
HOME = pathlib.Path.home()
_KEY_FILES = [pathlib.Path(os.environ["FP_RUNPOD_KEY_FILE"])] if os.environ.get("FP_RUNPOD_KEY_FILE") else [
    pathlib.Path(__file__).resolve().parents[3] / "work" / "runpod" / ".api-key", HOME / ".runpod" / "api_key"]
KEY = next(f for f in _KEY_FILES if f.is_file()).read_text().strip()
SSH_KEY = HOME / ".ssh" / "runpod_kpf"
STATE = pathlib.Path(os.environ.get("FP_POD_STATE", pathlib.Path(__file__).resolve().parents[3] / "work" / "holdout" / "pods"))
IMAGE = "nvcr.io/nvidia/cuda@sha256:46cb48a4abfbc40c836fe57bc05a07101b6458fffc63bbdfd6a50db98c9358bd"  # CUDA 12.6.3 runtime, as in the discovery runs


def api(method, path, data=None):
    cmd = ["curl", "--silent", "--show-error", "--fail-with-body", "--connect-timeout", "15", "--max-time", "60", "--request", method, "--header", "@-", BASE + path]
    if data is not None:
        cmd += ["--data-binary", json.dumps(data)]
    headers = f"Authorization: Bearer {KEY}\nContent-Type: application/json\nUser-Agent: foldboost/1.0\n"
    p = subprocess.run(cmd, input=headers, capture_output=True, text=True, timeout=70)
    if p.returncode:
        raise RuntimeError(("curl " + str(p.returncode) + " " + p.stderr + p.stdout)[:600].replace(KEY, "[REDACTED]"))
    return json.loads(p.stdout) if p.stdout.strip() else None


def own(name):
    if not name.startswith(PREFIX):
        raise SystemExit(f"refusing: pod names must start with {PREFIX}")


def receipt(name):
    return STATE / f"{name}.json"


def endpoint(name):
    r = json.loads(receipt(name).read_text())
    pod = api("GET", "/pods/" + r["id"])
    if pod["name"] != name:
        raise SystemExit("pod id/name mismatch")
    ip, port = pod.get("publicIp"), (pod.get("portMappings") or {}).get("22")
    if not (ip and port):
        raise SystemExit("pod has no SSH endpoint yet")
    return ip, port, pod


def ssh_base(host, port):
    return ["-i", str(SSH_KEY), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=20",
            "-o", "ServerAliveCountMax=6", "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={STATE / 'known_hosts'}", "-p", str(port)]


def cmd_create(a):
    own(a.name)
    STATE.mkdir(parents=True, exist_ok=True)
    pods = api("GET", "/pods")
    if any(p["name"] == a.name for p in pods):
        raise SystemExit("a pod with this name already exists")
    mine = [p for p in pods if p["name"].startswith(PREFIX)]
    if len(mine) >= a.max_pods:
        raise SystemExit(f"{len(mine)} fp-hybrid pods already allocated (limit {a.max_pods})")
    pub = (HOME / ".ssh" / "runpod_kpf.pub").read_text().strip()
    startup = ("set -e; mkdir -p /workspace /results /run/sshd /root/.ssh; chmod 700 /root/.ssh; printf '%s\\n' \"$PUBLIC_KEY\" > /root/.ssh/authorized_keys; "
               "chmod 600 /root/.ssh/authorized_keys; apt-get update; DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openssh-server python3 ca-certificates curl; "
               "ssh-keygen -A; exec /usr/sbin/sshd -D -e -o PasswordAuthentication=no -o PermitRootLogin=prohibit-password")
    payload = {"name": a.name, "cloudType": "SECURE", "computeType": "GPU", "gpuCount": 1, "gpuTypeIds": ["NVIDIA H100 80GB HBM3"], "imageName": IMAGE,
               "containerDiskInGb": a.disk, "volumeInGb": 0, "minRAMPerGPU": 48, "minVCPUPerGPU": 4, "ports": ["22/tcp"], "dockerEntrypoint": ["/bin/bash", "-lc"],
               "dockerStartCmd": [startup], "env": {"PUBLIC_KEY": pub}, "interruptible": False}
    pod = api("POST", "/pods", payload)
    receipt(a.name).write_text(json.dumps({"id": pod["id"], "name": a.name, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "costPerHr": pod.get("costPerHr")}, indent=1))
    print("created", pod["id"], "cost/hr", pod.get("costPerHr"), flush=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < a.wait:
        p = api("GET", "/pods/" + pod["id"])
        if p.get("desiredStatus") == "EXITED":
            raise SystemExit("pod exited before SSH: " + str(p.get("lastStatusChange")))
        if p.get("publicIp") and (p.get("portMappings") or {}).get("22"):
            host, port = p["publicIp"], p["portMappings"]["22"]
            r = subprocess.run(["ssh", *ssh_base(host, port), f"root@{host}", "echo SSH_READY"], capture_output=True, text=True, timeout=40)
            if "SSH_READY" in r.stdout:
                gpu = subprocess.run(["ssh", *ssh_base(host, port), f"root@{host}", "nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader"], capture_output=True, text=True, timeout=40).stdout.strip()
                print("ready", host, port, gpu, flush=True)
                return
        time.sleep(15)
    raise SystemExit("pod not reachable in time; it is still allocated: run delete if unwanted")


def cmd_status(a):
    own(a.name)
    host, port, pod = endpoint(a.name)
    print(json.dumps({"id": pod["id"], "status": pod.get("desiredStatus"), "host": host, "port": port, "costPerHr": pod.get("costPerHr")}))


def cmd_ssh(a):
    own(a.name)
    host, port, _ = endpoint(a.name)
    command = " ".join(a.command)
    p = subprocess.run(["ssh", *ssh_base(host, port), f"root@{host}", command], timeout=a.timeout)
    sys.exit(p.returncode)


def scp(a, up):
    own(a.name)
    host, port, _ = endpoint(a.name)
    base = ["scp", "-O", "-r", "-i", str(SSH_KEY), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={STATE / 'known_hosts'}", "-P", str(port)]
    src, dst = (a.src, f"root@{host}:{a.dst}") if up else (f"root@{host}:{a.src}", a.dst)
    sys.exit(subprocess.run([*base, src, dst], timeout=a.timeout).returncode)


def cmd_delete(a):
    own(a.name)
    r = json.loads(receipt(a.name).read_text())
    pod = api("GET", "/pods/" + r["id"])
    if pod["name"] != a.name:
        raise SystemExit("pod id/name mismatch; not deleting")
    api("DELETE", "/pods/" + r["id"])
    r["deleted"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    receipt(a.name).write_text(json.dumps(r, indent=1))
    print("deleted", a.name)


def cmd_list(a):
    for p in api("GET", "/pods"):
        if p["name"].startswith(PREFIX):
            print(p["id"], p["name"], p.get("desiredStatus"), p.get("costPerHr"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create"); c.add_argument("name"); c.add_argument("--disk", type=int, default=150); c.add_argument("--wait", type=int, default=900); c.add_argument("--max-pods", type=int, default=20); c.set_defaults(f=cmd_create)
    s = sub.add_parser("status"); s.add_argument("name"); s.set_defaults(f=cmd_status)
    h = sub.add_parser("ssh"); h.add_argument("name"); h.add_argument("--timeout", type=int, default=7200); h.add_argument("command", nargs=argparse.REMAINDER); h.set_defaults(f=cmd_ssh)
    for nm, up in (("put", True), ("get", False)):
        u = sub.add_parser(nm); u.add_argument("name"); u.add_argument("src"); u.add_argument("dst"); u.add_argument("--timeout", type=int, default=3600); u.set_defaults(f=lambda a, up=up: scp(a, up))
    d = sub.add_parser("delete"); d.add_argument("name"); d.set_defaults(f=cmd_delete)
    l = sub.add_parser("list"); l.set_defaults(f=cmd_list)
    a = ap.parse_args()
    if a.cmd == "ssh" and a.command and a.command[0] == "--":
        a.command = a.command[1:]
    a.f(a)


if __name__ == "__main__":
    main()

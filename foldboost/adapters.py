"""Vendored candidate adapters: integrity check, single-adapter isolation, interpreter gate.

Each adapter directory holds the byte-identical files that were validated in the H100 benchmark
bundles. A run exposes exactly one adapter directory to the kit's Python through
FLASHP_HYBRID_ADAPTER_DIR; the interpreter gate (a one-line .pth) imports that directory's
``flashpair_bootstrap`` and nothing else, so two adapters whose modules share a name can never
collide. The adapter's own enable flag is 1 only when our code was selected; for every other
provider the flag is 0 and the bootstrap stays inert, exactly as in the measured arms.
"""
from __future__ import annotations

import hashlib
import json
import site
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
GATE_NAME = "000_foldboost_gate.pth"
GATE_LINE = ("import os,sys;d=os.environ.get('FLASHP_HYBRID_ADAPTER_DIR');"
             "d and (sys.path.insert(0,d),__import__('flashpair_bootstrap'))\n")


class AdapterError(RuntimeError):
    pass


def registry() -> dict:
    return json.loads((DATA / "adapters.json").read_text())


def adapter_dir(adapter_id: str) -> Path:
    return DATA / "adapters" / adapter_id


def verify_adapter(adapter_id: str) -> None:
    info = registry().get(adapter_id)
    if info is None:
        raise AdapterError(f"unknown adapter {adapter_id!r}")
    base = adapter_dir(adapter_id)
    present = sorted(p.name for p in base.iterdir() if p.is_file() and p.suffix == ".py")
    if present != sorted(info["files"]):
        raise AdapterError(f"{adapter_id}: files {present} differ from registry {sorted(info['files'])}")
    for name, expected in info["files"].items():
        actual = hashlib.sha256((base / name).read_bytes()).hexdigest()
        if actual != expected:
            raise AdapterError(f"{adapter_id}/{name}: sha256 {actual} != validated {expected}")


def adapter_environment(adapter_id: str, enabled: bool, extra: dict | None = None) -> dict:
    verify_adapter(adapter_id)
    env = {"FLASHP_HYBRID_ADAPTER_DIR": str(adapter_dir(adapter_id)), registry()[adapter_id]["env_flag"]: "1" if enabled else "0"}
    env.update(extra or {})
    return env


def install_gate(site_packages: Path | None = None, dry_run: bool = False) -> Path:
    target = (site_packages or Path(site.getsitepackages()[0])) / GATE_NAME
    if not dry_run:
        target.write_text(GATE_LINE)
    return target


def gate_installed(site_packages: Path | None = None) -> bool:
    target = (site_packages or Path(site.getsitepackages()[0])) / GATE_NAME
    return target.is_file() and target.read_text() == GATE_LINE


if __name__ == "__main__":
    for adapter in registry():
        verify_adapter(adapter)
    print("adapters verified", len(registry()), file=sys.stderr)

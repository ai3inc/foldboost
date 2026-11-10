"""Copy the validated candidate adapters byte-for-byte into foldboost/data/adapters.

Every copied file must match the sha256 recorded in the originating bundle's manifest.json
(the manifest the remote entrypoint verified before each benchmark). Nothing is edited.

    python tools/vendor_adapters.py --work /path/to/work/three-engine-72
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

# adapter id -> (bundle, runtime files, enabling env flag, kit(s), experiments that validated it)
ADAPTERS = {
    "ob0-inline-021": ("bundle-022", ["flashpair_bootstrap.py", "inline_loader.py"], "FLASHP_INLINE_WORKER", ["openfold3_ob0"], ["022"]),
    "of3-inline-023": ("bundle-027", ["flashpair_bootstrap.py", "inline_loader.py"], "FLASHP_INLINE_WORKER", ["openfold3"], ["027", "030"]),
    "boltz-mmap-031": ("bundle-031", ["flashpair_bootstrap.py", "mmap_loader.py"], "FLASHP_BOLTZ_MMAP", ["boltz2"], ["031"]),
    "mpnn-triton-key-036": ("bundle-036", ["flashpair_bootstrap.py", "cached_key.py"], "FLASHP_MPNN036", ["proteinmpnn"], ["036"]),
    "atlas-mmap-037": ("bundle-037", ["flashpair_bootstrap.py", "mmap_loader.py"], "FLASHP_ATLAS_MMAP", ["atlasfold"], ["037"]),
    "esmif1-mmap-039": ("bundle-039", ["flashpair_bootstrap.py", "mmap_loader.py"], "FLASHP_ESMIF1_MMAP", ["esm_if1"], ["039"]),
    "ptx-cif-040": ("bundle-040", ["flashpair_bootstrap.py", "cif_index.py"], "FLASHP_PTX_CIF", ["protenix_v1"], ["040"]),
    "genie3-dispatch-050": ("bundle-050", ["flashpair_bootstrap.py", "dispatch_loader.py"], "FLASHP_GENIE3_DISPATCH", ["genie3"], ["050"]),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "foldboost" / "data" / "adapters")
    args = ap.parse_args()
    registry = {}
    for adapter_id, (bundle, files, flag, kits, experiments) in ADAPTERS.items():
        src = args.work / bundle
        manifest = json.loads((src / "manifest.json").read_text())
        dest = args.out / adapter_id
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        pinned = {}
        for name in files:
            digest = sha(src / name)
            if manifest[name] != digest:
                raise SystemExit(f"{bundle}/{name}: bundle manifest says {manifest[name]}, file is {digest}")
            shutil.copyfile(src / name, dest / name)
            pinned[name] = digest
        registry[adapter_id] = {"source_bundle": bundle, "files": pinned, "env_flag": flag, "kits": kits, "validated_by": experiments}
    (args.out.parent / "adapters.json").write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(f"vendored {len(registry)} adapters")


if __name__ == "__main__":
    main()

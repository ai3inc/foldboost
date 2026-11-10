"""Input size per kit: the one number that defines the 'hull of tested inputs' a family scope may cover.

Sizes are tokens or residues read directly from the input file; a file that cannot be read yields None and
such a request is never covered by a family scope (it falls back to Anthropic).
"""
from __future__ import annotations

import json
import re
from pathlib import Path


def _lines(path):
    return Path(path).read_text(errors="replace").splitlines()


def genie3_length(yaml_path) -> int | None:
    """Unconditional design length: min_length must equal max_length."""
    text = Path(yaml_path).read_text(errors="replace")
    lo = re.search(r"^\s*min_length:\s*(\d+)\s*$", text, re.M)
    hi = re.search(r"^\s*max_length:\s*(\d+)\s*$", text, re.M)
    return int(lo.group(1)) if lo and hi and lo.group(1) == hi.group(1) else None


def openfold3_tokens(query_json) -> int | None:
    """Polymer tokens: sum over protein/RNA/DNA chain entries of len(sequence) x number of chain ids."""
    try:
        data = json.loads(Path(query_json).read_text())
        total = 0
        for query in data["queries"].values():
            for chain in query["chains"]:
                if chain.get("molecule_type") in ("protein", "rna", "dna"):
                    ids = chain["chain_ids"]
                    total += len(chain["sequence"]) * (len(ids) if isinstance(ids, list) else 1)
        return total or None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def fasta_residues(path) -> int | None:
    seq = sum(len(line.strip()) for line in _lines(path) if line and not line.startswith(">"))
    return seq or None


def pdb_ca_count(path, chain: str | None = None) -> int | None:
    n = 0
    for line in _lines(path):
        if line.startswith("ATOM") and line[12:16].strip() == "CA" and (chain is None or line[21:22] == chain):
            n += 1
    return n or None


def pdb_dir_residues(directory) -> int | None:
    d = Path(directory)
    if not d.is_dir():
        return None
    total = sum(pdb_ca_count(p) or 0 for p in sorted(d.glob("*.pdb")))
    return total or None


def protenix_tokens(json_path) -> int | None:
    try:
        data = json.loads(Path(json_path).read_text())
        data = data[0] if isinstance(data, list) else data
        total = 0
        for entity in data["sequences"]:
            for kind in ("proteinChain", "dnaSequence", "rnaSequence"):
                if kind in entity:
                    total += len(entity[kind]["sequence"]) * int(entity[kind].get("count", 1))
        return total or None
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None


def yaml_sequence_residues(path) -> int | None:
    """Boltz-style input: total length of every `sequence:` value."""
    total = sum(len(m.group(1)) for m in re.finditer(r"^\s*sequence:\s*([A-Za-z]+)\s*$", Path(path).read_text(errors="replace"), re.M))
    return total or None


# kit -> (function, which request field it reads). "input" is the primary input path of the parsed request.
MEASURES = {
    "genie3": "genie3_length",
    "openfold3": "openfold3_tokens",
    "openfold3_ob0": "openfold3_tokens",
    "atlasfold": "fasta_residues",
    "esm_if1": "pdb_ca_count",
    "proteinmpnn": "pdb_dir_residues",
    "protenix_v1": "protenix_tokens",
    "boltz2": "yaml_sequence_residues",
}


def measure(kit: str, path: str, options: dict | None = None) -> int | None:
    name = MEASURES.get(kit)
    if name is None:
        return None
    try:
        if kit == "esm_if1":
            return pdb_ca_count(path, (options or {}).get("chain"))
        return globals()[name](path)
    except (OSError, ValueError):
        return None

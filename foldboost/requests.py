"""Turn a kit command line (``bash run.sh <sub> ... --mode M ...``) into a comparable request.

The evidence scopes in the policy were produced by running exactly these command shapes, so the
same parser is used both when the policy is built from recorded benchmark commands and when a
user request is routed. A flag the parser does not know ends up in ``options`` and therefore
cannot match any evidence scope: unknown means out of scope, never "probably the same".
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

# Per kit: the sub-command, which flags carry input files/directories, flags that never take a
# value, flags that only say where results go (ignored for matching) and the flags that are
# understood. Understood flags are matched by value; anything else is reported as unrecognised.
KITS: dict[str, dict] = {
    "genie3": dict(sub="design", files=["input"], bools=[], ignore=["out_dir"],
                   known=["seed", "n", "batch_size", "det"]),
    "proteinmpnn": dict(sub="design", files=["input"], bools=[], ignore=["out"],
                        known=["seed", "num_seq_per_target", "batch_size", "save_score", "save_probs"]),
    "atlasfold": dict(sub="pred", files=["input_fasta"], bools=["save_confidence_arrays", "save_distogram"],
                      ignore=["out_dir"], known=["det", "seed"]),
    "boltz2": dict(sub="pred", files=["input"], bools=["write_full_pae", "write_full_pde"],
                   ignore=["out_dir"], known=["seeds", "seed"]),
    "esm_if1": dict(sub="design", files=[], bools=[], ignore=["outpath"],
                    known=["chain", "num_samples", "seed", "batch_size"], positional_input=True),
    "protenix_v1": dict(sub="pred", files=["input"], bools=[], ignore=["out_dir"],
                        known=["det", "seeds", "sample", "cycle", "step", "use_msa", "use_template"]),
    "openfold3": dict(sub="pred", files=["query_json", "runner_yaml"], bools=[], ignore=["output_dir"],
                      known=["det", "use_msa_server", "use_templates", "num_diffusion_samples"]),
    "openfold3_ob0": dict(sub="pred", files=["query_json", "runner_yaml"], bools=[], ignore=["output_dir"],
                          known=["det", "use_msa_server", "use_templates", "num_diffusion_samples"]),
}
# The first listed file flag is the "primary" input; further files become <flag>_sha256 options.
COMMON_IGNORED = ("config", "mode")


@dataclass
class Request:
    kit: str
    sub: str
    config: str
    mode: str | None
    input_sha256: str | None
    options: dict
    unrecognized: list = field(default_factory=list)
    problems: list = field(default_factory=list)
    size: int | None = None            # tokens/residues of the primary input (see sizes.py); None when unreadable
    extra: dict = field(default_factory=dict)  # facts read from the inputs that exact-scope matching ignores (family scopes use them)

    def match_view(self) -> dict:
        return {"input_sha256": self.input_sha256, "options": self.options}


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def directory_digest(entries: dict[str, str]) -> str:
    """Digest of a flat directory: sha256 over sorted ``name:sha256`` lines."""
    return hashlib.sha256("".join(f"{name}:{entries[name]}\n" for name in sorted(entries)).encode()).hexdigest()


def hash_path(path: str | Path) -> str:
    p = Path(path)
    if p.is_dir():
        return directory_digest({q.name: file_sha256(q) for q in sorted(p.iterdir()) if q.is_file()})
    return file_sha256(p)


def _norm(value: str):
    return int(value) if value.lstrip("-").isdigit() else value


def parse_request(kit: str, argv: Iterable[str], hasher: Callable[[str], str] = hash_path) -> Request:
    if kit not in KITS:
        return Request(kit, "", "", None, None, {}, problems=[f"no request parser for kit {kit!r}"])
    spec = KITS[kit]
    toks = list(argv)
    if toks and toks[0] == "bash":
        toks = toks[1:]
    if toks and toks[0].endswith("run.sh"):
        toks = toks[1:]
    problems: list = []
    if not toks:
        return Request(kit, "", "", None, None, {}, problems=["empty command"])
    sub = toks.pop(0)
    if sub != spec["sub"]:
        problems.append(f"sub-command {sub!r} is not {spec['sub']!r}")
    raw: dict = {}
    positionals: list = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == "--":
            i += 1
            continue
        if t.startswith("--"):
            name = t[2:].replace("-", "_")
            if name in spec["bools"]:
                raw[name] = True
            elif i + 1 < len(toks):
                raw[name] = toks[i + 1]
                i += 1
            else:
                problems.append(f"flag {t} has no value")
        else:
            positionals.append(t)
        i += 1
    config = str(raw.pop("config", ""))
    mode = raw.pop("mode", None)
    for name in spec["ignore"]:
        raw.pop(name, None)
    files = list(spec["files"])
    sha_of: dict = {}
    positionals_input = None
    if spec.get("positional_input"):
        if positionals:
            positionals_input = positionals.pop(0)
            sha_of["input"] = _safe_hash(hasher, positionals_input, problems)
        else:
            problems.append("input file missing")
        files = []
    paths = {}
    for name in files:
        if name in raw:
            paths[name] = str(raw.pop(name))
            sha_of[name] = _safe_hash(hasher, paths[name], problems)
        else:
            problems.append(f"--{name.replace('_', '-')} missing")
    primary_flag = files[0] if files else "input"
    primary_path = positionals_input if spec.get("positional_input") else paths.get(primary_flag)
    primary = sha_of.pop(primary_flag, None)
    options: dict = {f"{k}_sha256": v for k, v in sha_of.items()}
    unrecognized = []
    for key, value in raw.items():
        if key not in spec["known"] and key not in spec["bools"]:
            unrecognized.append(key)
        options[key] = _norm(value) if isinstance(value, str) else value
    if positionals:
        options["positional"] = positionals
    extra = {}
    if kit in ("openfold3", "openfold3_ob0") and paths.get("runner_yaml"):
        extra["runner_yaml_template_seed"] = _template_seed(paths["runner_yaml"])
    from . import sizes
    size = sizes.measure(kit, primary_path, options) if primary_path else None
    return Request(kit, sub, config, mode, primary, dict(sorted(options.items())), unrecognized, problems, size, extra)


def _template_seed(path: str):
    """Seed of a runner yaml that is exactly the one-seed template the benchmarks used, else None (anything else is not vouched for)."""
    import re
    try:
        m = re.fullmatch(r"experiment_settings:\n  seeds: \[(\d+)\]\n+", Path(path).read_text())
    except OSError:
        return None
    return int(m.group(1)) if m else None


def _safe_hash(hasher: Callable[[str], str], path: str, problems: list) -> str | None:
    try:
        return hasher(path)
    except OSError as error:
        problems.append(f"cannot read input {path!r}: {error.strerror or error}")
        return None


def with_mode(argv: Iterable[str], mode: str) -> list:
    toks = list(argv)
    for i, t in enumerate(toks[:-1]):
        if t == "--mode":
            toks[i + 1] = mode
            return toks
    raise ValueError("command has no --mode")

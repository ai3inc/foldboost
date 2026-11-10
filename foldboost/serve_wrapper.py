"""Derive a resident-process launcher from a kit's own run.sh without touching the kit.

Every kit's run.sh prepares the environment (config file, caches, pin checks) and ends with
``exec python -m <kit>_opt <verb> ...``. The derived script is that file with only the final exec
replaced by the resident runner, so the environment preparation is byte-for-byte the kit's.

    python -m foldboost.serve_wrapper KIT_DIR [--out KIT_DIR/.fp_run_serve.sh]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# The exec may sit inside a case branch on one line (atlasfold: `  check|warm|pred) exec python3 -m atlasfold_opt "$verb" "$@" ;;`).
# The interpreter is `python`/`python3` or a shell variable the kit set (`exec "$PY" -m progen2_opt ...`; the kit's own choice of
# interpreter is kept in that case).
EXEC = re.compile(r"^(?P<indent>\s*)(?P<case>[A-Za-z0-9_|*-]+\)\s*)?exec\s+(?P<py>python3?|\"?\$\{?[A-Za-z_][A-Za-z0-9_]*\}?\"?)\s+-m\s+(?P<pkg>[A-Za-z0-9_]+)(?P<rest>\s.*)$")
# A bare word right after the package (`... -m enformer_opt check "$@"`) means the exec runs one fixed verb: not a launcher for requests.
FIXED_VERB = re.compile(r"^\s+[A-Za-z][A-Za-z0-9_-]*(\s|$)")


ENTRY_OVERRIDES = {"atlasfold_opt": "atlasfold_opt.cli:main"}  # packages whose __main__ calls main() at import


def derive(run_sh: str, entry: str | None = None) -> tuple[str, str]:
    lines = run_sh.splitlines(keepends=True)
    hits = [(i, EXEC.match(line)) for i, line in enumerate(lines) if EXEC.match(line)]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one `exec python -m <pkg>` line, found {len(hits)}")
    i, m = hits[0]
    pkg = m["pkg"]
    rest = m["rest"]
    if FIXED_VERB.match(rest):
        raise SystemExit(f"the only `exec -m {pkg}` runs a fixed verb ({rest.split()[0]!r}); it is not a request launcher")
    py = "python" if m["py"] in ("python", "python3") else m["py"]
    # `exec python -m pkg "$CMD" ...` becomes `exec python -m foldboost.serve_kit --entry pkg.__main__:main -- "$CMD" ...`
    # `entry` overrides the default when the package's __main__ runs main() at import (atlasfold_opt: use atlasfold_opt.cli:main).
    lines[i] = f'{m["indent"]}{m["case"] or ""}exec {py} -m foldboost.serve_kit --entry {entry or ENTRY_OVERRIDES.get(pkg) or pkg + ".__main__:main"} --{rest}\n'
    return "".join(lines), pkg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kit_dir", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--entry", help="module:function overriding <pkg>.__main__:main (needed when __main__ calls main() at import)")
    a = ap.parse_args()
    text, pkg = derive((a.kit_dir / "run.sh").read_text(), a.entry)
    out = a.out or a.kit_dir / ".fp_run_serve.sh"
    out.write_text(text)
    out.chmod(0o755)
    print(f"{out} (entry {a.entry or pkg + '.__main__:main'})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

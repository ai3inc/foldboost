"""Resident-process runner for kits whose CLI is a Python function.

A fresh kit CLI pays Python imports, lever installation and CUDA initialisation on every call. This
runner pays them once and then calls the kit's own entry function again for each request, in the same
process. It optimises nothing in the model: a request is executed by exactly the code the CLI runs, so
its outputs must equal a fresh CLI's, and that equality is what the validation checks.

Server (inside the kit's environment, started through the wrapper that serve_wrapper derives from run.sh):

    python -m foldboost.serve_kit --entry openfold3_opt.__main__:main --compat openfold3 \
        --socket /run/user/1000/fp/of3.sock -- pred --mode exact --det 1 ...        # first request (optional)

Client:

    python -m foldboost.serve_kit client --socket PATH --log run.log -- pred --mode exact ...

Without --socket the server reads one JSON array (a request's argv) per line from stdin (prototype/tests).

Safety rules:
* The first request fixes the process configuration. A later request must produce the same compatibility
  key (every flag except the per-request ones, plus the kit's own gate classes, see COMPAT); otherwise it is
  refused with rc 75 and nothing runs. The client then runs a fresh CLI instead.
* One request at a time; the socket lives in an owner-only directory with mode 0600.
* A request's stdout/stderr go to its own log file (file descriptors 1 and 2 are redirected for its duration).
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import socket
import stat
import sys
import time

REFUSED = 75  # EX_TEMPFAIL: the resident configuration cannot serve this request; run a fresh CLI

# Flags that may differ between requests of one resident process. Everything else must match the first request.
VARY = {
    "openfold3": {"--query-json", "--output-dir", "--runner-yaml"},
    "openfold3_ob0": {"--query-json", "--output-dir", "--runner-yaml"},
    # esm_if1 takes the structure as a positional argument (flag key ""), plus --outpath and the per-input --seed
    "esm_if1": {"", "--input", "--out", "--outpath", "--seed"},
    "default": {"", "--input", "--out", "--out_dir", "--output-dir", "--outpath", "--seed", "--seeds", "--query-json", "--runner-yaml", "--input-fasta", "--out-dir",
                "--pdbdir", "--starting-pdb", "--parent-path", "--mutants-path", "--output-path"},
}


def flag_items(argv: list) -> tuple:
    items, i = [], 0
    while i < len(argv):
        t = argv[i]
        if t.startswith("--"):
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                items.append((t, argv[i + 1]))
                i += 1
            else:
                items.append((t, None))
        else:
            items.append(("", t))
        i += 1
    return tuple(items)


def compat_key(kit: str, argv: list) -> tuple:
    vary = VARY.get(kit, VARY["default"])
    key = tuple(sorted((f, v) for f, v in flag_items(argv) if f not in vary))
    extra = ()
    if kit in ("openfold3", "openfold3_ob0"):
        extra = _openfold3_gate_class(argv)
    return key + (("__gate__", extra),)


def _openfold3_gate_class(argv: list) -> tuple:
    """enable() judges the CUDA-graph size gate once, from the first request's token count and reuses it afterwards
    ("idempotent: repeated calls return the first report"). A request on the other side of the gate must not reuse it."""
    try:
        from openfold3_opt import cli, modes  # type: ignore
        a = cli.build_parser().parse_args(list(argv))
        mode = cli.resolve_mode(a.mode)
        lv, el = cli.route_options(a, mode)
        n_tokens = cli.query_tokens(a.query_json)
        return (mode, lv, el, bool(modes.graphed_call(mode, el, n_tokens)))
    except Exception as error:  # an unreadable request is judged by the kit itself on a fresh CLI
        return ("unclassified", repr(error)[:80])


def resolve(entry: str):
    module, _, name = entry.partition(":")
    return getattr(importlib.import_module(module), name)


@contextlib.contextmanager
def redirected(path: str | None):
    """Send file descriptors 1 and 2 to `path` for the duration (C-level writers included), and point Python's own
    sys.stdout/sys.stderr at the same file. The latter matters when sys.stdout is not backed by file descriptor 1
    (an embedding host, pytest's capture): print() would otherwise bypass the log."""
    if not path:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    py_out, py_err = sys.stdout, sys.stderr
    # a dup of the same open file description shares its offset, so Python-level and C-level writes interleave in order
    stream = os.fdopen(os.dup(fd), "w", buffering=1, encoding="utf-8", errors="replace")
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        sys.stdout = sys.stderr = stream
        yield
    finally:
        try:
            stream.flush()
        finally:
            sys.stdout, sys.stderr = py_out, py_err
            stream.close()
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for d in (*saved, fd):
            os.close(d)


def reserved_mib() -> int | None:
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        return int(torch.cuda.memory_reserved() // (1 << 20))
    return None


def release_memory() -> None:
    """Between requests: collect garbage and hand cached-but-unused CUDA blocks back (graph pools stay with their graphs).
    Added after an H100 resident run showed GPU memory creeping up with request count."""
    import gc
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


# Kits whose `of3_transition` lever keeps per-module packed weights in a module-level dict keyed by id(module) (identical code in both packages).
LEAKY_TRANSITION_PACKAGES = {"openfold3": "openfold3_opt", "openfold3_ob0": "openfold3_ob0_opt"}


def install_leak_guards(kit: str) -> dict:
    """Make caches of a kit's levers that outlive the model they were built for die with that model.

    A fresh CLI builds one model and exits, so a lever may keep per-module state in a module-level dict keyed by ``id(module)``. A resident process builds a new
    model for every request, so such a dict grows by one entry per module per request and never shrinks. Measured on H100, OpenFold3 ``exact``: the kit's
    ``of3_transition._PACKS`` kept 59 packed-weight sets (about 90 MiB of GPU memory) per request, +97% GPU memory over 20 requests; upstream OpenFold3 called
    in a loop stayed flat. The B300 run was flat only because the provider has no tuned cell there and declines before packing.

    The guard wraps the lever's own ``_packed`` and registers ``weakref.finalize(module, _PACKS.pop, id(module))`` once per module: a live module keeps its
    pack, a dead one loses it, and new modules pack exactly as before, so no output can change (30 of 30 outputs byte-identical, GPU memory +0.27% over the same
    20 requests). The kit's file is not edited; where the lever is absent (another kit, another mode, another version) this does nothing.
    The ``fast`` mode's other lever buffers (pair cache, trunk graph, trimul buffers, ...) are not covered; the growth limit below still recycles the process.
    """
    import importlib
    import weakref

    package = LEAKY_TRANSITION_PACKAGES.get(kit)
    if not package:
        return {}
    try:
        T = importlib.import_module(package + ".of3_transition")
    except Exception as error:  # the kit is not importable here: nothing to guard
        return {"of3_transition": f"absent:{type(error).__name__}"}
    packed, packs = getattr(T, "_packed", None), getattr(T, "_PACKS", None)
    if packed is None or packs is None:
        return {"of3_transition": "layout-changed"}
    if getattr(packed, "_fp_finalizer", False):
        return {"of3_transition": "already"}

    def guarded(module, device):
        pack = packed(module, device)
        key = id(module)
        if key in packs and not module.__dict__.get("_fp_pack_finalizer"):
            weakref.finalize(module, packs.pop, key, None)   # runs when the module dies, before its id can be reused
            module.__dict__["_fp_pack_finalizer"] = True
        return pack

    guarded._fp_finalizer = True
    T._packed = guarded
    return {"of3_transition": "guarded"}


# The `fast` lever stack of the OpenFold3 kits (CUDA-graph generations, pair cache, trunk graphs, memoised rollout tensors, atom invariants, trimul native packs) keeps per-model
# state alive after a request. A fresh CLI exits and never notices; a resident process builds a model per request, so the state piles up: +~625 MiB of GPU memory per request on
# H100 (fork L study). The kit documents most of these as its "item-boundary release" hooks (its post_release lever calls them at large token counts); the trimul native serve
# cache is the exception: its per-call memo holds the owner Parameter strongly while weakref.finalize (whose registry entry holds that cache) is meant to evict it, a reference
# cycle through a module-global registry, so the finalizer never fires. Everything below only drops entries that cannot be hit by the next request (they are keyed by this
# request's weight addresses, graph generations or epochs) or that the kit rebuilds on demand; outputs stay byte-identical (13/13 and 30/30 compared with fresh CLIs).
KIT_LEVER_RELEASES = (
    ("of3_graphs", "release"),                                   # captured generations, gather tables, shared pool (graphs first: whatever a resident graph reads must stay until it is gone)
    ("of3t_paircache", "release"),
    ("opt_core.of3_trunk.trunk_graph", "release"),
    ("opt_core.of3_trunk.apb_hoist", "release"),
    ("opt_core.of3_sampler.rollout_memo", "STORE.drop_all"),
    ("opt_core.of3_sampler.atom_hoist", "_clear_inv"),
)
FAST_KITS = ("openfold3", "openfold3_ob0")
# Modes in which a kit's resident process gets the release hooks between requests. OpenFold3-ob0 also needs them in `exact`: its exact mode keeps the atom hoist and the other
# per-model state alive in the same way (H100, 30 requests: whole-GPU memory +92% with only the of3_transition guard, flat with the hooks, fork M); OpenFold3's exact mode needs
# only the of3_transition guard and is left as it was.
RELEASE_MODES = {"openfold3": ("fast",), "openfold3_ob0": ("fast", "exact")}


def _purge_trimul_native_packs(module) -> int:
    """Drop the pack-class entries and the per-call memo of every shared serve cache (kit's own _evict_key); the geometry-keyed workspace buffers stay."""
    dropped = 0
    for cache in list(module._SHARED.values()):
        for key in list(module._packclass_keys(cache)):
            module._evict_key(cache, key, None, "gc")
            dropped += 1
        cache.pop(module.CALLS_MEMO_KEY, None)
    return dropped


def release_kit_levers(kit: str, mode: str | None) -> dict:
    """Between requests (fast mode of the OpenFold3 kits, exact mode of OpenFold3-ob0 too, see RELEASE_MODES): call the kit's item-boundary release hooks that are present in this process. A hook that is absent
    (another kit, another version) is skipped; one that raises is left as it was and named in the result. Returns {hook: ok | error} for the hooks that ran."""
    if mode not in RELEASE_MODES.get(kit, ()):
        return {}
    out = {}
    for module_name, attr in KIT_LEVER_RELEASES:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        try:
            target = module
            for part in attr.split("."):
                target = getattr(target, part)
            target()
            out[f"{module_name}.{attr}"] = "ok"
        except Exception as error:  # noqa: BLE001
            out[f"{module_name}.{attr}"] = f"{type(error).__name__}: {error}"[:120]
    # the atom-window kernel memoises a TF32-rounded copy of every weight in a module-level dict keyed by id(parameter) whose values hold the parameter itself
    for module in [mod for mod in list(sys.modules.values()) if mod is not None and hasattr(mod, "weight_tf32") and isinstance(getattr(mod, "_WR", None), dict)]:
        try:
            module._WR.clear()
            out["atom_window._WR"] = "ok"
        except Exception as error:  # noqa: BLE001
            out["atom_window._WR"] = f"{type(error).__name__}: {error}"[:120]
    native = sys.modules.get("opt_core.kernels.trimul.native")
    if native is not None and all(hasattr(native, n) for n in ("_SHARED", "_packclass_keys", "_evict_key", "CALLS_MEMO_KEY")):
        try:
            out["trimul_native.packs"] = _purge_trimul_native_packs(native)
        except Exception as error:  # noqa: BLE001
            out["trimul_native.packs"] = f"{type(error).__name__}: {error}"[:120]
    torch = sys.modules.get("torch")
    clear = getattr(getattr(torch, "_C", None), "_cuda_clearCublasWorkspaces", None) if torch is not None else None
    if callable(clear):
        try:
            clear()
        except Exception as error:  # noqa: BLE001
            out["cublas"] = f"{type(error).__name__}: {error}"[:120]
    return out


def run_one(fn, argv: list) -> int:
    try:
        rc = fn(list(argv))
    except SystemExit as exit_:
        rc = exit_.code if isinstance(exit_.code, int) else (0 if exit_.code is None else 1)
    except BaseException as error:  # a kit crash must not take the resident process down silently
        print(f"foldboost-serve: request raised {error!r}", file=sys.stderr)
        rc = 1
    return int(rc or 0)


class Server:
    def __init__(self, fn, kit: str, keep_cache: bool = False, max_requests: int = 0, max_growth_mib: int = 0):
        self.fn, self.kit, self.key, self.count, self.worst = fn, kit, None, 0, 0
        self.keep_cache, self.max_requests, self.max_growth_mib = keep_cache, max_requests, max_growth_mib
        self.baseline_mib: int | None = None   # reserved CUDA memory after the second request (the first compiles/captures)
        self.recycling = False

    def handle(self, argv: list, log: str | None = None, cwd: str | None = None) -> dict:
        key = compat_key(self.kit, argv)
        if self.key is None:
            self.key = key
        elif key != self.key:
            diff = sorted({f for f, _ in set(key) ^ set(self.key) if f})
            return {"id": None, "rc": REFUSED, "seconds": 0.0, "refused": "configuration differs from the resident process: " + ",".join(map(str, diff))[:200]}
        if self.recycling or (self.max_requests and self.count >= self.max_requests):
            self.recycling = True
            return {"id": None, "rc": REFUSED, "seconds": 0.0, "refused": "resident process is recycling (request or memory limit reached)"}
        self.count += 1
        t0 = time.perf_counter()
        previous = os.getcwd()
        try:
            if cwd:
                os.chdir(cwd)  # the client's relative paths mean what they meant to the client
            with redirected(log):
                rc = run_one(self.fn, argv)
        finally:
            os.chdir(previous)
            if not self.keep_cache:
                release_kit_levers(self.kit, dict(self.key or ()).get("--mode"))
                release_memory()
        self.worst = max(self.worst, rc)
        reserved = reserved_mib()
        result = {"id": self.count, "rc": rc, "seconds": round(time.perf_counter() - t0, 3), "reserved_mib": reserved}
        if self.max_requests and self.count >= self.max_requests:
            self.recycling = True  # the cap is reached: this reply is the last one, the process ends and the supervisor starts a fresh one
            result["recycle"] = f"request cap {self.max_requests} reached"
        if reserved is not None:
            if self.count == 2:
                self.baseline_mib = reserved
            elif self.baseline_mib is not None and self.max_growth_mib and reserved - self.baseline_mib > self.max_growth_mib:
                self.recycling = True  # H100 resident runs crept up ~60 MB per request; stop before the creep matters
                result["recycle"] = f"reserved CUDA memory grew {reserved - self.baseline_mib} MiB over the baseline"
        return result


def private_socket_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, mode=0o700, exist_ok=True)
    st = os.lstat(d)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise SystemExit(f"socket directory {d} must be a directory owned by you with mode 0700")


def serve_socket(server: Server, path: str) -> None:
    private_socket_dir(path)
    if os.path.exists(path):
        os.unlink(path)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(path)
    os.chmod(path, 0o600)
    s.listen(4)
    print(f"foldboost-serve: listening on {path}", flush=True)
    quitting = False
    while not quitting:
        conn, _ = s.accept()
        with conn, conn.makefile("rwb") as stream:
            for line in stream:  # a client may send several messages on one connection (ping, then the request)
                if not line.strip():
                    continue
                request = json.loads(line)
                if request.get("ping"):
                    reply = {"pid": os.getpid(), "kit": server.kit, "requests": server.count, "adapter": os.environ.get("FLASHP_HYBRID_ADAPTER_DIR")}
                elif request.get("quit"):
                    reply, quitting = {"quit": True}, True
                else:
                    reply = server.handle(request["argv"], request.get("log"), request.get("cwd"))
                    if server.recycling:
                        quitting = True  # this reply (a completed request that hit a limit, or the refusal that announces recycling) is the last one; the supervisor starts a fresh process
                stream.write((json.dumps(reply) + "\n").encode())
                stream.flush()
                if quitting or "argv" in request:
                    break
    s.close()
    os.unlink(path)


def client(args) -> int:
    request = {"argv": args.rest, "log": args.log, "cwd": os.getcwd()}
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(args.socket)
        with s.makefile("rwb") as stream:
            stream.write((json.dumps(request) + "\n").encode())
            stream.flush()
            reply = json.loads(stream.readline())
    print(json.dumps(reply), file=sys.stderr)
    if args.log and os.path.exists(args.log):
        sys.stdout.write(open(args.log, errors="replace").read())
    return int(reply["rc"])


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "client":
        cut = argv.index("--") if "--" in argv else len(argv)
        ap = argparse.ArgumentParser(prog="serve_kit client")
        ap.add_argument("--socket", required=True)
        ap.add_argument("--log")
        a = ap.parse_args(argv[1:cut])
        a.rest = argv[cut + 1:]
        return client(a)
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", required=True, help="module:function taking the kit CLI's argv list (after the program name)")
    ap.add_argument("--compat", default=os.environ.get("FP_SERVE_COMPAT", "default"), help="kit name selecting which flags may vary between requests")
    ap.add_argument("--socket", default=os.environ.get("FP_SERVE_SOCKET"), help="serve on this Unix socket (owner-only directory; env FP_SERVE_SOCKET); without it read argv arrays from stdin")
    ap.add_argument("--keep-cache", action="store_true", help="do not empty the CUDA cache between requests")
    ap.add_argument("--max-growth-mib", type=int, default=int(os.environ.get("FP_SERVE_MAX_GROWTH_MIB", "6144")),
                    help="after the 2nd request remember the reserved CUDA memory; recycle when it exceeds that by this much (0 = off)")
    ap.add_argument("--max-requests", type=int, default=int(os.environ.get("FP_SERVE_MAX_REQUESTS", "500")),
                    help="serve this many requests, then refuse further ones so clients run fresh CLIs and a supervisor can restart the process (0 = unlimited)")
    cut = argv.index("--") if "--" in argv else len(argv)
    a = ap.parse_args(argv[:cut])
    first = argv[cut + 1:]
    guards = install_leak_guards(a.compat)
    if guards:
        print("foldboost-serve: leak guards " + json.dumps(guards), flush=True)
    server = Server(resolve(a.entry), a.compat, a.keep_cache, a.max_requests, a.max_growth_mib)
    if first:
        result = server.handle(first)
        print("FPSERVE " + json.dumps(result), flush=True)
    if a.socket:
        serve_socket(server, a.socket)
    else:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            message = json.loads(line)
            if isinstance(message, dict) and message.get("quit"):
                break
            print("FPSERVE " + json.dumps(server.handle(message)), flush=True)
    return server.worst


if __name__ == "__main__":
    sys.exit(main())

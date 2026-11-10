import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from foldboost import serve_kit, serve_wrapper  # noqa: E402
from foldboost.requests import parse_request  # noqa: E402
import fakekit  # noqa: E402

_MISSING = object()


class Handling(unittest.TestCase):
    def setUp(self):
        fakekit.CALLS.clear()
        self.server = serve_kit.Server(fakekit.main, "default")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_per_request_flags_may_vary_but_configuration_may_not(self):
        out = lambda n: str(Path(self.tmp.name) / n)
        r1 = self.server.handle(["pred", "--mode", "exact", "--det", "1", "--input", "a.pdb", "--out", out("1")])
        r2 = self.server.handle(["pred", "--mode", "exact", "--det", "1", "--input", "b.pdb", "--out", out("2")])
        r3 = self.server.handle(["pred", "--mode", "fast", "--det", "1", "--input", "b.pdb", "--out", out("3")])
        r4 = self.server.handle(["pred", "--mode", "exact", "--det", "0", "--input", "b.pdb", "--out", out("4")])
        self.assertEqual((r1["rc"], r2["rc"]), (0, 0))
        self.assertEqual((r3["rc"], r4["rc"]), (serve_kit.REFUSED, serve_kit.REFUSED))
        self.assertIn("--mode", r3["refused"])
        self.assertEqual(len(fakekit.CALLS), 2)  # refused requests never reached the kit
        self.assertTrue(Path(out("2")).read_text().startswith("exact:b.pdb"))

    def test_exit_codes_and_crashes_are_returned_not_raised(self):
        self.assertEqual(self.server.handle(["pred", "--mode", "exact", "--input", "fail"])["rc"], 3)
        self.assertEqual(self.server.handle(["pred", "--mode", "exact", "--input", "boom"])["rc"], 1)
        self.assertEqual(self.server.handle(["pred", "--mode", "exact", "--input", "ok"])["rc"], 0)  # still serving

    def test_each_request_logs_stdout_stderr_and_raw_fd_to_its_own_file(self):
        logs = [str(Path(self.tmp.name) / f"log{i}") for i in (1, 2)]
        self.server.handle(["pred", "--mode", "exact"], logs[0])
        self.server.handle(["pred", "--mode", "exact"], logs[1])
        text1, text2 = (Path(p).read_text() for p in logs)
        for text, n in ((text1, 1), (text2, 2)):
            self.assertIn(f"fakekit stdout {n}", text)
            self.assertIn(f"fakekit stderr {n}", text)
            self.assertIn("fakekit raw fd1", text)
        self.assertNotIn("stdout 2", text1)

    def test_python_level_output_is_logged_even_when_sys_stdout_is_not_fd1(self):
        # an embedding host or pytest's capture replaces sys.stdout/sys.stderr with objects that do not write to fd 1/2
        import io
        log = str(Path(self.tmp.name) / "captured.log")
        old = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        try:
            self.server.handle(["pred", "--mode", "exact"], log)
            leaked = sys.stdout.getvalue() + sys.stderr.getvalue()
        finally:
            sys.stdout, sys.stderr = old
        text = Path(log).read_text()
        self.assertIn("fakekit stdout 1", text)
        self.assertIn("fakekit stderr 1", text)
        self.assertEqual(leaked, "")  # nothing escaped to the host's streams
        self.assertIs(sys.stdout, old[0])  # and they were put back


class LeakGuard(unittest.TestCase):
    """The kit's of3_transition lever caches packed weights by id(module) and never lets go; a resident process builds a model per request."""

    def _fake_kit(self, package):
        import gc
        import types
        pkg = types.ModuleType(package)
        pkg.__path__ = []
        mod = types.ModuleType(package + ".of3_transition")
        mod._PACKS = {}
        mod._built = 0

        def _packed(module, device):
            hit = mod._PACKS.get(id(module))
            if hit is not None:
                return hit[0]
            mod._built += 1
            mod._PACKS[id(module)] = (("pack", mod._built), str(device), 1)
            return mod._PACKS[id(module)][0]

        mod._packed = _packed
        names = (package, package + ".of3_transition")
        before = {k: sys.modules.get(k, _MISSING) for k in names}  # a kit environment may have the real packages loaded
        sys.modules[package] = pkg
        sys.modules[package + ".of3_transition"] = mod
        self.addCleanup(lambda: [sys.modules.pop(k, None) if v is _MISSING else sys.modules.__setitem__(k, v) for k, v in before.items()])
        return mod, gc

    class Module:  # stands for an nn.Module: has a __dict__ and supports weak references
        pass

    def test_entries_die_with_their_module_and_packs_are_unchanged(self):
        for kit, package in serve_kit.LEAKY_TRANSITION_PACKAGES.items():
            mod, gc = self._fake_kit(package)
            self.assertEqual(serve_kit.install_leak_guards(kit), {"of3_transition": "guarded"})
            self.assertEqual(serve_kit.install_leak_guards(kit), {"of3_transition": "already"})  # idempotent
            live = self.Module()
            first = mod._packed(live, "cuda:0")
            self.assertEqual(mod._packed(live, "cuda:0"), first)  # the same live module keeps its pack
            self.assertEqual(mod._built, 1)
            for _ in range(5):  # a resident process: a new model per request, dropped after it
                m = self.Module()
                mod._packed(m, "cuda:0")
                del m
                gc.collect()
            self.assertEqual(len(mod._PACKS), 1)  # only the live module's entry is left
            del live
            gc.collect()
            self.assertEqual(mod._PACKS, {})

    def test_other_kits_and_missing_or_changed_levers_are_left_alone(self):
        self.assertEqual(serve_kit.install_leak_guards("genie3"), {})
        names = ("openfold3_opt", "openfold3_opt.of3_transition")
        before = {k: sys.modules.get(k, _MISSING) for k in names}
        for k in names:
            sys.modules[k] = None  # an import of a None entry raises ImportError, even where the real kit is installed
        try:
            self.assertTrue(serve_kit.install_leak_guards("openfold3")["of3_transition"].startswith("absent"))
        finally:
            for k, v in before.items():
                sys.modules.pop(k, None) if v is _MISSING else sys.modules.__setitem__(k, v)
        mod, _ = self._fake_kit("openfold3_opt")
        del mod._PACKS
        self.assertEqual(serve_kit.install_leak_guards("openfold3"), {"of3_transition": "layout-changed"})


class RecycleLimits(unittest.TestCase):
    def test_the_request_that_hits_the_cap_is_the_last_one_served(self):
        fakekit.CALLS.clear()
        server = serve_kit.Server(fakekit.main, "default", max_requests=2)
        with tempfile.TemporaryDirectory() as tmp:
            out = lambda n: str(Path(tmp) / n)
            r1 = server.handle(["pred", "--mode", "exact", "--out", out("1")])
            self.assertEqual(r1["rc"], 0)
            self.assertNotIn("recycle", r1)
            self.assertFalse(server.recycling)
            r2 = server.handle(["pred", "--mode", "exact", "--out", out("2")])
            self.assertEqual(r2["rc"], 0)  # served normally, then announced as the last
            self.assertIn("request cap 2", r2["recycle"])
            self.assertTrue(server.recycling)
            r3 = server.handle(["pred", "--mode", "exact", "--out", out("3")])
            self.assertEqual(r3["rc"], serve_kit.REFUSED)
            self.assertEqual(len(fakekit.CALLS), 2)


class KitLeverRelease(unittest.TestCase):
    """fast mode of the OpenFold3 kits: the kit's own item-boundary release hooks run between requests, exact mode and other kits are left alone."""

    def _install(self):
        import types
        calls = []
        mods = {}
        for name, attr in serve_kit.KIT_LEVER_RELEASES:
            m = types.ModuleType(name)
            target = m
            parts = attr.split(".")
            for part in parts[:-1]:  # e.g. STORE.drop_all
                holder = types.SimpleNamespace()
                setattr(target, part, holder)
                target = holder
            setattr(target, parts[-1], (lambda n=name, a=attr: calls.append(f"{n}.{a}")))
            mods[name] = m
        native = types.ModuleType("opt_core.kernels.trimul.native")
        native.CALLS_MEMO_KEY = "trimul_native.calls"
        pack = ("trimul_native.pack", 1, 2)
        native._SHARED = {"geom": {pack: "packed", ("other", 1): "workspace", native.CALLS_MEMO_KEY: {"call": {"wsrc": object()}}}}
        native._packclass_keys = lambda cache: [k for k in cache if isinstance(k, tuple) and k[0] == "trimul_native.pack"]

        def evict(cache, key, w_ptr=None, why="gc"):
            cache.pop(key, None)
            calls.append(f"evict:{key[0]}")
        native._evict_key = evict
        mods[native.__name__] = native
        aw = types.ModuleType("fake_opt_core_atom_window_kernel")
        aw._WR = {id(object()): ("param", 1, "copy")}
        aw.weight_tf32 = lambda w: w
        mods[aw.__name__] = aw
        self._aw = aw
        saved = {k: sys.modules.get(k, _MISSING) for k in mods}
        sys.modules.update(mods)
        self.addCleanup(lambda: [sys.modules.pop(k, None) if v is _MISSING else sys.modules.__setitem__(k, v) for k, v in saved.items()])
        return calls, native

    def test_fast_mode_calls_every_hook_and_purges_only_pack_entries(self):
        calls, native = self._install()
        out = serve_kit.release_kit_levers("openfold3", "fast")
        for name, attr in serve_kit.KIT_LEVER_RELEASES:
            self.assertIn(f"{name}.{attr}", calls)
            self.assertEqual(out[f"{name}.{attr}"], "ok")
        self.assertEqual(out["trimul_native.packs"], 1)
        self.assertEqual(out["atom_window._WR"], "ok")
        self.assertEqual(self._aw._WR, {})                             # the memo of TF32 weight copies (strong references to the parameters) is emptied
        cache = native._SHARED["geom"]
        self.assertNotIn(("trimul_native.pack", 1, 2), cache)           # the pack entry (keyed by this request's weight addresses) is gone
        self.assertNotIn("trimul_native.calls", cache)                  # and so is the per-call memo that held the owner tensors
        self.assertEqual(cache[("other", 1)], "workspace")             # geometry-keyed workspace buffers stay
        # the kit's own order: graphs first
        self.assertEqual(calls.index("of3_graphs.release"), 0)

    def test_exact_mode_other_kits_and_missing_hooks_are_left_alone(self):
        calls, native = self._install()
        self.assertEqual(serve_kit.release_kit_levers("openfold3", "exact"), {})   # OpenFold3 exact needs only the of3_transition guard
        self.assertEqual(serve_kit.release_kit_levers("genie3", "fast"), {})
        self.assertEqual(calls, [])
        self.assertIn(("trimul_native.pack", 1, 2), native._SHARED["geom"])
        # a hook that raises is named and does not stop the others
        sys.modules["of3_graphs"].release = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        out = serve_kit.release_kit_levers("openfold3_ob0", "fast")
        self.assertTrue(out["of3_graphs.release"].startswith("RuntimeError"))
        self.assertEqual(out["of3t_paircache.release"], "ok")
        # a process without these modules (another kit version): nothing to do
        for name in [n for n, _ in serve_kit.KIT_LEVER_RELEASES] + ["opt_core.kernels.trimul.native", "fake_opt_core_atom_window_kernel"]:
            sys.modules.pop(name, None)
        self.assertEqual(serve_kit.release_kit_levers("openfold3", "fast"), {})

    def test_openfold3_ob0_exact_gets_the_hooks_too_but_openfold3_exact_does_not(self):
        calls, _ = self._install()
        out = serve_kit.release_kit_levers("openfold3_ob0", "exact")
        self.assertEqual(out["of3_graphs.release"], "ok")
        self.assertIn("of3_graphs.release", calls)
        calls.clear()
        self.assertEqual(serve_kit.release_kit_levers("openfold3", "exact"), {})
        self.assertEqual(serve_kit.release_kit_levers("openfold3_ob0", "off"), {})
        self.assertEqual(calls, [])
        ob0 = serve_kit.Server(fakekit.main, "openfold3_ob0")
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(ob0.handle(["pred", "--mode", "exact", "--query-json", str(Path(tmp) / "a.json")])["rc"], 0)
        self.assertEqual(calls.count("of3_graphs.release"), 1)

    def test_the_server_runs_the_hooks_after_every_fast_request_only(self):
        calls, _ = self._install()
        fake = serve_kit.Server(fakekit.main, "openfold3")
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(fake.handle(["pred", "--mode", "fast", "--query-json", str(Path(tmp) / "a.json")])["rc"], 0)
            n = calls.count("of3_graphs.release")
            self.assertEqual(n, 1)
            self.assertEqual(fake.handle(["pred", "--mode", "fast", "--query-json", str(Path(tmp) / "b.json")])["rc"], 0)
            self.assertEqual(calls.count("of3_graphs.release"), 2)
        calls2, _ = self._install()
        exact = serve_kit.Server(fakekit.main, "openfold3")
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(exact.handle(["pred", "--mode", "exact", "--query-json", str(Path(tmp) / "c.json")])["rc"], 0)
        self.assertEqual(calls2, [])


class SocketRoundTrip(unittest.TestCase):
    def test_server_client_and_refusal_over_a_private_socket(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "sock"
            d.mkdir(mode=0o700)
            path = str(d / "s")
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(Path(__file__).resolve().parent)]))
            with subprocess.Popen([sys.executable, "-m", "foldboost.serve_kit", "--entry", "fakekit:main", "--socket", path],
                                  env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as proc:
                try:
                    for _ in range(100):
                        if os.path.exists(path):
                            break
                        time.sleep(0.05)
                    self.assertEqual(oct(os.stat(path).st_mode & 0o777), "0o600")
                    run = lambda *rest, log=None: subprocess.run(
                        [sys.executable, "-m", "foldboost.serve_kit", "client", "--socket", path, *(["--log", log] if log else []), "--", *rest],
                        env=env, capture_output=True, text=True)
                    log = str(Path(tmp) / "r.log")
                    out = str(Path(tmp) / "o.txt")
                    ok = run("pred", "--mode", "exact", "--input", "x", "--out", out, log=log)
                    self.assertEqual(ok.returncode, 0, ok.stderr)
                    self.assertIn("fakekit stdout 1", ok.stdout)
                    pid_in_output = Path(out).read_text().strip().split(":")[-1]
                    self.assertEqual(pid_in_output, str(proc.pid))  # executed inside the resident process
                    refused = run("pred", "--mode", "fast")
                    self.assertEqual(refused.returncode, serve_kit.REFUSED)
                    self.assertIn("refused", refused.stderr)
                    with socket.socket(socket.AF_UNIX) as s:
                        s.connect(path)
                        s.sendall(b'{"quit": true}\n')
                        s.recv(100)
                    proc.wait(timeout=10)
                finally:
                    if proc.poll() is None:
                        proc.kill()

    def test_insecure_socket_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.chmod(tmp, 0o755)
            with self.assertRaises(SystemExit):
                serve_kit.private_socket_dir(str(Path(tmp) / "s"))


class Wrapper(unittest.TestCase):
    def test_only_the_final_exec_changes(self):
        src = "#!/bin/bash\nset -e\nHERE=x\nexport A=1\nexec python -m demo_opt \"$CMD\" ${ARGS[@]+\"${ARGS[@]}\"}\n"
        out, pkg = serve_wrapper.derive(src)
        self.assertEqual(pkg, "demo_opt")
        a, b = src.splitlines(), out.splitlines()
        self.assertEqual(a[:-1], b[:-1])
        self.assertIn("foldboost.serve_kit --entry demo_opt.__main__:main --", b[-1])

    def test_exec_inside_a_case_branch_and_entry_override(self):
        line = '  check|warm|pred) exec python3 -m atlasfold_opt "$verb" "$@" ;;\n'
        src = "#!/bin/bash\nset -e\ncase \"$verb\" in\n  install) echo x ;;\n" + line + "  *) exit 2 ;;\nesac\n"
        out, pkg = serve_wrapper.derive(src, "atlasfold_opt.cli:main")
        self.assertEqual(pkg, "atlasfold_opt")
        changed = [b for a, b in zip(src.splitlines(), out.splitlines()) if a != b]
        self.assertEqual(changed, ['  check|warm|pred) exec python -m foldboost.serve_kit --entry atlasfold_opt.cli:main -- "$verb" "$@" ;;'])

    def test_interpreter_variable_is_kept_and_a_fixed_verb_exec_is_not_a_launcher(self):
        src = '#!/bin/bash\nset -e\nPY="${PROGEN2_PYTHON:-python}"\nARGS=("$@")\nexec "$PY" -m progen2_opt "${ARGS[@]}"\n'
        out, pkg = serve_wrapper.derive(src)
        self.assertEqual(pkg, "progen2_opt")
        self.assertEqual(out.splitlines()[-1], 'exec "$PY" -m foldboost.serve_kit --entry progen2_opt.__main__:main -- "${ARGS[@]}"')
        self.assertEqual(out.splitlines()[:-1], src.splitlines()[:-1])
        # enformer / gpnstar / evo2 only exec the fixed verb `check`: deriving a launcher from it would serve nothing
        with self.assertRaises(SystemExit):
            serve_wrapper.derive('PY="${PYTHON:-python}"\ncase "$1" in\n  check)\n    exec "$PY" -m enformer_opt check "$@" ;;\nesac\n')

    def test_refuses_ambiguous_scripts(self):
        with self.assertRaises(SystemExit):
            serve_wrapper.derive("echo hi\n")


class CompatEsm(unittest.TestCase):
    def test_esm_if1_requests_for_other_structures_share_a_process_but_other_modes_do_not(self):
        a = ["design", "--mode", "fast", "/code/1UBQ.pdb", "--chain", "A", "--num-samples", "2", "--outpath", "/o/a/samples.fasta", "--seed", "1", "--batch_size", "2"]
        b = ["design", "--mode", "fast", "/code/2LZM.pdb", "--chain", "A", "--num-samples", "2", "--outpath", "/o/b/samples.fasta", "--seed", "2", "--batch_size", "2"]
        c = ["design", "--mode", "off", "/code/2LZM.pdb", "--chain", "A", "--num-samples", "2", "--outpath", "/o/b/samples.fasta", "--seed", "2", "--batch_size", "2"]
        d = ["design", "--mode", "fast", "/code/2LZM.pdb", "--chain", "B", "--num-samples", "2", "--outpath", "/o/b/samples.fasta", "--seed", "2", "--batch_size", "2"]
        self.assertEqual(serve_kit.compat_key("esm_if1", a), serve_kit.compat_key("esm_if1", b))
        self.assertNotEqual(serve_kit.compat_key("esm_if1", a), serve_kit.compat_key("esm_if1", c))
        self.assertNotEqual(serve_kit.compat_key("esm_if1", a), serve_kit.compat_key("esm_if1", d))


class RouterToResident(unittest.TestCase):
    def test_launcher_uses_the_resident_process_and_falls_back_when_it_cannot(self):
        import argparse
        from unittest import mock
        from foldboost import launch
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "sock"
            d.mkdir(mode=0o700)
            path = str(d / "s")
            kit = Path(tmp) / "kit"
            kit.mkdir()
            (kit / "run.sh").write_text('#!/bin/bash\necho FRESH-CLI "$@"\n')
            (kit / "run.sh").chmod(0o755)
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(Path(__file__).resolve().parent)]))
            with subprocess.Popen([sys.executable, "-m", "foldboost.serve_kit", "--entry", "fakekit:main", "--socket", path],
                                  env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as proc:
                try:
                    for _ in range(100):
                        if os.path.exists(path):
                            break
                        time.sleep(0.05)
                    ns = argparse.Namespace(kit="genie3", kit_dir=str(kit), python=sys.executable, weights=[], trust_weights=True, allow_cold=True,
                                            cache_dir=str(Path(tmp) / "c"), log_dir=str(Path(tmp) / "logs"), site_packages=None, dry_run=False,
                                            allow_inactive=False, server=path)
                    out = str(Path(tmp) / "o.txt")
                    argv = ["design", "--config", "h100", "--mode", "exact", "--input", "x.yaml", "--out", out]
                    # unmeasured request: provider anthropic, mode exact -> served by the resident process, relative cwd preserved
                    from foldboost.selector import load_policy
                    supported = dict(load_policy())
                    supported["resident"] = {"genie3": {"exact": {"status": "supported"}, "fast": {"status": "supported"}}}
                    with mock.patch("foldboost.launch.load_policy", lambda: supported), \
                         mock.patch("foldboost.launch.probe_gpu", lambda: "NVIDIA H100 80GB HBM3"), \
                         mock.patch("foldboost.launch.probe_python", lambda p: {}), \
                         mock.patch("foldboost.launch.parse_request", lambda kit, a, hasher=None: parse_request(kit, a, lambda p: "0" * 64)):
                        self.assertEqual(launch.run(ns, argv), 0)
                        record = json.loads((Path(tmp) / "logs" / "decisions.jsonl").read_text().splitlines()[-1])
                        self.assertEqual(record["route"], "resident")
                        self.assertEqual(Path(out).read_text().split(":")[-1].strip(), str(proc.pid))
                        # a different mode is refused by the resident process -> fresh CLI runs instead
                        self.assertEqual(launch.run(ns, ["design", "--config", "h100", "--mode", "fast", "--input", "x.yaml", "--out", out]), 0)
                        record = json.loads((Path(tmp) / "logs" / "decisions.jsonl").read_text().splitlines()[-1])
                        self.assertEqual(record["route"], "fresh")
                        self.assertIn("refused", record["resident"]["skipped"])
                        # no server at all -> fresh CLI
                        ns.server = str(Path(tmp) / "missing.sock")
                        self.assertEqual(launch.run(ns, argv), 0)
                        record = json.loads((Path(tmp) / "logs" / "decisions.jsonl").read_text().splitlines()[-1])
                        self.assertEqual(record["route"], "fresh")
                        self.assertIn("unavailable", record["resident"]["skipped"])
                    with socket.socket(socket.AF_UNIX) as q:
                        q.connect(path)
                        q.sendall(b'{"quit": true}\n')
                        q.recv(100)
                    proc.wait(timeout=10)
                finally:
                    if proc.poll() is None:
                        proc.kill()

    def test_kit_without_measured_resident_support_never_uses_the_server(self):
        import argparse
        from unittest import mock
        from foldboost import launch
        from foldboost.selector import load_policy
        policy = load_policy()
        ns = argparse.Namespace(kit="gpnstar", server="/nonexistent.sock")
        decision = launch.Decision("anthropic", "exact", "t")
        code, info = launch.try_resident(ns, ["pred", "--mode", "exact"], decision, "/tmp/x.log", policy)
        self.assertIsNone(code)
        self.assertIn("not validated", info["skipped"])
        self.assertIn("unsupported", info["skipped"])

    def test_stock_decisions_use_the_resident_process_only_when_it_is_recorded_as_faster_than_stock(self):
        import argparse
        from foldboost import launch
        policy = {"resident": {"atlasfold": {"exact": {"status": "supported", "beats_stock": True, "gpu_names": ["G"]}},
                               "openfold3": {"exact": {"status": "supported", "gpu_names": ["G"]}}}}
        decision = launch.Decision("stock", "off", "t")
        ns = argparse.Namespace(kit="openfold3", server="/nonexistent.sock")
        code, info = launch.try_resident(ns, ["pred", "--mode", "exact"], decision, "/tmp/x.log", policy, "G", "exact")
        self.assertIsNone(code)
        self.assertIn("not recorded as faster than fresh stock", info["skipped"])
        ns = argparse.Namespace(kit="atlasfold", server="/nonexistent.sock")
        code, info = launch.try_resident(ns, ["pred", "--mode", "exact"], decision, "/tmp/x.log", policy, "G", "exact")
        self.assertIn("unavailable", info["skipped"])  # it tried the socket (absent here): the support check passed
        code, info = launch.try_resident(ns, ["pred", "--mode", "exact"], decision, "/tmp/x.log", policy, "OTHER-GPU", "exact")
        self.assertIn("validated for", info["skipped"])


class ServeControl(unittest.TestCase):
    def test_start_serve_stop_with_a_derived_launcher(self):
        from foldboost import serve_ctl
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            kit = tmp / "kit"
            kit.mkdir()
            (kit / "run.sh").write_text('#!/bin/bash\nset -e\nexport KIT_ENV_PREPARED=yes\nexec python -m fakepkg "$@"\n')
            (kit / "run.sh").chmod(0o755)
            sockdir = tmp / "s"
            sockdir.mkdir(mode=0o700)
            path = str(sockdir / "k.sock")
            (tmp / "bin").mkdir()
            (tmp / "bin" / "python").symlink_to(sys.executable)  # the kit's `python` (this machine may only have python3)
            tests_dir = str(Path(__file__).resolve().parent)
            old = os.environ.get("PYTHONPATH")
            os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [tests_dir, old]))
            try:
                first_out = str(tmp / "first.txt")
                self.assertEqual(serve_ctl.start("fakepkg", kit, path, ["pred", "--mode", "exact", "--out", first_out], python=str(tmp / "bin" / "python"), max_requests=2, wait=60), 0)
                info = serve_ctl.status(path)
                self.assertIsNotNone(info)
                self.assertEqual(info["kit"], "fakepkg")
                out2 = tmp / "second.txt"
                reply = serve_ctl._rpc(path, {"argv": ["pred", "--mode", "exact", "--out", str(out2)], "cwd": str(tmp)})
                self.assertEqual(reply["rc"], 0)
                self.assertEqual(out2.read_text(), str(info["pid"]))  # ran inside the resident process
                # request cap (first + one) reached: this reply is the last one and the process ends by itself, so a supervisor can start a fresh one.
                # (Before the fix the server stayed up and refused every later request for ever.)
                self.assertIn("request cap", reply.get("recycle", ""))
                serve_ctl.stop(path)  # harmless if it is already gone
                for _ in range(100):
                    if serve_ctl.status(path) is None:
                        break
                    time.sleep(0.1)
                self.assertIsNone(serve_ctl.status(path))
            finally:
                serve_ctl.stop(path)
                if old is None:
                    os.environ.pop("PYTHONPATH", None)
                else:
                    os.environ["PYTHONPATH"] = old


if __name__ == "__main__":
    unittest.main()

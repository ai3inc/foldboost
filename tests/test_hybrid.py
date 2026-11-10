"""CPU tests for the hybrid router. They prove routing logic and adapter isolation, not GPU speed
and not that an adapter fires inside a real model; those need the pinned H100 environments."""
import argparse
import copy
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foldboost import adapters, launch  # noqa: E402
from foldboost.requests import parse_request, with_mode  # noqa: E402
from foldboost.selector import Facts, load_policy, select  # noqa: E402

POLICY = load_policy()


def facts_for(scope: dict, **override) -> Facts:
    env = scope["environment"]
    f = Facts(gpu_name=env["gpu_name"], torch=env["torch"], triton=env["triton"], python=env.get("python") or "3.11.5",
              satisfied=set(scope.get("requires", [])), verified_weights={w["sha256"] for w in scope.get("weights") or []})
    for key, value in override.items():
        setattr(f, key, value)
    return f


def request_for(scope: dict, argv=None):
    ex = scope["example"]
    return parse_request(scope["kit"], argv or ex["argv"], ex["input_paths"].__getitem__)


class PolicyIntegrity(unittest.TestCase):
    def test_digest_and_adapters(self):
        self.assertEqual(len(POLICY["scopes"]), 18)
        for adapter_id in adapters.registry():
            adapters.verify_adapter(adapter_id)
        self.assertTrue({s["adapter"] for s in POLICY["scopes"]} <= set(adapters.registry()))

    def test_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = copy.deepcopy(POLICY)
            bad["scopes"][0]["decision"]["provider"] = "anthropic"
            path = Path(tmp) / "p.json"
            path.write_text(json.dumps(bad))
            with self.assertRaises(Exception):
                load_policy(path)

    def test_coverage_is_72_rows_with_separate_statuses(self):
        rows = POLICY["coverage"]
        self.assertEqual(len(rows), 72)
        statuses = {r["status"] for r in rows}
        self.assertEqual(statuses, {"scoped_evidence", "not_measured", "candidate_rejected", "weights_access_blocked"})
        self.assertTrue(all(r["status"] != "scoped_evidence" or r["gpu"] == "H100" for r in rows))

    def test_status_is_honest_about_what_was_validated(self):
        self.assertIn("discovery-frozen", POLICY["status"])
        self.assertIn("anthropic", POLICY["status"])
        self.assertTrue(all(s["validation"]["status"] == "discovery-only" for s in POLICY["scopes"]))  # exact scopes stay discovery evidence
        self.assertTrue(all("holdout" in f["evidence"] for f in POLICY["families"]))  # widening needs a holdout

    def test_ours_never_chosen_in_a_statistical_tie(self):
        for s in POLICY["scopes"]:
            if s["decision"]["provider"] == "ours":
                for other in s["correctness"]["group_members"]:
                    if other != "ours":
                        self.assertGreaterEqual(s["pairwise_ratio_slower_over_faster"][f"ours_over_{other}"]["lower95"], 1.0, s["scope_id"])


class Routing(unittest.TestCase):
    def test_every_recorded_request_routes_to_its_scope(self):
        for s in POLICY["scopes"]:
            d = select(POLICY, request_for(s), facts_for(s))
            self.assertEqual(d.scope_id, s["scope_id"])
            self.assertEqual(d.provider, s["decision"]["provider"])
            self.assertEqual(d.adapter_enabled, d.provider == "ours")

    def test_atlas_228_selects_stock_in_off_mode_because_anthropic_exact_is_slower(self):
        s = next(x for x in POLICY["scopes"] if x["scope_id"].endswith("037.228"))
        d = select(POLICY, request_for(s), facts_for(s))
        self.assertEqual((d.provider, d.mode, d.adapter_enabled), ("stock", "off", False))
        self.assertIsNotNone(d.adapter)  # inert bootstrap stays present, as in the measured arms

    def test_any_change_leaves_scope_and_selects_anthropic_with_mode_kept(self):
        for s in POLICY["scopes"]:
            base = s["example"]["argv"]
            mutated = []
            for i, tok in enumerate(base[:-1]):
                if tok in ("--seed", "--seeds") and base[i + 1].isdigit():
                    m = list(base)
                    m[i + 1] = str(int(m[i + 1]) + 1)
                    mutated.append(("seed", m))
            for label, argv in mutated:
                d = select(POLICY, request_for(s, argv), facts_for(s))
                self.assertEqual((d.provider, d.mode), ("anthropic", s["contract"]), (s["scope_id"], label))
                self.assertIn("outside_all_evidence_scopes", d.reason)
            d = select(POLICY, request_for(s, base + ["--surprise", "1"]), facts_for(s))
            self.assertEqual(d.provider, "anthropic")
            self.assertTrue(d.reason.startswith("unrecognized_options"))

    def test_environment_gpu_and_mode_mismatches_fall_back(self):
        s = next(x for x in POLICY["scopes"] if x["scope_id"].endswith("050.0"))
        req = request_for(s)
        for override in (dict(gpu_name="NVIDIA B300"), dict(torch="2.7.2+cu126"), dict(triton="3.3.0"), dict(python="3.10.14"), dict(verified_weights=set())):
            d = select(POLICY, req, facts_for(s, **override))
            self.assertEqual(d.provider, "anthropic", override)
        fast = parse_request("genie3", with_mode(s["example"]["argv"], "fast"), s["example"]["input_paths"].__getitem__)
        d = select(POLICY, fast, facts_for(s))
        self.assertEqual((d.provider, d.mode), ("anthropic", "fast"))  # never silently switched
        big = parse_request("genie3", with_mode(s["example"]["argv"], "big"), s["example"]["input_paths"].__getitem__)
        self.assertEqual(select(POLICY, big, facts_for(s)).reason, "mode_without_evidence:big")

    def test_off_is_respected_and_never_overridden(self):
        s = next(x for x in POLICY["scopes"] if x["scope_id"].endswith("050.0"))
        req = parse_request("genie3", with_mode(s["example"]["argv"], "off"), s["example"]["input_paths"].__getitem__)
        d = select(POLICY, req, facts_for(s))
        self.assertEqual((d.provider, d.mode, d.adapter_enabled), ("stock", "off", False))

    def test_unmeasured_kit_gets_anthropic(self):
        req = parse_request("chai1", ["pred", "--mode", "fast"], lambda p: "x")
        self.assertEqual(select(POLICY, req, Facts()).provider, "anthropic")

    def test_cold_cache_precondition(self):
        s = next(x for x in POLICY["scopes"] if x["kit"] == "proteinmpnn")
        d = select(POLICY, request_for(s), facts_for(s, satisfied=set()))
        self.assertEqual(d.provider, "anthropic")
        self.assertIn("precondition:warm_fingerprint_cache", json.dumps(d.near_misses))


class Isolation(unittest.TestCase):
    def test_gate_imports_only_the_selected_adapter_and_is_inert_otherwise(self):
        probe = subprocess.run([sys.executable, "-I", "-c", "import sys;print('flashpair_bootstrap' in sys.modules)"], capture_output=True, text=True)
        if probe.stdout.strip() == "True":  # e.g. a kit environment whose own .pth already loads a flashpair_bootstrap
            self.skipTest("this interpreter already imports a flashpair_bootstrap module at start-up")
        with tempfile.TemporaryDirectory() as tmp:
            adapters.install_gate(Path(tmp))
            self.assertTrue(adapters.gate_installed(Path(tmp)))
            code = "import site,sys;site.addsitedir(sys.argv[1]);print('flashpair_bootstrap' in sys.modules, getattr(sys.modules.get('flashpair_bootstrap'),'__file__',None))"
            base = {k: v for k, v in os.environ.items() if not k.startswith("FLASHP_")}
            off = subprocess.run([sys.executable, "-I", "-c", code, tmp], capture_output=True, text=True, env=base)
            self.assertTrue(off.stdout.startswith("False"), off.stdout + off.stderr)
            for adapter_id in ("genie3-dispatch-050", "atlas-mmap-037"):
                env = dict(base, **adapters.adapter_environment(adapter_id, False))
                on = subprocess.run([sys.executable, "-I", "-c", code, tmp], capture_output=True, text=True, env=env)
                self.assertIn(str(adapters.adapter_dir(adapter_id)), on.stdout, on.stderr)

    def test_modified_adapter_is_refused(self):
        target = adapters.adapter_dir("genie3-dispatch-050") / "dispatch_loader.py"
        original = target.read_bytes()
        try:
            target.write_bytes(original + b"\n# edit\n")
            with self.assertRaises(adapters.AdapterError):
                adapters.verify_adapter("genie3-dispatch-050")
        finally:
            target.write_bytes(original)


class Launcher(unittest.TestCase):
    def _kit(self, tmp: Path, body: str) -> Path:
        kit = tmp / "kit"
        kit.mkdir()
        (kit / "run.sh").write_text(body)
        (kit / "run.sh").chmod(stat.S_IRWXU)
        return kit

    def _args(self, tmp: Path, kit: Path, scope: dict, **extra):
        site = tmp / "site"
        site.mkdir(exist_ok=True)
        adapters.install_gate(site)
        ns = dict(kit=scope["kit"], kit_dir=str(kit), python=sys.executable, weights=[], trust_weights=True, allow_cold=True,
                  cache_dir=str(tmp / "cache"), log_dir=str(tmp / "logs"), site_packages=str(site), dry_run=False, allow_inactive=False)
        ns.update(extra)
        return argparse.Namespace(**ns)

    def _patched(self, scope):
        env = scope["environment"]
        paths = scope["example"]["input_paths"]
        for target, value in (
            ("foldboost.launch.probe_gpu", lambda: env["gpu_name"]),
            ("foldboost.launch.probe_python", lambda py: {"python": env.get("python") or "3.11.5", "torch": env["torch"], "triton": env["triton"]}),
            ("foldboost.launch.parse_request", lambda kit, argv, hasher=None: parse_request(kit, argv, paths.__getitem__)),
        ):
            patcher = mock.patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_run_records_decision_and_flags_an_adapter_that_did_not_fire(self):
        s = next(x for x in POLICY["scopes"] if x["scope_id"].endswith("050.0"))
        self._patched(s)
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            kit = self._kit(tmp, '#!/bin/bash\necho "ARGS $@"\necho "FLAG=$FLASHP_GENIE3_DISPATCH DIR=$FLASHP_HYBRID_ADAPTER_DIR"\n')
            args = self._args(tmp, kit, s)
            code = launch.run(args, list(s["example"]["argv"]))
            self.assertEqual(code, launch.INACTIVE_EXIT)  # selected ours, no audit line printed
            rec = json.loads((tmp / "logs" / "decisions.jsonl").read_text().splitlines()[-1])
            self.assertEqual(rec["decision"]["provider"], "ours")
            self.assertFalse(rec["activation"]["active"])
            self.assertIn("warning", rec)
            self.assertIn("FLAG=1", (tmp / "logs" / "last-run.log").read_text())
            args.allow_inactive = True
            self.assertEqual(launch.run(args, list(s["example"]["argv"])), 0)

    def test_active_adapter_is_confirmed_from_audit_lines(self):
        s = next(x for x in POLICY["scopes"] if x["scope_id"].endswith("050.0"))
        self._patched(s)
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            kit = self._kit(tmp, '#!/bin/bash\necho \'[flashpair-model-audit] {"adapter": "x", "candidate_calls": 1, "fallback_calls": 0}\'\n')
            self.assertEqual(launch.run(self._args(tmp, kit, s), list(s["example"]["argv"])), 0)
            rec = json.loads((tmp / "logs" / "decisions.jsonl").read_text().splitlines()[-1])
            self.assertTrue(rec["activation"]["active"])

    def test_fallback_runs_with_inert_environment_and_unchanged_mode(self):
        s = next(x for x in POLICY["scopes"] if x["scope_id"].endswith("050.0"))
        self._patched(s)
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            kit = self._kit(tmp, '#!/bin/bash\necho "ARGS $@"\n')
            argv = list(s["example"]["argv"]) + ["--extra-unknown", "1"]
            self.assertEqual(launch.run(self._args(tmp, kit, s), argv), 0)
            rec = json.loads((tmp / "logs" / "decisions.jsonl").read_text().splitlines()[-1])
            self.assertEqual(rec["decision"]["provider"], "anthropic")
            self.assertIn("--mode exact", (tmp / "logs" / "last-run.log").read_text())


    def test_missing_gate_fails_closed_without_a_server_and_downgrades_with_one(self):
        s = next(x for x in POLICY["scopes"] if x["scope_id"].endswith("050.0"))
        self._patched(s)
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            (tmp / "nosite").mkdir()
            kit = self._kit(tmp, '#!/bin/bash\necho "ARGS $@"\necho "FLAG=${FLASHP_GENIE3_DISPATCH:-unset} DIR=${FLASHP_HYBRID_ADAPTER_DIR:-unset}"\n')
            argv = list(s["example"]["argv"])
            # no resident server: the documented fail-closed behaviour is kept
            direct = self._args(tmp, kit, s, site_packages=str(tmp / "nosite"))
            self.assertEqual(launch.run(direct, argv), 2)
            # a resident deployment: the fresh CLI behind a refusing or restarting server must not fail for an optional gate
            served = self._args(tmp, kit, s, site_packages=str(tmp / "nosite"), server=str(tmp / "absent.sock"))
            self.assertEqual(launch.run(served, argv), 0)
            rec = json.loads((tmp / "logs" / "decisions.jsonl").read_text().splitlines()[-1])
            self.assertEqual(rec["decision"]["provider"], "anthropic")
            self.assertFalse(rec["decision"]["adapter_enabled"])
            self.assertIn("downgraded", rec["decision"]["reason"])
            self.assertIn("interpreter gate missing", rec["downgraded"])
            self.assertEqual(rec["route"], "fresh")
            log = (tmp / "logs" / "last-run.log").read_text()
            self.assertIn("--mode exact", log)   # the requested mode is unchanged
            self.assertIn("FLAG=unset DIR=unset", log)  # Anthropic's path: no adapter environment

    def test_resident_route_records_what_actually_ran(self):
        s = next(x for x in POLICY["scopes"] if x["scope_id"].endswith("050.0"))
        self._patched(s)
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            kit = self._kit(tmp, '#!/bin/bash\nexit 9\n')
            served = self._args(tmp, kit, s, server=str(tmp / "s.sock"))
            with mock.patch("foldboost.launch.try_resident", return_value=(0, {"socket": "x", "adapter_in_resident_process": False})):
                self.assertEqual(launch.run(served, list(s["example"]["argv"])), 0)
            rec = json.loads((tmp / "logs" / "decisions.jsonl").read_text().splitlines()[-1])
            self.assertEqual(rec["route"], "resident")
            self.assertEqual(rec["decision"]["provider"], "ours")          # what the policy chose
            self.assertEqual(rec["executed_provider"], "anthropic")        # what ran: the server's unmodified path


class Sizes(unittest.TestCase):
    def test_measures_read_tokens_and_lengths(self):
        from foldboost import sizes
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "g.yaml").write_text("generation:\n  dataset:\n    min_length: 224\n    max_length: 224\n")
            self.assertEqual(sizes.measure("genie3", str(t / "g.yaml")), 224)
            (t / "g2.yaml").write_text("min_length: 100\nmax_length: 120\n")
            self.assertIsNone(sizes.measure("genie3", str(t / "g2.yaml")))
            (t / "q.json").write_text(json.dumps({"queries": {"u": {"chains": [{"molecule_type": "protein", "chain_ids": list("ABC"), "sequence": "M" * 76}]}}}))
            self.assertEqual(sizes.measure("openfold3", str(t / "q.json")), 228)
            (t / "a.fasta").write_text(">x\nMQIF\nVKTL\n")
            self.assertEqual(sizes.measure("atlasfold", str(t / "a.fasta")), 8)
            atom = lambda ch, n: f"ATOM  {n:5d}  CA  ALA {ch}{n:4d}    " + "0.000".rjust(8) * 3 + "\n"
            (t / "p.pdb").write_text("".join(atom("A", i) for i in range(1, 6)) + "".join(atom("B", i) for i in range(1, 4)))
            self.assertEqual(sizes.measure("esm_if1", str(t / "p.pdb"), {"chain": "A"}), 5)
            self.assertEqual(sizes.measure("proteinmpnn", str(t)), 8)
            self.assertIsNone(sizes.measure("openfold3", str(t / "missing.json")))
            self.assertIsNone(sizes.measure("chai1", str(t / "p.pdb")))


class Families(unittest.TestCase):
    def _genie(self, tmp, length, extra=()):
        yaml = Path(tmp) / f"g{length}.yaml"
        yaml.write_text(f"generation:\n  dataset:\n    min_length: {length}\n    max_length: {length}\n")
        argv = ["design", "--config", "h100", "--mode", "exact", "--input", str(yaml), "--out_dir", "out", "--det", "1", "--seed", "123", "--n", "2", "--batch_size", "1", *extra]
        return parse_request("genie3", argv, lambda p: "f" * 64)

    def test_unseen_input_inside_the_tested_hull_uses_the_family_and_outside_does_not(self):
        fam = next(f for f in POLICY["families"] if f["kit"] == "genie3")
        scope = next(s for s in POLICY["scopes"] if s["kit"] == "genie3")
        facts = facts_for(scope)
        with tempfile.TemporaryDirectory() as tmp:
            lo, hi = fam["size_hull"]
            inside = select(POLICY, self._genie(tmp, (lo + hi) // 2), facts)
            self.assertEqual((inside.provider, inside.adapter_enabled), (fam["decision"]["provider"], True))
            self.assertTrue(inside.reason.startswith("in_family"))
            for bad in (lo - 1, hi + 1):
                out = select(POLICY, self._genie(tmp, bad), facts)
                self.assertEqual(out.provider, "anthropic")
                self.assertIn("size", json.dumps(out.near_misses))
            other_option = select(POLICY, self._genie(tmp, lo, ("--steps", "5")), facts)
            self.assertEqual(other_option.provider, "anthropic")
            self.assertEqual(select(POLICY, self._genie(tmp, lo), facts_for(scope, gpu_name="NVIDIA B300")).provider, "anthropic")

    def test_every_family_rests_on_one_provider_across_all_tested_sizes_and_a_passed_holdout(self):
        for fam in POLICY["families"]:
            self.assertEqual(fam["gpu"], "H100")
            self.assertEqual(fam["size_hull"], [min(fam["tested_sizes"]), max(fam["tested_sizes"])])
            self.assertIn("holdout passed", fam["decision"]["basis"])
            self.assertTrue(fam["evidence"]["holdout"])


if __name__ == "__main__":
    unittest.main()

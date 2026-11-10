# FoldBoost hybrid: stock, Anthropic or ours — chosen from measurements — plus a resident process

Anthropic's [optimization kits](https://github.com/anthropics/uplifting-biomolecular-modeling/tree/f4f62fa6592ae4938d49b1757bea0cfeff9f468e) speed up 36 biomolecular tools, each with its own modes (`off` = the unmodified release, `exact` = identical outputs, `fast` = documented small differences). Their README states "Status: not maintained and not accepting contributions". This independent repository adds, on top of that release:

1. **Candidate adapters** — eight small opt-in hooks that remove per-process cost (checkpoint loading, an unpickler, a Triton library fingerprint, a CIF parser, a worker start). Each is byte-identical to the file validated in its benchmark bundle and is refused at launch if a single byte differs.
2. **A router** that picks, per request, stock, Anthropic's mode or our adapter from measurements, and otherwise Anthropic's mode unchanged.
3. **A resident-process runner** (`serve_kit`) that calls a kit's own CLI function repeatedly in one process, so imports, lever installation and CUDA initialisation are paid once. It changes no model code; outputs must equal a fresh CLI's byte for byte.

The full comparison, with every number read from the archived results, is `BENCHMARK_REPORT.html` in this folder's release notes (`tools/build_report.py` regenerates it).

## What the data say (read the limits too)

- **Fresh CLI, stock vs Anthropic's mode vs ours (H100, whole fresh process, 18 discovery inputs, geometric mean of speed relative to stock):** Anthropic 1.83×, ours 2.05× (exact-mode inputs only, where outputs equal stock's: 1.80× and 1.96×). Our share on top of Anthropic's mode is +2 % to +31 % on the eight tools (1.12× on average), almost all of the rest is Anthropic's. On inputs never used for selection, every tool kept the gain. Genie3's length-288 holdout input had one 58 s stalled repetition and failed the original pre-registered rule; it passes the owner's trimmed-mean rule (fastest and slowest run dropped). Both verdicts are shown.
- **Three-way surprises:** Anthropic's *exact* mode is slower than unmodified stock for AtlasFold (all tested sizes) and gpnstar (1.31×), and no faster for colabfold. The router never assumes Anthropic is the fastest.
- **Resident process, request latency after the first request, exact mode:** OpenFold3 exact 1.45–1.58× on B300 (1.61–1.80× with our adapter inside the process) and 1.37–1.49× on H100 (two pods), OpenFold3 fast 1.60–1.82× on H100 (1.76× in a 30-request study; before the fast-mode release described below the growth limit recycled the process about every 14 requests, net 1.65× including restarts); OpenFold3-ob0 1.86–1.99× on B300 (quick look); AtlasFold 1.62–1.96× on H100 (against stock: 1.50–1.69×, because Anthropic's exact mode is slower than stock there). All outputs byte-identical to fresh CLIs. Same-input chain, OpenFold3 exact, B300, 760 tokens: stock 78.1 s → Anthropic 34.2 s → ours 32.4 s → resident 22.9 s (20.0 s with our adapter in the process).
- **Resident process does not help, or is unsafe, for:** Genie3 (1.03–1.05×, the kit starts a driver subprocess per request), ESM-IF1 (1.02–1.05×, a worker process per request), colabfold (≈1.05×), ProteinMPNN (second request refused by the kit: "one mode per process"), gpnstar (latency and memory grow every request), Enformer, Enformer-DeepMind, Borzoi and Evo2 (no derivable launcher or subprocess design). The router sends them to fresh CLIs.
- **B300:** OpenFold3 exact is measured (stock 78 s, Anthropic 34 s, ours 32.4 s at 760 tokens) — the kit's own B300 config says "Records on this card: none yet". Anthropic's *fast* mode is refused by the kit on B300 (`fold:v2:arch-sm103-not-measured`). AtlasFold cannot run there (its pinned Triton does not know sm_103a).
- **Not improved:** a single fresh CLI of most kits is limited by library import and CUDA start-up that no adapter removes (pre-compiling the kit's bytecode changed nothing: 31.4 s → 31.3 s). Of the 36 kits, 8 have a three-way measurement with holdout, 27 a quick-look (stock, Anthropic, resident; no adapter) and one (Protenix v2) could not be measured because the weights are not accessible; most quick-look speed-ups belong to Anthropic's mode, not to our adapters, and several depend on the kit's persistent JIT cache being warm (see [docs/KIT_FINDINGS.md](docs/KIT_FINDINGS.md)). **"Faster in every field" is not shown by this work.**

## The selection rule

A request is in scope only if the kit, GPU, requested mode, input, every matched option, GPU name, torch/triton/python versions, listed preconditions and weight digests equal a recorded exact scope — or, for kits whose unseen-input holdout passed, a *family* scope: same options except the listed free ones (seed, chain, runner-yaml seed) and an input size inside the tested hull (design length, polymer tokens, chain residues). Inside a scope:

1. Candidates are the providers whose outputs equalled Anthropic's on every repetition. Exact scopes: stock, Anthropic, ours. Fast scopes: Anthropic and ours.
2. A candidate is *dominated* if another is faster with 95 % one-sided paired-bootstrap confidence (exact scopes) or by ≥ 2 % in trimmed means (holdout-based choices).
3. The first undominated candidate in the order Anthropic, stock, ours is used. The mode is never switched between exact and fast; `off` is used only inside an exact scope where stock's output was identical.

Anything else — unknown input, other seed, other GPU, other mode (`big`), unrecognised flag, unverified weights, unmet precondition — selects Anthropic's mode with the requested mode unchanged. `--mode off` is always honoured.

With `--server SOCKET` the router first offers the request to a resident process, but only for kit/mode/GPU combinations the policy lists as *supported*, only when the server holds the same configuration (otherwise it refuses with status 75), and it falls back to a fresh CLI when the server is absent, refuses, or returns a non-zero exit.

## Use

```sh
python -m foldboost verify
python -m foldboost coverage
python -m foldboost install-gate --python /opt/venv/bin/python        # once per environment
python -m foldboost select --kit genie3 --kit-dir /kit/genie3 --python /opt/venv/bin/python \
    --weights /weights/genie3 -- design --config h100 --mode exact --input in.yaml --out_dir out ...
python -m foldboost run    ...same arguments...      # routes, runs, logs foldboost-logs/decisions.jsonl

# resident process (only for the kits the policy lists as supported)
python -m foldboost serve start --kit openfold3 --kit-dir /kit/openfold3 --socket ~/.fp/of3.sock -- \
    pred --config h100 --mode exact --det 1 --query-json warm.json ... --output-dir warm-out
python -m foldboost run --server ~/.fp/of3.sock --kit openfold3 ... -- pred --config h100 --mode exact ...
python -m foldboost serve status|stop --socket ~/.fp/of3.sock
```

The resident server releases cached CUDA memory after each request, recycles itself after 500 requests or when reserved GPU memory grows 6 GiB beyond its post-warm-up baseline (the request that reaches a limit is answered normally, then the process exits and its supervisor starts a fresh one; requests meanwhile fall back to fresh CLIs), and serves one request at a time on an owner-only Unix socket. With `--server`, a request whose policy choice is our adapter but whose fresh fallback finds no interpreter gate runs Anthropic's unmodified path instead of failing (recorded as `downgraded`); decision records of resident requests carry `executed_provider`, because the resident process runs the kit's own path.

**Resident memory.** On H100 the kit's `of3_transition._PACKS` cache (per-module packed weights, never freed) grew GPU memory by ~97 MiB per request in a process that builds a model per request (+97 % over requests 10–30); upstream OpenFold3 in a loop was flat. `serve_kit.install_leak_guards` frees each pack when its module dies (the kit's file is not edited): +0.27 % on OpenFold3, outputs byte-identical, speed kept. OpenFold3-ob0 has the same cache and also keeps atom-level state in `exact` mode, so with 0.2.2 it gets the release hooks in both modes: on H100 the guard alone left +92 % over requests 10–30 (405 MiB per request), the hooks make it flat (741 MiB `exact`, 1,005 MiB `fast`, outputs identical, 1.85× / 1.68–1.94× faster than a fresh CLI). The policy still lists resident OpenFold3-ob0 for B300 only until it has a router-level holdout on H100. `fast` mode: `serve_kit.release_kit_levers` calls the kit's item-boundary release hooks after every fast request (they never run in a process that builds a model per request) and clears two caches that form reference cycles through module-global registries (`trimul_native`, `atom_window._WR`): flat GPU memory over 30 requests (761 → 941 MiB, 4.5 MiB per request instead of ~625), outputs identical, 1.76× vs a fresh CLI. See [docs/KIT_FINDINGS.md](docs/KIT_FINDINGS.md) and the kit patch in `docs/kit-patches/`. A 60-request mixed-shape run on H100 with the growth limit forced low restarted the server three times (5 requests ran as fresh CLIs meanwhile); requests during a restart fell back to fresh CLIs, and the 56 outputs compared with the fresh reference were all identical (4 of the 60 were not compared before the pod was released).

## Files

- `foldboost/` — router, request parser, sizes, adapter isolation, launcher, resident runner and its controller; `data/policy-v1.json` (digest-checked; exact scopes, family scopes, resident table), `data/adapters/` (verbatim candidate files; their internal names such as `flashpair_bootstrap` and `FLASHP_*` are part of the measured bytes and are kept).
- `tools/` — `build_policy.py`, `build_policy_v2.py`, `analyze_holdout.py`, `analyze_serve.py`, `build_report.py`, `pod_ctl.py` (RunPod controller), `phase_timeline.py`.
- `HOLDOUT_PLAN.json` — plans written before measurement and the amendments, including the owner's later change of the verdict rule.
- `docs/KIT_FINDINGS.md`, `docs/kit-patches/` — observations about the pinned kits with their evidence level, and a patch proposal for the kit.
- `tests/` — 43 CPU tests: routing, tamper detection, adapter isolation, launcher logging and fallback, resident runner (sockets, recycling, leak guard). They do not prove GPU speed; the GPU end-to-end runs are in the work folder (`work/holdout/e2e*`, not in this repository).

## Attribution

Kits, modes and the common runtime are Anthropic's (Apache-2.0); our adapters only wrap them at import time. Nothing here is affiliated with or endorsed by Anthropic.

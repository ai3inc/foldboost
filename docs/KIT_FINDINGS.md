# Findings about Anthropic's optimization kits (pinned commit f4f62fa)

Observations from measuring the kits next to the unmodified releases, collected while building the hybrid router. The kit repository states
"Status: not maintained and not accepting contributions", so these are documented here instead of reported upstream. Each item names its
evidence: **formal** = three-way measurement with a pre-registered rule and a holdout, **study** = a dedicated, pre-planned study, **quick-look**
= small example input, 3-5 repetitions, no pre-registration. Nothing here is a claim about inputs that were not run.

## 1. A resident process exposes a leak in `of3_transition` (OpenFold3, OpenFold3-ob0) — study

`openfold3_opt/of3_transition.py` (and the identical `openfold3_ob0_opt/of3_transition.py`) keeps packed transition weights in a module-level
dict `_PACKS`, keyed by `id(module)`, and never removes entries. A fresh CLI builds one model and exits, so this is invisible. A process that
serves several requests builds a new model per request: on an H100 in `exact` mode 59 packed-weight sets (about 90 MiB of GPU memory) stay
alive per request.

- Upstream OpenFold3 called in a loop in one process: flat (663 -> 663 MiB over 15 requests).
- Kit loop without our server wrapper: +97 MiB per request; with the wrapper: the same slope. The wrapper adds nothing.
- Attribution: all of the growth lives in live PyTorch tensors (93% reachable from Python, no growth outside the allocator, no CUDA graph objects);
  each pack is 12 live storages and the count grows by exactly 59 packs x 12 per request.
- On B300 the same study is flat: the provider has no tuned cell there and declines before packing.
- Fix on our side, without editing the kit: `foldboost.serve_kit.install_leak_guards` wraps `_packed` and registers
  `weakref.finalize(module, _PACKS.pop, id(module), None)` once per module (the kit's own trimul cache does the same on its owner tensor).
  nvidia-smi memory: +97% -> +0.27% over requests 10-30, 30/30 outputs byte-identical, resident speed kept (1.44x in the isolated study, 1.45x through the router on the released code).
- The equivalent patch for the kit file is in `docs/kit-patches/of3_transition_packs.diff`.
- OpenFold3-ob0 carries the same cache (`openfold3_ob0_opt/of3_transition.py`) and the guard is applied to it, but on its own it is **not enough**: on H100, `exact`, 30 requests, whole-GPU memory grew +109% (518 MiB per request) without the guard and +92% (405 MiB per request) with it, outputs identical and 1.86x lower latency in both. A census names the next carrier: `opt_core.of3_sampler.atom_hoist._INV_DICTS`, a module-level list that keeps every block's dictionary of hoisted CUDA tensors (entries 48 -> 126 and 763 -> 2,605 MiB of kept CUDA memory over 13 requests of the same input); a scratch guard on it lowered the slope from 405 to 222 MiB per request (+69%, not enough), and the rest is not yet attributed. Growth depends strongly on the input (about 176 MiB per request for 3 chains, about 1,282 MiB for 9 chains). Resident latency stays 1.86x lower than a fresh run (2.1x-2.6x vs unmodified). Resolution (0.2.2, `RELEASE_MODES`): ob0 also needs the same release hooks as `fast` mode in its `exact` mode (its exact mode keeps the atom hoist and other per-model state alive as well). H100, 30 requests, pre-registered rules: `fast` 1,005 -> 1,005 MiB (0.0%, one distinct value), `exact` 741 -> 741 MiB (0.0%), all outputs byte-identical to fresh CLIs, 1.94x/1.68x (`fast`, 3/9 chains) and 1.85x (`exact`) faster than a fresh CLI; the unmodified control arm reproduced +91.7%. OpenFold3 `exact` needs only the `of3_transition` guard. The router's policy keeps resident OpenFold3-ob0 on B300 only; H100 needs its own router-level holdout.
- `fast` mode has more lever state that survives a request (CUDA-graph generations, pair cache, trunk graphs, rollout memo, atom invariants, trimul native packs): +300 to +650 MiB per
  request on H100. Three causes: the kit's own item-boundary release hooks are never called in a process that builds a model per request; the `trimul_native` call memo holds the owner
  Parameter strongly while `weakref.finalize` (whose registry holds that cache) is meant to evict it, a cycle through a module-global registry, so the finalizer never fires; and
  `atom_window._WR` keeps a TF32 copy of each weight in a dict whose values hold the parameter. `serve_kit.release_kit_levers` calls the hooks and clears those two caches after each fast
  request: GPU memory 761 -> 941 MiB, flat over 30 requests (4.5 MiB per request instead of ~625), 30/30 outputs byte-identical, 1.76x vs a fresh CLI. The first attempt of the
  pre-registered rule failed at +20.7% (a hook looked up by a module name that did not exist); the second passed. The two-point rule is noisy because memory alternates with the request size
  (8/10/12 chains), so the slope (60 requests: -0.7 MiB per request) and the set of distinct values are the stronger evidence.

## 2. Anthropic's `exact` mode is slower than the unmodified release in some cases — formal and quick-look

| Tool | stock / Anthropic exact | Evidence |
|---|---:|---|
| AtlasFold (all tested sizes, H100) | 0.90x | formal; our adapter is statistically equal to stock, the router picks stock |
| gpnstar | 0.76x (stock 1.31x faster) | quick-look |
| enformer_deepmind, one call per fresh process | 0.54x | quick-look; steady state inside one process is 4.9x faster |
| enformer, one call per fresh process | 0.97x | quick-look; steady state 1.44x faster |
| flashzoi | 0.98x | quick-look |
| esmc | 0.88x - 0.91x | quick-look |

The kits' gains accrue over repeated calls in one process (their README says so for the Enformer family); a single fresh CLI pays graph
capture and start-up.

## 3. Persistent JIT caches decide the first-run result — quick-look

af3_jax, af3_torch, af2ig and colabdesign: `exact`/`fast` are 1.7x-6x faster than stock only with the kit's persistent JIT cache warm. A cold-cache first run
equals stock (af3_jax 53.7 s vs stock 54.4 s; af2ig 110.1 s vs 109.9 s). Stock has no such cache and recompiles every run. colabdesign: `exact` is 2.2x-3.1x faster than stock with the persistent cache warm and outputs byte-identical,
but about 95% of that is compilation (first model call about 87 s per stock run vs about 10 s warm; time per design step 0.322 s vs 0.306 s, 1.05x); a cold-cache
first `exact` run is 1.12x. `fast` is slower than
`exact` for af3_torch and rfdiffusion3. mosaic's first `fast` run per input shape took 339-372 s (compile), longer than stock's 214-228 s, then 75 s.

## 4. `exact` was not bit-identical to stock in one environment — quick-look

ef2inv: the README says `--det 1` gives bit-identical results to stock. On an H100 pod (driver 595, flash-attn/TransformerEngine wheels built
locally) the loss trajectory differs from the first step (intra_contact_loss 4.5659 vs 4.5637) and the final designs differ; repeats inside each
arm are identical and the final critic ipTM is 0.91-0.95 in both. Cause not investigated; the router does not route ef2inv.
e1 `exact` equals stock only with `--det 1`; without it stock repeats differ among themselves. chrombpnet `exact` equals stock `--det 1` on all data files.

## 5. Resident-process hazards by kit — study and quick-look

| Kit | Observation |
|---|---|
| OpenFold3 (H100) | leak of section 1, fixed by the guard (+0.27%) |
| OpenFold3-ob0 (H100) | 1.85x-1.9x lower latency in a resident process; +109% GPU memory over requests 10-30 (+92% with the section 1 guard), flat (0.0%) with the release hooks in both modes (0.2.2); not yet in the router's resident table for H100 |
| esmfold2 | reserved GPU memory 14 -> 34 -> 41 -> 61 -> 68 GB over five requests; the fifth died out of memory on an 80 GB card (CUDA graphs rebuilt per request) |
| flashzoi | the kit refuses the second in-process request (exit 3, partial activation); about 8.7 GB more per request |
| opendde | about 3 GB more GPU memory per request; the server hit its limit after four requests |
| rfdiffusion3 | about 0.9 GB more per request |
| gpnstar | latency 8.8 s -> 46.8 s and growing memory over requests |
| ProteinMPNN, Protenix | the kit refuses a second request ("one mode per process") |
| Genie3, ESM-IF1, colabfold, RFdiffusion 1, BoltzGen, RoseTTAFold3, chai1, af3_jax, af3_torch, af2ig, mosaic | the kit starts a child process per request; resident gives about 1.0x |
| pxdesign | resident 1.11x-1.13x lower latency than a fresh exact CLI (vs unmodified: 1.28x on one input), GPU memory flat at 1427 MiB over 20 requests, 21 of 21 resident outputs byte-identical (two inputs, two quick-looks on two pods). Not added to the router's resident table because that table only holds configurations that passed a pre-registered study; a router entry needs a proper holdout |

## 6. Other environment facts

- B300 (sm_103): esmc fails (the prebuilt `flash_attn` has only sm_80/sm_90 kernels); complexa and opendde fail (pinned torch 2.7.x+cu126 stops at
  sm_90); ef2inv fails (flash-attn and transformer_engine built for compute capability 9.0, and the kit's `check` passes anyway); e1 and evo2 need a
  CUDA 12.9 `ptxas` for Triton; AtlasFold cannot run (pinned Triton 3.3.1 does not know sm_103a); OpenFold3 `fast` is refused by the kit
  (`fold:v2:arch-sm103-not-measured`).
- esmc: the Hugging Face repository `biohub/ESMC-300M` has a newer default revision than the one the kit pins (`a59b831...`); a fresh install on
  the default revision fails the pin check.
- BoltzGen with a typical design profile (exact, `--diffusion_batch_size 2`, 8 designs, binder length 80, the kit's public PD-L1 input, H100): `exact` is 1.43x faster than stock;
  `fast` is 12% slower than `exact`; batch size 8 is only 3.7% faster in wall time (below the 5% rule) and its design_to_target_iptm median is lower in
  a post-hoc comparison. No reason found to change `exact`, batch 2.

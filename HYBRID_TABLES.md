# Three-engine table (H100, whole fresh CLI, discovery inputs)

Policy digest `7d4ba0b5713c765c`. Status: discovery-frozen; NOT validated on a holdout; routing outside the listed scopes always selects anthropic.

Mean seconds of 5 timed fresh CLI processes per provider (1 excluded warmup). `A/ours` is Anthropic's mode time divided by ours; the bound is the one-sided 95% paired-bootstrap lower bound. `stock` is the unmodified release (`--mode off`); it is in the correctness group only where its outputs were identical to Anthropic's exact mode.

| tool | mode | input | stock s | Anthropic s | ours s | A/ours | lower95 | stock in group | router choice |
|---|---|---|---:|---:|---:|---:|---:|:--:|---|
| openfold3_ob0 | fast | 6 | 44.19 | 31.97 | 24.85 | 1.288 | 1.247 | no | **ours** |
| openfold3_ob0 | fast | 7 | 46.94 | 33.31 | 25.00 | 1.331 | 1.273 | no | **ours** |
| openfold3 | fast | 8 | 95.37 | 28.36 | 24.57 | 1.154 | 1.114 | no | **ours** |
| openfold3 | fast | 9 | 97.44 | 28.60 | 25.48 | 1.122 | 1.110 | no | **ours** |
| openfold3 | exact | 10 | 88.05 | 36.20 | 33.67 | 1.075 | 1.047 | yes | **ours** |
| openfold3 | exact | 11 | 93.22 | 37.95 | 36.16 | 1.049 | 1.037 | yes | **ours** |
| boltz2 | fast | 1 | 42.95 | 22.51 | 20.90 | 1.077 | 1.057 | no | **ours** |
| boltz2 | fast | 2 | 43.69 | 23.18 | 21.52 | 1.077 | 1.066 | no | **ours** |
| proteinmpnn | exact | 0 | 6.04 | 3.48 | 3.21 | 1.084 | 1.063 | yes | **ours** |
| proteinmpnn | exact | 1 | 7.86 | 3.84 | 3.61 | 1.066 | 1.021 | yes | **ours** |
| atlasfold | exact | 228 | 13.82 | 15.78 | 13.66 | 1.155 | 1.129 | yes | **stock** |
| atlasfold | exact | 304 | 15.42 | 16.64 | 14.65 | 1.136 | 1.125 | yes | **ours** |
| esm_if1 | fast | 0 | 9.23 | 7.22 | 6.34 | 1.139 | 1.127 | no | **ours** |
| esm_if1 | fast | 1 | 11.36 | 7.69 | 6.96 | 1.107 | 1.071 | no | **ours** |
| protenix_v1 | exact | 0 | 70.84 | 21.69 | 20.96 | 1.034 | 1.017 | yes | **ours** |
| protenix_v1 | exact | 1 | 82.09 | 29.91 | 29.24 | 1.023 | 1.009 | yes | **ours** |
| genie3 | exact | 0 | 28.15 | 18.23 | 15.78 | 1.156 | 1.127 | yes | **ours** |
| genie3 | exact | 1 | 29.20 | 19.63 | 17.42 | 1.127 | 1.102 | yes | **ours** |

Findings that need the three-way view:

- `atlasfold.h100.exact.037.228`: Anthropic's exact mode is slower than unmodified stock here (stock/Anthropic bound 1.126); our adapter beats Anthropic but is statistically tied with stock (ours/stock bound 0.994), so the rule picks stock.

# Coverage of the 36 kits x 2 GPUs

| status | rows |
|---|---:|
| candidate_rejected | 1 |
| not_measured | 61 |
| scoped_evidence | 8 |
| weights_access_blocked | 2 |

Total 72. `scoped_evidence` means the listed inputs only; every other row routes to Anthropic.

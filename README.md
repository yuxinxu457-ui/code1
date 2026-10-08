# Closing Loops the Map Has Forgotten: A Fixed-Size Neuromorphic Place Memory for Bounded-Memory SLAM

Code and results for the paper (authors anonymized for double-blind review; manuscript under review).

A bounded symbolic pose-graph window (Layer 1: the W = 25 most recently used places, least-recently-used
eviction) is coupled to a fixed 256 x 579 synaptic place-to-grid memory (Layer 2). Each place is stored with a
pattern-separated code, k = 8 random place cells per place. Its pattern is learned while the place is in the
window, consolidated once when the place is evicted, and read-only afterwards. A revisit to an evicted place,
which the window alone can no longer close, is served by injecting the coincidence (elementwise minimum) of the
place's rows into the grid attractor; the attractor settles and the position is decoded from the three grid modules.
The memory size is fixed by the neuron counts and does not grow with the trajectory. This repository contains
the simulator and spiking substrate, the two-layer estimator, the simulation campaign, the result files, and
the scripts that produce every figure and supplementary table of the paper.

## Install

Python 3.14, CPU only (the code sets `JAX_PLATFORMS=cpu`).

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All commands below are run from the repository root.

## Repository layout

```
README.md, requirements.txt
src/                     bundled spiking SLAM substrate, imported as src.*
  snn_slam_system.py       sensor streams (LiveEnvironment) and the neuromorphic front end (SNNSLAMSystem)
  sparse_forest.py         2 x 2 m arena, event camera and ToF simulation; courses 'l2probe' and 'explore'
  snn_pose_cann.py         grid-cell modules (11^2 + 13^2 + 17^2 = 579 cells) and heading ring attractor
  snn_place_cells.py       place-cell network; produces the 256-bit sparse appearance key (8 active bits)
  snn_vision_*.py          vision front end: frozen CSNN (event camera + forward ToF beam) and STDP stream
  frozen_csnn_weights.msgpack  trained CSNN weights (loaded at start-up)
slam/                    the two-layer estimator
  weight_l2_slam.py        Layer 1 window, Layer 2 memory (L2WeightMemory), verifier, weight_l2_stream()
  place_graph_slam.py      SE(2) transforms and the translation-only window relaxation
  attractor_fusion.py      substrate interface (make_system) and recognition constants
scripts/
  paper2_driver.py         simulation campaign (all SLAM configurations)
  pattern_separation_pilot.py  memory capacity test (the Layer-2 memory tested in isolation)
  paper2_summarize.py      per-configuration summary of the campaign
  supplement_tables.py     Supplementary Tables S-II to S-V
  figstyle.py              shared figure style (print size, fonts, colour palette)
  make_fig1.py, fig2_cliff.py, fig3_recall.py, fig4_capacity.py, fig5_components.py   Figures 1-5
  figS1_mechanism.py       Supplementary Fig. S1 (readout of the memory at two loads)
  lifecycle_data.py, figS2_lifecycle.py   Supplementary Fig. S2 (lifecycle of every place on one run)
results/
  paper2_results.jsonl     one JSON row per SLAM run (240 runs)
  npz/                     per-run arrays: est, gt (per-keyframe estimate and ground truth), snaps, alias_list,
                           recall_rows, lost_kfs (per-closure records), wset, plgt, plxy, nullgap_list
  capacity_test.json       capacity-test rows (arm, load M, code draw)
  mechanism_examples.npz   the three example places of Supplementary Fig. S1 (rebuilt by figS1_mechanism.py if deleted)
  lifecycle_s930.npz       creation, eviction and recall times of every place on seed 930 (from lifecycle_data.py)
figures/                 figure outputs
```

## Reproducing the figures and tables

From the shipped results (each takes seconds):

```bash
python scripts/make_fig1.py          # figures/fig1_architecture.{svg,pdf,png}
python scripts/fig2_cliff.py         # figures/fig2_cliff_footprint.{pdf,png}   (window sweep; memory size vs length)
python scripts/fig3_recall.py        # figures/fig3_recall.{pdf,png}            (recall of evicted places)
python scripts/fig4_capacity.py      # figures/fig4_capacity.{pdf,png}          (capacity test; memory size N in the loop)
python scripts/fig5_components.py    # figures/fig5_components.{pdf,png}        (component dependence, forest plot)
python scripts/figS1_mechanism.py    # figures/figS1_mechanism.{pdf,png}        (readout at 256 and 640 places)
python scripts/figS2_lifecycle.py    # figures/figS2_lifecycle.{pdf,png}        (lifecycle of every place, seed 930)
python scripts/supplement_tables.py  # Tables S-II (capacity, deployed code), S-III (alternatives, code sizes), S-IV (window sweep), S-V (operations)
python scripts/paper2_summarize.py   # means with 95% CIs for every configuration, paired deployed - l1only, deletion test
```

Regenerating the results:

```bash
python scripts/paper2_driver.py                                    # full campaign, 240 runs, ~8 h
python scripts/paper2_driver.py --phases core --arms deployed --seeds 42 --out /tmp/check   # one run, ~2 min
python scripts/pattern_separation_pilot.py                         # capacity test -> results/capacity_test.json, ~30 min
python scripts/lifecycle_data.py 930                               # lifecycle of every place on one run, ~2 min
```

One SLAM run at the base length (3500 steps) takes about 105 s on a laptop CPU; the length phase runs take about
3.5 min (`*_len2`) and 7 min (`*_len4`). The driver is resumable: it skips any configuration whose row is already in
`<out>/paper2_results.jsonl` (default `results/`), so point `--out` at a new directory to regenerate from scratch.
`--phases`, `--arms` and `--seeds` select a subset. Runs are deterministic: re-running a configuration reproduces its
shipped row (every metric and every npz array) exactly, apart from the wall-clock field `secs`.

### Memory safety

A simulation process peaks at about 4.6 GB of RAM. The sensor simulation of long paths is evaluated in slices of
`SLAM_OBS_SLICE` time steps; the scripts default it to 800, and it should be kept at that value
(`export SLAM_OBS_SLICE=800`): slicing bounds memory and does not change the results. Run one JAX process at a
time; do not launch several drivers or capacity tests in parallel.

## Configurations and paper terms

Result keys are `phase:arm:capW:seed`. Seeds: 42, 153, 264, 375, 486, 597, 708, 819, 930, 1041, 1152, 1263
(the length phase uses the first three).

| Phase | Arm | Paper term |
|---|---|---|
| core | `deployed` | deployed system (pattern-separated code, k = 8, N = 256, W = 25) |
| core | `l1only` | window alone (recall of evicted places off) |
| components | `eta0` | in-window learning off (eta = 0) |
| components | `noverify` | sequence verifier off (evicted-place verifier accepts every candidate) |
| components | `nogate` | relative-displacement check off (0.20 m sequence check) |
| components | `noverify_nogate` | both off |
| components | `noconsol` | consolidating write at eviction off |
| components | `symstore` | symbolic-recall control (a bump generated from Layer 1's retained position replaces synaptic recall) |
| store | `readout` | copy control (stored patterns are direct copies of Layer 1's positions) |
| store | `poison_evicted` | deletion test (symbolic position of a place deleted at eviction) |
| cliff | `l1only_cap30/35/40/60` | window alone with window size W = 30 / 35 / 40 / 60 |
| capacity | `onehot_N256/N48/N32` | one-cell code with memory size N = 256 / 48 / 32 place cells |
| capacity | `khot_N48/N32` | pattern-separated code with memory size N = 48 / 32 place cells |
| length | `deployed_len2/len4`, `l1only_len2/len4` | ~1.8x and ~3.4x longer schedules |

Identifier conventions: `khot` = pattern-separated code, `onehot` = one-cell code, `l1only` = window alone,
`capN` = window size W = N places, `_NX` = memory size of X place cells, L1 / L2 = Layer 1 / Layer 2.

Main result fields (errors in cm): `raw` = position error (unaligned mean per-keyframe error); `ate` = ATE (after
optimal rigid alignment); `loops` = closures served; `relaxes` = in-window closures; `n_served` = closures to
evicted places served by recall; `n_lost` = revisits to evicted places lost by the window alone; `at_lost` = error
at those revisits; `before` / `after` = error at served evicted closures before and after recall; `catastrophic` =
recalls that increased the error by more than 20 cm; `l1_gate_rej` = in-window candidates rejected by the 0.25 m
metric gate; `l2_verify_rej` = evicted-place candidates rejected by the verifier; `sop_*` = Layer-2 synaptic
operations by update type. In the capacity test, `prec10_all` = accuracy (recall within 0.10 m of the place),
`ident` = identification, `cat` = catastrophic rate.

## Citation

Withheld for double-blind review.

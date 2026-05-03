# SE-RGCN: Spatially-Encoded Relational Graph Convolutional Network

**Forecasting Geopolitical Edge Evolution: Geographic Periodic Priors in Relational Graph Neural Networks.**

This repository implements **SE-RGCN**, a Relational GCN that injects a non-linear periodic spatial prior — derived from Kautilya's *Mandala* theory — into a multiplex dyad-year graph of the Correlates of War (COW) and ATOP datasets. The objective is to forecast *changes* in the alliance / conflict topology (Δedge formation and dissolution), not steady-state edges, on a strict chronological OOD split (Train 1950–2000, Val 2001–2005, Test 2006–2018).

See [PROPOSAL_V4.md](PROPOSAL_V4.md) for the methodology and [REVIEWER_NOTES.md](REVIEWER_NOTES.md) for the critical statistical / architectural traps the implementation must avoid.

---

## Repository structure

```
.
├── configs/              # Hydra / YAML experiment configs
├── data/
│   ├── raw/              # Untouched downloads (COW, ATOP, CShapes 2.0). Git-ignored.
│   ├── interim/          # Cleaned but pre-feature-engineering. Git-ignored.
│   └── processed/        # Final dyad-year tensors / graph snapshots. Git-ignored.
├── docs/                 # Methodology notes, distance-matrix decision log, figures.
├── notebooks/            # Exploratory analysis only. No model training here.
├── scripts/              # Entry points: train.py, eval.py, build_distance_matrix.py.
├── src/
│   ├── data/             # Loaders, distance builder, dyad-year graph construction.
│   ├── models/           # SE-RGCN, baselines (persistence, trade-only, SGformer, HGT).
│   ├── training/         # Loop, curriculum schedules, hard-negative miners.
│   ├── evaluation/       # PR-AUC, Recall@k, Brier, lift, causal permutation probe.
│   └── utils/            # Logging, seeding, config helpers.
└── tests/                # Unit tests (esp. relation-routing tests for RGCN).
```

---

## Environment setup

The project uses **conda for environment management** and **pip for all package installations**. Hardware target: a single local GPU with **4 GB VRAM** — keep batch sizes and basis-decomposition ranks accordingly.

### 1. Create the conda environment (Python 3.10)

```bash
conda create -n mandala_env python=3.10 -y
conda activate mandala_env
```

### 2. Install PyTorch (CUDA 12.1 build)

PyTorch must be installed *before* PyTorch Geometric so the PyG wheels can match the torch + CUDA version. Adjust the `cu121` tag if your driver requires a different CUDA build (`cu118` is the common alternative).

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA is visible:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

### 3. Install PyTorch Geometric and its companion wheels

The `torch-scatter` / `torch-sparse` / `torch-cluster` / `torch-spline-conv` wheels are version-pinned to the torch + CUDA build. Use the official PyG wheel index:

```bash
pip install torch_geometric==2.5.3
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
```

### 4. Install the rest of the project requirements

```bash
pip install -r requirements.txt
```

### 5. (Optional) Log in to Weights & Biases

```bash
wandb login
```

---

## Quickstart (planned entry points — not yet implemented)

```bash
# Stage 1 — build the canonical distance matrix from CShapes 2.0.
python scripts/build_distance_matrix.py --config configs/distance.yaml

# Stage 2 — run the persistence + trade-only baselines (reviewer-mandated, week 1).
python scripts/train.py --config configs/baseline_persistence.yaml
python scripts/train.py --config configs/baseline_trade_only.yaml

# Stage 3 — train SE-RGCN.
python scripts/train.py --config configs/se_rgcn.yaml
```

---

## Roadmap

Following the reviewer-recommended sequencing in `REVIEWER_NOTES.md`:

1. **Week 1** — End-to-end CShapes 2.0 distance matrix. Lock every choice (temporal borders, maritime adjacency, trans-oceanic, split states).
2. **Week 2** — Persistence and trade-only baselines on the test set with PR-AUC / Recall@k confidence intervals. *Gate:* if the trade-only gap over persistence is small, pivot the paper before architecture work.
3. **Weeks 3–4** — Minimum SE-RGCN, full kernel, no curriculum.
4. **Weeks 5–8** — Kernel ablation grid, curriculum variants (hard-freeze vs. annealed vs. auxiliary-loss), causal permutation probe with distance-band-stratified shuffling.
5. **Weeks 9–10** — SGformer / HGT comparisons, frontier open-weight LLM evaluations.
6. **Weeks 11–12** — Writing, follow-up analyses.

---

## License

TBD.

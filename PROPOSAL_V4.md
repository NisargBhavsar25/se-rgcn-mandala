# Research Proposal (Version 4.0)

**Title:** Forecasting Geopolitical Edge Evolution: Geographic Periodic Priors in Relational Graph Neural Networks

## 1. Abstract
Current Graph Neural Networks (GNNs) for geopolitical forecasting frequently fall into the "persistence trap," predicting steady-state network topology rather than the formation or dissolution of alliances. This paper introduces the **Spatially-Encoded Relational GCN (SE-RGCN)**, designed to forecast *changes* in multiplex state systems. We translate Kautilya’s classical *Mandala* theory into a periodic spatial positional encoding, acting as a geographic inductive bias. To prevent dense economic covariates from diluting sparse military signals, we employ an RGCN architecture over dyad-year snapshots of the Correlates of War (COW) and ATOP datasets. Evaluated on a strict chronological regime-shift split (Train 1950-2000, Val 2001-2005, Test 2006-2018), we demonstrate that SE-RGCN accurately forecasts alliance formations and conflict initiations, outperforming naive persistence baselines, pure trade-gravity models, and zero-shot open-weight Large Language Models (LLMs). Through rigorous causal probing and kernel ablations, we prove the periodic geographic prior contributes structural reasoning independent of pure distance decay.

## 2. Introduction and Motivation
*   **The Persistence and Confounding Traps:** Forecasting international relations is confounded by the extreme persistence of treaties and the overwhelming predictive power of bilateral trade (the gravity model). Models evaluating steady-state accuracy often mask trivial persistence tracking or trade-proxy classification.
*   **The Contribution:** We reframe the task as temporal edge evolution (predicting $\Delta E$ rather than $E$). We introduce a non-linear periodic positional encoding based on Kautilya's *Mandala* theory, injecting classical spatial priors into a modern Relational GCN, and empirically proving its utilization via causal inference permutations.

## 3. Methodology: Positional Encodings and Relational Aggregation
We model the global state system using Politically Relevant Dyads (PRD). The *Mandala* spatial kernel is computed as a continuous positional encoding:

$$ S(d_{ij}) = \cos(\beta d_{ij}) \cdot e^{-\alpha d_{ij}} $$

To prevent the Multi-Layer Perceptron (MLP) from immediately down-weighting this feature due to bootstrap gradient failures against high-magnitude trade features, we utilize a curriculum learning strategy: freezing trade features for $N$ warmup epochs to force the network to explore the spatial prior. 

Aggregation is handled via a **Relational Graph Convolutional Network (RGCN)**, which learns relation-specific weight matrices ($W_{military}$, $W_{trade}$) to ensure the dense economic scaffolding does not wash out the sparse structural balance signal of the military layer.

## 4. Dataset and Chronological Evaluation
We utilize COW Alliance v4, COW MID, and COW Bilateral Trade, joined into directed dyad-year multiplex graphs following Maoz's (2011) conventions. 

To test true forecasting capability across global regime shifts, we employ a **Chronological OOD Split**:
*   **Train:** 1950–2000 (Cold War Bipolarity).
*   **Validation:** 2001–2005 (China WTO entry / Post-9/11 shift).
*   **Test:** 2006–2018 (Multipolarity / Post-Financial Crisis).

## 5. Experimental Setup and Baselines
The primary task is predicting edge formation and dissolution. We evaluate using AUC and F1-Macro on transition matrices.

**Tier 1: Diagnostics (The "Trap" Checks)**
1.  **The Persistence Baseline:** Naively predicting $E_{t+1} = E_t$.
2.  **The Trade-Only Baseline:** RGCN trained utilizing exclusively the trade layer.

**Tier 2: The Spatial Kernel Ablation Grid**
We swap the $S(d_{ij})$ module to isolate the Mandala claim:
1.  Full Kernel vs. Decay-Only ($e^{-\alpha d}$) vs. Periodic-Only ($\cos(\beta d)$) vs. Raw Distance.

**Tier 3: State-of-the-Art Comparisons**
1.  **Topological SOTA:** SGformer (2024).
2.  **LLM Reasoning:** Open-weight frontier models (e.g., Llama 3 70B / Qwen 2) evaluated on Anonymized vs. Named prompt settings.

**Causal Probing:** Post-training, we will permute the $S(d_{ij})$ features at inference while holding all other features (including raw distance) constant. A resultant collapse in predictive accuracy will empirically verify the model's reliance on the periodic geographic prior.
Reviewer 2 hat on, last time. V4.0 is a real paper. The framing is honest, the ablations are principled, the task reformulation kills the most dangerous critiques. I would actually recommend reviewing this favorably *if* the empirics hold. So let me earn my keep by attacking the parts that are still load-bearing assumptions.

## 1. Class Imbalance — Your Positive Class Is Catastrophically Rare And Standard Tricks Will Mislead You

You've correctly identified the structural problem. Let me put numbers on it so you internalize the scale.

**Base rate arithmetic.** Politically Relevant Dyads globally average ~1,500-2,000 per year (depending on PRD definition). Alliance formations per year in COW v4: typically 3-15 globally, mostly clustered in specific years (1949 NATO founding, 1955 Warsaw Pact, post-Soviet realignment 1991-1995). Conflict initiations (new MID dyads) per year: ~30-80, again clustered. So your positive rate is in the range of **0.1% to 1%** depending on which transition you predict. Across 1950-2000 training (~50 years), you have maybe 200-500 alliance formation events total. **That is your training set size for the positive class.** Not edges. *Events.*

This is not "imbalanced classification." This is a low-data forecasting problem dressed up as classification, and the distinction matters because most imbalance tricks fail in this regime.

**Why the standard playbook breaks here:**

*Focal loss* (Lin et al. 2017) was designed for object detection where positives are rare but exist in dense spatial structure. On 200-500 total positive events with high temporal clustering, focal loss has nothing to focus on — the model can't develop calibrated confidence on positives because there are too few.

*SMOTE/synthetic oversampling* requires that interpolation between positives is meaningful. Interpolating a feature vector between "France-Germany 1963 alliance" and "US-Pakistan 1959 alliance" produces a meaningless centroid in dyad-feature space. Synthetic minorities here are noise.

*Class-weighted cross-entropy* with a 1000:1 weight ratio gives positives dominant gradient magnitude per example, but you have so few positive examples that gradient noise from the minority class becomes the dominant training signal. The model trains on whatever specific quirks happen to be in your handful of positives. Severe overfitting.

*Negative subsampling.* This works for link prediction in dense graphs (sample $k$ negatives per positive). But your prediction task is *transitions*, not edges. Most negatives are "this stable non-alliance dyad continued not being an alliance" — which is correct but uninformative. Random negatives mostly teach the model "Burkina Faso and Bolivia stayed unallied," which it learns trivially.

**What you actually need to do:**

*Hard negative mining via near-misses.* Negatives should be dyads that *plausibly could have* formed an alliance but didn't — high-trade dyads, geographically proximate dyads, dyads with shared rivals. These are the negatives that force the model to learn discriminative features. Without this, negatives are too easy and AUC inflates artificially.

*Time-aware negative sampling.* For each positive event at year $t$, sample negatives from year $t \pm w$ (small window). This controls for temporal confounders (geopolitical era, system size). A positive in 1962 paired with negatives from 1962 forces the model to learn what was special about *this* dyad, not what was special about the 1960s.

*Frame as ranking, not classification.* Within each year, rank dyads by predicted formation probability. Evaluate with NDCG@k or recall@k where $k$ is the actual number of formations that year. This is the right metric for the actual decision problem (which dyads should an analyst watch?) and it's robust to base-rate shift across eras.

*Calibrated event-time models.* Look at Cox proportional hazards or neural survival models (e.g., DeepHit, Nagpal et al.'s deep survival work). The right framing for "when does this dyad form an alliance" is survival analysis, not binary classification per dyad-year. This is a non-trivial reframing but the right one statistically. At minimum, cite this literature and justify why you're not using it.

*Stratify by transition type.* Don't lump alliance-formation, alliance-dissolution, conflict-initiation, conflict-cessation into one task. Each has different base rates, different driving features, different temporal dynamics. Report per-transition-type metrics or you mask which transitions the model actually learns.

**The metric trap you've already half-fallen into.** You list AUC and F1-Macro. With 0.1% positive rate, AUC of 0.95 is achievable by a model that is functionally useless — the top 5% of predicted scores still contains 50x more negatives than positives. F1-Macro averages across classes and looks reasonable while masking poor positive-class precision. Use:

- *Precision-Recall AUC* (not ROC-AUC). Standard for severe imbalance (Davis & Goadrich 2006, Saito & Rehmsmeier 2015).
- *Recall@k* where $k$ matches the actual annual base rate.
- *Brier score* for calibration on the positive class.
- *Lift* over base rate at top-$k$ predictions.

Report ROC-AUC if you must, but make PR-AUC the headline. A reviewer who sees only ROC-AUC on a 0.1% positive task will (correctly) suspect you're hiding something.

**The deeper problem: alliance formation is not IID.** Most alliance formations in your dataset are *clustered events* — single geopolitical shocks (post-WWII reorganization, Soviet collapse, post-9/11 War on Terror coalition) that produce many simultaneous formations. Your model trained on 1950-2000 has learned from maybe 5-10 *underlying causal events*, not 500 independent positive examples. Cross-validation that splits dyads but not events massively overstates effective sample size. Be very cautious about confidence intervals.

## 2. Curriculum Warmup — Sound In Principle, Almost Certainly Wrong As Specified

The instinct is correct. The implementation as written will not do what you want and may make things worse.

**The fundamental tension.** RGCN learns relation-specific weight matrices $W_{\text{military}}, W_{\text{trade}}, W_{\text{spatial}}$. "Freezing trade features" can mean two different things in this architecture, and the proposal doesn't distinguish:

*Interpretation A: Zero out the trade relation entirely during warmup.* The model trains as if the trade layer doesn't exist. Then trade is added back at epoch $N$.

*Interpretation B: Keep trade features in the input but freeze $W_{\text{trade}}$ at initialization.* The trade messages flow but with random projections.

Neither does what you want.

Under (A), the model learns to predict alliance formation using only military and spatial information for $N$ epochs. The features are different, the optimal hidden representations are different, the inductive biases are different. When trade is suddenly unfrozen at epoch $N$, the model is now solving a different optimization problem with a partial initialization that has no reason to be a good starting point. *This is exactly catastrophic forgetting* — the spatial-reliant representations developed during warmup are no longer locally optimal for the new objective and will be overwritten quickly.

Under (B), you're forcing trade messages to flow as noise for $N$ epochs. This actively impairs learning. The model develops representations robust to noisy trade input, then you abruptly inject signal. Same forgetting problem.

**What actually works in the curriculum learning literature for this kind of problem:**

*Gradual unfreezing with annealing.* Don't freeze/unfreeze hard. Apply a multiplier $\lambda_{\text{trade}}(t)$ to the trade relation's contribution that ramps from 0 to 1 over many epochs. The spatial kernel develops its representations while trade information enters smoothly. This is the diffusion-model-style noise schedule applied to feature reliance.

*Auxiliary loss on the spatial prior.* Add a head that predicts edges *using only* the spatial kernel features, with its own loss term. This forces gradient signal to the kernel parameters $(\alpha, \beta)$ regardless of what the main MLP decides to do. Bengio et al.'s curriculum work and multi-task learning theory support this. You can decay this auxiliary loss over training.

*Variance-matching initialization.* Standardize features such that the kernel feature has comparable variance to other features at initialization. This is mundane but skipped in the proposal. If the kernel value has variance 0.1 and trade volume has variance 100, no curriculum saves you.

*Information bottleneck on trade.* Pass trade through a low-dimensional projection (rank 1 or 2) for the first $N$ epochs, then expand. This explicitly limits the capacity available to trade early on without zeroing it.

**The specific failure mode you should fear most.** You run curriculum warmup. Spatial kernel develops nontrivial $(\alpha, \beta)$ values during warmup because it's the only signal. You unfreeze trade. The model rapidly discovers trade is more predictive than spatial structure. Gradients to $(\alpha, \beta)$ shrink. The kernel parameters drift back toward initialization-like values because they're no longer in the high-loss-gradient regime, and any regularization (weight decay) pulls them toward zero. You finish training with a model that *looks* like it's using the kernel (because the values aren't exactly at init) but functionally has minimal kernel utilization.

Your causal permutation probe will catch this — and that's good, that's why the probe is the right experiment. But you should expect this to happen, not be surprised by it. Plan for what you do if the probe shows minimal kernel utilization despite the curriculum. Acceptable outcomes: report it honestly as "curriculum did not produce sustained utilization, suggesting the periodic prior is dominated by trade signal in this domain." That's a legitimate negative result. Unacceptable: tune the curriculum until permutation effects look big enough to publish, which is p-hacking.

**The curriculum needs ablation too.** Add to your ablation grid:
- No curriculum (joint training from epoch 1).
- Curriculum with hard freeze (your current spec).
- Curriculum with annealing (recommended).
- Curriculum with auxiliary loss (recommended).

If "no curriculum" matches "annealing curriculum" on permutation-probe sensitivity, the curriculum did nothing. If hard freeze underperforms annealing, you have evidence for catastrophic forgetting and should change the spec.

## 3. Remaining Fatal Flaws And Implementation Failure Points

Yes. Several. In rough order of how likely they are to bury you.

**Failure point #1: Distance computation will eat 2-3 weeks of your timeline.** "Minimum geographic distance between state polygons" sounds simple. It is not. CShapes 2.0 gives you historical state borders, but: (a) historical territorial changes mean dyad distances are time-varying (Germany 1989, USSR 1991, Yugoslavia 1991-2008, Sudan 2011) and you must decide whether to use temporally-correct borders or fix to a reference year; (b) maritime adjacency has no canonical definition (UNCLOS EEZs? territorial waters? closest-point-on-coast?); (c) trans-oceanic distances need great-circle vs. shipping-route choice; (d) split-state cases (Pakistan 1947-1971, US-Alaska, France-Réunion) require choices about which territory to measure from. Each of these choices materially affects $S(d_{ij})$ and therefore your results. The proposal allocates zero space to this. **Build the distance matrix first, lock it, and document every choice.** If you change distance definitions mid-experiment, all prior results are invalidated.

**Failure point #2: The trade-only baseline will probably be very strong, and you need to know this before architecture work.** Run this in week 1. If trade-only RGCN achieves PR-AUC within 10% of full SE-RGCN on edge evolution, you have a problem the rest of the architecture cannot fix. Either reframe the contribution (the spatial prior helps where trade is uninformative — show the conditional gain) or pivot to a different prediction target (conflict initiation, where trade-alliance correlation is weaker). Do not discover this in month 4.

**Failure point #3: Your test set may not contain enough positives to support claims.** 2006-2018 alliance formations in COW v4: roughly 30-60 events globally depending on coding. Across PRD-filtered dyads. After the year-by-year stratification I recommended above, you may have 2-5 positive events per test year. Statistical power to distinguish models on this is minimal — 95% CIs on PR-AUC with 30 positives are very wide (Boyd et al. 2013 on PR-AUC CI estimation). You need to either (a) include ATOP formations to enlarge the positive class, (b) extend the test period (which means waiting for more data — COW v4 already extends past 2018 in newer releases, check current version), or (c) include conflict transitions to multiply effective sample size, accepting the multi-task complication.

**Failure point #4: COW Alliance v4, ATOP, and your stated mention of both is now inconsistent.** V3.0 had ATOP; V4.0 says "COW Alliance v4 ... and ATOP" implicitly via earlier mentions but the dataset section only lists COW. COW v4 and ATOP encode different things (COW is narrower, ATOP includes consultation pacts and ententes). Pick one as primary, or fuse explicitly with documented joining rules. Reviewers will ask. Maoz 2011 actually uses a custom alliance dataset that differs from both — if you're following Maoz's conventions, specify exactly which dataset he provides and use it.

**Failure point #5: SGformer comparison is non-trivial.** SGformer is a transformer over signed graphs. It expects relatively dense graphs. On your sparse military layer (avg degree 3-5), its attention mechanism has very few entries to attend to — it degrades to a 2-layer MLP with extra parameters. Comparing your relational architecture to SGformer-on-sparse-graphs may make SGformer look bad in ways that aren't really about your contribution. Either run SGformer on the multiplex graph (and explain the adaptation) or include a fairer transformer-on-relational-graphs baseline (e.g., HGT — Hu et al. 2020). The current spec invites a critique of unfair comparison.

**Failure point #6: Llama 3 70B is fading; pick the current frontier open-weight at submission.** As of mid-2026 the open-weight landscape has moved. Use whatever is current and pin the version. Also consider: a small frontier closed-weight model (with cutoff before 2006) as additional point. Your contamination concern with named-graph LLMs is real but partially addressable by using models with documented training cutoffs that pre-date your test period — though for 2006-2018 test data, this is essentially impossible with current models. Acknowledge this limitation prominently.

**Failure point #7: The causal permutation test has a subtle confound.** When you permute $S(d_{ij})$ at inference while holding distance constant, you're creating dyads where $S$ is inconsistent with $d$. The model has only ever seen consistent (S, d) pairs in training. Performance might collapse not because the model relied on $S$ structurally, but because it's encountering out-of-distribution inputs. Mitigate by permuting *across* dyads at similar distance bands — preserve the marginal $P(S|d)$ approximately, break only the dyad-specific assignment. This is the right intervention; the naive permutation isn't.

**Failure point #8: Implementation pitfalls in PyTorch Geometric / DGL.** RGCN in PyG (`RGCNConv`) has memory issues with many relation types — basis decomposition is essential for >5 relations. DGL's RGCN handles this differently. If you use heterogeneous graph types in DGL, the directed-vs-undirected interaction with edge types can silently misroute messages. Whichever framework you use, write a unit test that confirms messages on the military layer don't cross-pollinate to trade layer outputs. This is not paranoia; it's a known footgun.

**Failure point #9: The "non-IID events" problem from section 1 also applies to validation.** Your val set is 2001-2005, which contains the post-9/11 coalition formation — a single major event driving many positives. Models tuned on this val set will be biased toward features that explained this specific event. Consider rolling-origin validation across the training period (e.g., walk-forward CV on 1950-2000) and treat 2001-2005 as a held-out tuning check rather than the primary tuning signal. Otherwise your hyperparameters overfit to one specific historical moment.

## What I'd Do If I Were You, In Order

Week 1: Build the distance matrix end-to-end. Document every choice. Lock it.

Week 2: Run two experiments, *only*. Trade-only RGCN. Persistence baseline. Get PR-AUC numbers on the test set with confidence intervals. **If the gap between persistence and trade-only is small, your task formulation is wrong and you need to fix it before any architecture work.** If trade-only is very strong, you need a conditional analysis (where does the spatial prior help over trade?) rather than headline AUC comparisons.

Weeks 3-4: Implement the minimum SE-RGCN with full kernel, no curriculum, no fancy tricks. See if it beats trade-only at all. If not, the architecture has fundamental issues and curriculum tricks won't save it.

Weeks 5-8: Ablation grid (kernel variants), curriculum variants, causal probes.

Weeks 9-10: SOTA baselines and LLMs.

Weeks 11-12: Writing, additional analyses suggested by results, addressing whatever broke.

If you discover in week 2 that the gap between trade-only and persistence is small, *that itself is a publishable finding* and you should consider pivoting the paper to "Why GNNs in international relations forecasting overstate their contribution: a careful baseline analysis." This would be a more impactful paper than another modest-improvement architecture paper, and the field needs it.

## Bottom Line

V4.0 is a paper I would review favorably given competent execution. The core architectural and evaluation choices are now defensible. The remaining risks are execution risks (distance matrix, sample size, baseline strength) rather than conceptual flaws.

The single most important thing: **do not commit to the architecture before you have run the persistence and trade-only baselines on the actual test data with proper metrics.** Everything in this proposal is contingent on those baselines leaving meaningful headroom. If they don't, no amount of methodological cleverness in the upper layers of the model will produce a credible paper.

The Mandala framing is now properly positioned: as a scientific hypothesis tested by the kernel ablation grid and causal permutation probe, rather than as a claim asserted by the architecture's existence. That's the right structure. Whether the hypothesis survives contact with the data is the actual research question, and you have set up an honest test of it. That is what a good paper looks like at the proposal stage.

Go write the code. Come back if the persistence or trade-only baselines surprise you — that's where the science actually starts.
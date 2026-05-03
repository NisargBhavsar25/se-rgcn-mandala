# Engineering Failure / Surprise Log

Per project plan section 6.4, every time something doesn't work as expected,
log it with: what was expected, what happened, hypothesis, resolution.
This becomes the limitations / lessons section of the paper.

---

## 2026-05-03 — Track B Week 1 setup

### Surprise: torch-geometric-signed-directed.SGCN is graph-bound

**Expected.** Use the package's `SGCN` class directly per the project plan's
"use authors' official implementations where available" policy.

**Observed.** The package's `SGCN(num_nodes, edge_index_s, ...)` constructor
takes the signed edge_index at __init__ time, precomputing graph-specific
normalization. This is fine for the canonical static-graph signed-network
benchmarks (Bitcoin, Wiki-RfA, etc.) the package targets, but is
incompatible with our temporal multi-graph training where weights must
share across yearly graphs.

**Hypothesis.** The package was designed assuming the standard signed-graph
benchmark suite, all of which are single-graph. Our temporal task is outside
its design scope. Subclassing to override the per-graph parts would be
hackier than reimplementing.

**Resolution.** Reimplement SGCN from Derr et al. 2018 with a graph-agnostic
forward pass (graph passed at forward time). Architecture (signed first/deep
convs, balance theory aggregation) is faithful to the paper; only the
wiring changes. Documented in src/baselines/sgcn.py docstring with
deviation explained. The package is cited as comparison reference.

**For paper:** mention this in the limitations section. "We reimplemented
SGCN's message passing with a graph-agnostic forward pass to support
temporal multi-graph training; the architecture (signed first/deep
convolutions, balance-theory aggregation) is faithful to Derr et al. 2018,
verified against the public reference implementation
torch-geometric-signed-directed.nn.SGCN."

---

## 2026-05-03 — make_dyad_tensors signature change missed second call site

**Expected.** Updating make_dyad_tensors() to take dyad_feature_cols would
update both call sites (train + eval).

**Observed.** I only updated the call site in train() (line 224). The
parallel call in score_split() (line 305) was missed. The runner trained
SGCN/as_published for the full 20 epochs (~3 min) before crashing at the
val-eval step with TypeError: missing 1 required positional argument.

**Hypothesis.** I trusted that grepping for the function name alone would
catch all call sites, but I was only inspecting the train() call when I
made the edit and didn't double-check that the second call already had
the new argument.

**Resolution.** Updated the missed call site. Lesson: when changing a
function signature, search by function-name regex AND visually inspect
each call site, even if it looks identical.

**Defensive test that would have caught this:** end-to-end smoke test of
the runner on a tiny synthetic dataset (3 train years, 1 val year, 1 test
year, 1 epoch, 1 baseline, 1 config). Should add to tests/. Pre-registering
as a TODO.

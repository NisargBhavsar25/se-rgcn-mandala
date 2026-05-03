# MID 5.0 -> Dyadic Conflict Onsets — Locked Rules

The COW MID 5.0 release does not include a pre-built dyadic file. We derive
dyad-year conflict onsets from `MIDB 5.0.csv` (state-dispute participants).
This document records the locked derivation rules.

**Status:** locked 2026-05-03. Changing any rule below changes the conflict
positives count, the censoring set, and every downstream PR-AUC. Pair any
change with a re-run of `tests/test_mid.py`.

**Source citation:** Palmer, McManus, D'Orazio, Kenwick, Karstens, Bloch,
Dietrich, Kahn, Ritter, Soules — *The MID5 Dataset, 2011-2014: Procedures,
coding rules, and description*. Conflict Management and Peace Science (2022).

---

## Decision M1 — Granularity: dispute-level (MIDB), not incident-level

**Choice.** Use MIDB (one row per dispute participant). Ignore MIDI
(incident-level) and MIDIP (incident participants).

**Rationale.** A single dispute can have many incidents; using MIDI would
multiplicatively over-count. The "did a conflict start between i and j in
year y" question is naturally dispute-level.

---

## Decision M2 — Onset coding: `styear` only is positive

**Choice.** A dyad-year (i, j, y) is a positive ONSET only if the dispute's
`styear` equals y for both participants. Years where the dispute is ongoing
(strictly between `styear` and `endyear`, or equal to `endyear`) are
**censored**, not negative.

**Rationale.** This is the analog of "alliance formation" -- predicting the
year of onset, not all years a conflict exists. Including ongoing-dispute
years as candidates double-counts and biases the base rate downward; including
them as negatives is even worse because the model is "right" to predict
nothing-new-starts when there's already a conflict, which is uninformative.

The censoring rule is what avoids the **duration trap** the staff feedback
explicitly called out.

---

## Decision M3 — Hostility threshold: `hostlev >= 3` (use of force)

**Choice.** Filter at the dispute level. Keep disputes whose maximum
participant `hostlev` is >= 3 ("use of force"). Drop levels 1-2 (threats,
display of force) by default. Configurable via
`derive_mid_onsets(hostility_threshold=...)` for sensitivity analysis.

**Rationale.** Levels 1-2 include verbal threats and demonstrations that are
barely conflict signal -- the literature standard for "militarized conflict"
is hostlev >= 3. Sensitivity to this threshold is a paper-grade ablation.

---

## Decision M4 — Originators only

**Choice.** Keep only rows with `orig == 1`. A dyad is an onset dyad iff both
participants are originators on opposing sides.

**Rationale.** Joiners enter via different dynamics (alliance-pull,
bandwagoning, multilateral coalitions). Their dynamics conflate the
prediction target. Originator-only is the cleaner test of "could a
contiguous / hostile dyad be predicted to start a new conflict?"

Configurable via `derive_mid_onsets(originators_only=False)` for
sensitivity / extension analysis. Expect joiner-inclusive variants to grow
the positive count by ~30-50% but introduce noise that suppresses model
discriminative power.

---

## Decision M5 — Translation at onset year, self-loop drop

**Choice.** COW -> GW translation is applied per the rules in
`gw_cow_mapping.py`, using the **onset year** as the time index. Self-loops
created by translation (e.g., COW 255 + COW 260 both -> GW 260 in some
historical encodings) are dropped with a logged count.

**Rationale.** Same logic as the alliance loaders. The onset year is the
correct time index for translation because it is the only year for which the
positive label is emitted; using mid-dispute years could pick up a different
GW code if a state's coding changed across the dispute span (e.g., a
Yugoslavia successor scenario). Onset year is the unambiguous reference.

---

## Decision M6 — PRD interaction documented as retention rate

**Operational, not a methodological choice.** Almost all originator MIDs are
PRD by definition (contiguity or major-power status drives most disputes).
The `scripts/run_persistence_baseline.py` runner reports PRD retention as a
sanity check. Expected retention: ~90%+ of onsets. Any retention below 80%
is a flag that the PRD definition or the GW-COW mapping is dropping
meaningful events; investigate before continuing.

---

## Output schema

`derive_mid_onsets` and `load_mid_onsets` return:

| column | dtype | semantics |
|---|---|---|
| dispnum | int | COW MID dispute number |
| onset_year | int | year the dispute started for both originators |
| end_year | int | year the dispute ended (last participant's `endyear`) |
| gwcode_i | int | smaller GW code in the dyad (canonical) |
| gwcode_j | int | larger GW code in the dyad |
| hostlev_max | int | maximum hostlev across the two participants (>= threshold) |

`build_mid_onset_edge_table` joins this onto a PRD universe and emits
two binary columns per dyad-year:

| column | semantics |
|---|---|
| edge_present | 1 iff year == onset_year for some dispute on this dyad |
| censored | 1 iff year is between onset_year+1 and end_year inclusive |

Censored rows are excluded from prediction (in the runner) and from training
(when the trade-only RGCN baseline is built).

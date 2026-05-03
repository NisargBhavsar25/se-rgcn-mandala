# Distance Matrix — Locked Design Decisions

This document records the four design decisions made for the CShapes 2.0 →
dyad-year geodesic distance matrix that feeds the Mandala spatial kernel
$S(d) = \cos(\beta d) \cdot e^{-\alpha d}$.

**These decisions are LOCKED.** Changing any of them changes every $S(d)$
value, which shifts the learned $(\alpha, \beta)$, the trade-only and full-model
PR-AUC numbers, the kernel-ablation deltas, and the causal permutation probe's
baseline. A change must be paired with:

1. An update to this document.
2. A re-run of `tests/test_distance_builder.py` (pinned distances).
3. A full re-run of every experiment that has consumed $S(d)$ to date.

Date locked: **2026-05-03**.

---

## Decision 1 — Time-varying borders, mid-year as-of date

**Choice.** Use CShapes 2.0's native state-period rows (`gwsdate` / `gwedate`
are full timestamps). For each year $t$ in the study window, select the
state-period row valid at the **mid-year as-of date June 30 of year $t$** and
compute distances only between states with a row valid at that date.

**Rationale.**

- The proposal's headline claim is forecasting *regime shifts*. Freezing
  borders to a single reference year would make the 1989 German reunification,
  the 1991 Soviet collapse, and the 1991–2008 Yugoslav dissolution invisible to
  $S(d)$ — the model would observe the alliance topology shift but no
  geographic shift. That is the wrong inductive bias for a Mandala-framed
  paper.
- CShapes 2.0 encodes border-change events as separate state-period rows
  (USA: pre-Alaska 1886-01-01..1959-01-02, Alaska-only 1959-01-03..1959-08-20,
  Alaska+Hawaii 1959-08-21..2019-12-31). Filtering by year alone returns
  multiple overlapping rows for any state that changed mid-year, so a
  specific as-of date is mandatory.
- Mid-year (June 30) is the locked compromise. Year-start would put Germany
  1990 as still divided (GDR exists), USSR 1991 as still extant, Sudan 2011
  as still unified — all biased toward the pre-transition state. Year-end
  would put Germany 1990 as unified, USSR 1991 as gone, Sudan 2011 as split
  — all biased toward the post-transition state. Mid-year picks no side,
  and aligns with the COW convention that the year's events are coded into
  the year regardless of when in the year they occurred.
- We are not inventing temporal logic. CShapes already did that work and is
  the canonical historical-borders dataset. Consuming its rasterization at
  a fixed mid-year date minimizes our defensible-choice surface.

**Operational.** Year-resolution distance feeds the model. The mid-year date
is configurable via `DistanceConfig.as_of_month` / `as_of_day`; do not change
without regenerating the full matrix and re-running pinned tests.

---

## Decision 2 — Maritime adjacency

**Choice.** Closest-point-on-coast between actual polygon geometries via
`shapely.ops.nearest_points`, then the geodesic distance between those two
points (Decision 3). No buffering, no territorial-waters extension, no EEZ
extension.

**Rationale.**

- Mandala is about whether one state can *project* into another's space. UK and
  France are coupled because the Channel is short, not because their EEZs
  overlap. Closest-point-on-coast captures this directly.
- EEZ (200 nm) makes every island state "adjacent" to half the hemisphere —
  too generous and reviewer-attackable.
- Territorial waters (12 nm) is computationally indistinguishable from
  closest-point-on-coast for nearly every dyad and adds extra footnotes for
  no gain.
- This is the *de facto* standard distance in the CShapes-using literature
  (Weidmann, Kuse, Gleditsch et al.). Sticking to the standard means no
  reviewer can challenge a non-canonical metric.

---

## Decision 3 — Trans-oceanic distance

**Choice.** Geodesic distance on the WGS84 ellipsoid via `pyproj.Geod`,
computed between the nearest polygon points (same primitive as Decision 2).
No shipping-route distance.

**Rationale.**

- Submarines, missiles, naval task forces, and air assets follow great-circle
  paths. Mandala is about strategic reach, not commercial logistics.
- Shipping-route datasets are non-canonical, time-varying (Suez closures), and
  would require their own decision log — recursive failure point.
- The cosine term in $S(d) = \cos(\beta d)\,e^{-\alpha d}$ is geometrically
  interpretable when $d$ is geodesic (it is a band-pass over angular separation
  on the sphere). It is *not* interpretable over shipping distance.

**Operational.** Never use `shapely.distance` on lat/lon coordinates — it
returns degree units. The builder uses `Geod(ellps='WGS84').inv(...)` on the
nearest-point pair returned by shapely, which is the only correct path on
EPSG:4326 polygons.

---

## Decision 4 — Split / non-contiguous territory

**Choice.** Minimum distance across the union of all polygons belonging to a
state. Implemented by dissolving CShapes polygons by `cowcode` per year before
distance computation; non-contiguous territories collapse into a single
MultiPolygon and `shapely.nearest_points` returns the closest pair across all
components automatically.

**Rationale.**

- The Mandala spirit is *geographic presence*, and a state's effective
  presence is the union of its territory. The US–Russia strategic relationship
  genuinely depends on Alaska's existence; capital-to-capital would erase that.
- "Largest contiguous polygon only" introduces arbitrary asymmetry no reviewer
  will accept (CONUS-only US, metropolitan-only France).
- Capital-to-capital is the worst of both worlds — simple but systematically
  wrong for any state with strategic outposts.
- Min-over-union handles split states (pre-1971 Pakistan, US, France, UK
  overseas territories) without any special-casing.

**Diagnostic.** A `d_capital_km` column is computed alongside `d_km` and
written to the parquet for sanity-checking only. It must never be consumed as
a model feature; the dual storage is for human review of dyads where
min-over-union and capital-to-capital diverge sharply. Capital coordinates are
read directly from CShapes 2.0's embedded `caplong` / `caplat` columns (no
separate capitals dataset is needed); like borders, they are time-varying via
the state-period rows.

---

## Decision 5 — State-ID system

**Choice.** Store **Gleditsch–Ward (GW) codes** in the distance matrix
(columns `gwcode_i` / `gwcode_j`). CShapes 2.0 is GW-native; there is no
`cowcode` column in the release. GW → COW translation is applied separately,
at the *join* layer when merging with COW Alliance / MID / Trade datasets,
via a maintained `data/processed/gw_to_cow.csv` translation table.

**Rationale.**

- Storing the matrix in CShapes' native coding is lossless. Pre-converting
  GW → COW at build time would silently drop or remap entities that GW codes
  but COW does not (and vice versa).
- The matrix is a single asset that may be re-used in non-COW contexts
  (ATOP joins, ICEWS joins, V-Dem joins). Native GW keeps it dataset-agnostic.
- The known GW vs COW divergences are small in count but high in importance:
  unified Germany (GW 260, COW 255), Yemen unification, Serbia / Yugoslavia
  successors. These cases must be hand-verified during the COW join, not
  buried inside the distance pipeline.
- Translation at the join layer concentrates the mapping logic in one place
  rather than diffusing it through every consumer of the matrix.

**Operational.** Maintain `data/processed/gw_to_cow.csv` with columns
`gwcode, cowcode, year_start, year_end, notes` to handle time-bounded
remappings. The pinned distance test asserts on GW codes (e.g., unified
Germany = 260) — never on COW codes.

---

## Operational details

| Concern | Decision |
|---|---|
| CRS | All geometry ops in EPSG:4326 (lat/lon). All distances via `pyproj.Geod` on WGS84. |
| Polygon hygiene | `shapely.validation.make_valid` applied to every polygon at load time (CShapes has self-intersections). |
| Distance floor | None. Adjacent dyads get `d = 0`, and `S(0) = cos(0) · e^0 = 1`, the correct max-coupling. |
| Normalization | Raw `d_km` stored in parquet. Downstream model code converts to megameters (`d_km / 1000`) before kernel evaluation, to keep initial gradients on $(\alpha, \beta)$ well-conditioned. |
| Symmetry | One row per unordered dyad (`gwcode_i < gwcode_j` enforced at write). |
| Output schema | `year` (int16), `gwcode_i` (int32), `gwcode_j` (int32), `d_km` (float32), `d_capital_km` (float32), `border_type` (category: land / maritime / oceanic). |
| Border classification | Diagnostic only. `land` = touching polygons; `maritime` = `d_km ≤ 400 km`; `oceanic` otherwise. The 400 km threshold is for reporting / EDA; not a model feature. |
| Output path | `data/processed/distance_matrix.parquet`. |

---

## Known sensitivity points (revisit if results are surprising)

- **Bering Strait (US ↔ USSR/Russia).** Whether CShapes includes the Diomede
  islands changes the US–USSR distance from ~85 km (mainland-to-mainland) to
  ~4 km (Little Diomede ↔ Big Diomede). The pinned test allows the wider
  range; if downstream results hinge on this, tighten the test and document.
- **Germany 1990.** Reunification mid-year. CShapes may code GDR (265) as
  ending and FRG (255) as continuing as unified Germany; alternate codings
  exist. Treat 1990 itself as a transition year and prefer 1991+ data when
  pinning regression tests.
- **Maritime adjacency threshold (400 km).** Used only for the `border_type`
  diagnostic. Does not affect the model. Change freely if it muddles EDA.

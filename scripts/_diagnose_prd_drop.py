"""One-off: identify why 53/244 MID onsets in 2006-2014 fall outside PRD."""

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.mid import load_mid_onsets
from src.data.prd import prd_dyads_all_years

distance = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "distance_matrix.parquet")
distance = distance[(distance.year >= 2006) & (distance.year <= 2014)]
prd = prd_dyads_all_years(distance)[["year", "gwcode_i", "gwcode_j"]]
prd_set = set(zip(prd.year, prd.gwcode_i, prd.gwcode_j))

onsets = load_mid_onsets()
onsets = onsets[(onsets.onset_year >= 2006) & (onsets.onset_year <= 2014)]

dist_lookup = distance.set_index(["year", "gwcode_i", "gwcode_j"])
dropped: list[dict] = []
for r in onsets.itertuples():
    key = (int(r.onset_year), int(r.gwcode_i), int(r.gwcode_j))
    if key in prd_set:
        continue
    info: dict = {
        "year": key[0], "gwcode_i": key[1], "gwcode_j": key[2],
        "dispnum": int(r.dispnum), "hostlev": int(r.hostlev_max),
    }
    if key in dist_lookup.index:
        d = dist_lookup.loc[key]
        info["d_km"] = float(d.d_km)
        info["border"] = str(d.border_type)
        info["reason"] = "non-PRD distance" if d.d_km > 241 else "in-distance-but-no-PRD"
    else:
        info["d_km"] = None
        info["border"] = None
        info["reason"] = "MISSING_FROM_DISTANCE"
    dropped.append(info)

print(f"In-window MID onsets:  {len(onsets):>4}")
print(f"Inside PRD universe:   {len(onsets) - len(dropped):>4}")
print(f"Dropped:               {len(dropped):>4}")
print()
print("Drop reasons:")
for reason, count in Counter(r["reason"] for r in dropped).most_common():
    print(f"  {count:>4}  {reason}")
print()
print(f"All {len(dropped)} dropped onsets:")
print(f"{'year':<6}{'i':<6}{'j':<6}{'dispnum':<10}{'hostlev':<9}{'d_km':<10}{'border':<10}{'reason'}")
for r in sorted(dropped, key=lambda x: (x["year"], x["gwcode_i"])):
    d_km = f"{r['d_km']:.0f}" if r["d_km"] is not None else "n/a"
    border = r["border"] if r["border"] is not None else "n/a"
    print(f"{r['year']:<6}{r['gwcode_i']:<6}{r['gwcode_j']:<6}"
          f"{r['dispnum']:<10}{r['hostlev']:<9}{d_km:<10}{border:<10}{r['reason']}")

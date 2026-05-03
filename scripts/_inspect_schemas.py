"""One-off schema inspection for the COW + ATOP datasets. Disposable."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"

FILES = {
    "COW Alliance dyad-year (v4.1)":
        ROOT / "cow_alliance_v4.1" / "version4.1_csv" / "alliance_v4.1_by_dyad_yearly.csv",
    "COW MIDB 5.0 (state-dispute)":
        ROOT / "cow_mid_5" / "MIDB 5.0.csv",
    "COW MIDA 5.0 (dispute-level)":
        ROOT / "cow_mid_5" / "MIDA 5.0.csv",
    "COW Trade 4.0 (dyadic)":
        ROOT / "cow_trade_4.0" / "COW_Trade_4.0" / "Dyadic_COW_4.0.csv",
    "ATOP 5.1 dyad-year (undirected)":
        ROOT / "atop_5.1" / "ATOP 5.1 (.csv)" / "atop5_1dy.csv",
    "ATOP 5.1 dir-dyad-year":
        ROOT / "atop_5.1" / "ATOP 5.1 (.csv)" / "atop5_1ddyr.csv",
}


def inspect(label: str, path: Path) -> None:
    print(f"\n=== {label} ===")
    print(f"  path: {path.relative_to(ROOT.parent.parent)}")
    if not path.exists():
        print("  MISSING")
        return
    with path.open("r", encoding="latin-1") as f:
        n_rows = sum(1 for _ in f) - 1
    df = pd.read_csv(path, nrows=3, encoding="latin-1", low_memory=False)
    print(f"  data rows: {n_rows:,}")
    print(f"  columns ({len(df.columns)}):")
    for c in df.columns:
        sample = df[c].iloc[0]
        sample_repr = repr(sample)[:60]
        print(f"    {c:30s}  {str(df[c].dtype):10s}  e.g. {sample_repr}")
    # Year coverage if a year-like column is present
    for cand in ["year", "Year", "syear", "eyear", "StYear", "EndYear"]:
        if cand in df.columns:
            full = pd.read_csv(path, usecols=[cand], encoding="latin-1", low_memory=False)
            print(f"  {cand} range: {full[cand].min()} -> {full[cand].max()}")
            break


if __name__ == "__main__":
    for label, path in FILES.items():
        inspect(label, path)

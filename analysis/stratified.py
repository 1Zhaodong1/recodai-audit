import json, statistics as st
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "results" / "data"
rows = []
for p in sorted(DATA.glob("strata_*.json")):
    rows += json.loads(p.read_text(encoding="utf-8"))
per = defaultdict(lambda: defaultdict(list))
for r in rows:
    per[(r["cell"], str(r["seed"]))][r["stratum"]].append(r["f1"])
cells = ["E1", "E2", "E3", "E4"]
strata = ["tiny", "small", "large"]
print("cell  stratum  mean  sd  n_images")
for c in cells:
    for s in strata:
        means = [st.mean(v[s]) for v in per[(c, seed)] if s in v for seed in [None]]
        means = [st.mean(per[(c, k[1])][s]) for k in per if k[0] == c and s in per[k]]
        n = sum(len(per[(c, k[1])][s]) for k in per if k[0] == c and s in per[k]) // len(means)
        print(f"{c}  {s}  {st.mean(means):.3f}  {st.stdev(means):.3f}  {n}")

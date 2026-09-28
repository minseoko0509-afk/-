"""X 만 이진, Y 는 연속(분할 배정 허용) — 개설 집합을 고르는 완화 MIP. 여러 개의 좋은 개설 집합을 뽑음."""
import json, sys
import numpy as np, highspy
from improve import cand, cust, d, h, nI, nJ, f, K, cold, TAU, V, inf

def split_mip(tl=600, forbid=()):
    sites = list(range(nJ)); Jc = [j for j in sites if cold[j]]
    keys = [(i, j, 1) for i in range(nI) for j in sites] + [(i, j, 2) for i in range(nI) for j in Jc]
    nX = nJ; ncol = nX + len(keys)
    cost = np.concatenate([f, [TAU[k] * h[k][i] * d[i, j] for i, j, k in keys]])
    hs = highspy.Highs(); hs.setOptionValue("time_limit", float(tl)); hs.setOptionValue("mip_rel_gap", 1e-5)
    hs.setOptionValue("output_flag", True)
    hs.addVars(ncol, np.zeros(ncol), np.ones(ncol)); ix = np.arange(ncol, dtype=np.int32)
    hs.changeColsCost(ncol, ix, cost)
    hs.changeColsIntegrality(nX, np.arange(nX, dtype=np.int32), np.array([highspy.HighsVarType.kInteger] * nX))
    starts, ci, cv, lo, up = [], [], [], [], []
    by_ik, by_j = {}, {j: ([j], [-K[j]]) for j in sites}
    for t, (i, j, k) in enumerate(keys):
        c = nX + t; by_ik.setdefault((i, k), []).append(c)
        by_j[j][0].append(c); by_j[j][1].append(V[k] * h[k][i])
        starts.append(len(ci)); ci += [c, j]; cv += [1, -1]; lo.append(-inf); up.append(0)
    for cols in by_ik.values():
        starts.append(len(ci)); ci += cols; cv += [1] * len(cols); lo.append(1); up.append(1)
    for j in sites:
        starts.append(len(ci)); ci += by_j[j][0]; cv += by_j[j][1]; lo.append(-inf); up.append(0)
    for S in forbid:   # no-good cut: 같은 개설 집합 금지
        cols = list(range(nJ)); vals = [1.0 if j in S else -1.0 for j in cols]
        starts.append(len(ci)); ci += cols; cv += vals; lo.append(-inf); up.append(len(S) - 1)
    hs.addRows(len(lo), np.array(lo, float), np.array(up, float), len(ci),
               np.array(starts, dtype=np.int32), np.array(ci, dtype=np.int32), np.array(cv, float))
    hs.run(); info = hs.getInfo(); sol = hs.getSolution().col_value
    X = [j for j in range(nJ) if sol[j] > 0.5]
    return info.objective_function_value, info.mip_dual_bound, X

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    forbid, out = [], []
    for r in range(n):
        obj, bd, X = split_mip(tl=900, forbid=forbid)
        print(f"#SPLIT {r}: obj {obj:,.2f} bound {bd:,.2f} open {len(X)} {[cand['name'][j] for j in X]}", flush=True)
        forbid.append(set(X)); out.append(dict(obj=obj, X=X))
        json.dump(out, open("split_sets.json", "w"))

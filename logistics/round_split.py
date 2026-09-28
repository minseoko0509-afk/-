"""분할-MIP 개설 집합 S 에서 X 고정 LP(분할 배정) → 최대 비율 센터로 반올림 → 용량 초과 수리 → 국소탐색."""
import sys, json
import numpy as np, highspy
from improve import build, cand, cust, d, h, nI, nJ, f, K, cold, TAU, V

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
S = json.load(open("split_sets.json"))[idx]["X"]
sites = sorted(S); Jc = [j for j in sites if cold[j]]
keys = [(i, j, 1) for i in range(nI) for j in sites] + [(i, j, 2) for i in range(nI) for j in Jc]
# X 고정 LP
r = build(sites, integer=False, fixX=set(sites))
print(f"X 고정 LP = {r['obj']:,.2f}")
# build() 는 Y 값을 돌려주지 않으므로 여기서 간단히 다시 풂
import highspy
nX = len(sites); pos = {j: t for t, j in enumerate(sites)}
hs = highspy.Highs(); hs.setOptionValue("output_flag", False)
ncol = len(keys)
hs.addVars(ncol, np.zeros(ncol), np.ones(ncol))
hs.changeColsCost(ncol, np.arange(ncol, dtype=np.int32), np.array([TAU[k]*h[k][i]*d[i,j] for i,j,k in keys]))
st, ci, cv, lo, up = [], [], [], [], []
by_ik, by_j = {}, {j: ([], []) for j in sites}
for t, (i, j, k) in enumerate(keys):
    by_ik.setdefault((i, k), []).append(t); by_j[j][0].append(t); by_j[j][1].append(V[k]*h[k][i])
for cols in by_ik.values(): st.append(len(ci)); ci += cols; cv += [1]*len(cols); lo.append(1); up.append(1)
for j in sites: st.append(len(ci)); ci += by_j[j][0]; cv += by_j[j][1]; lo.append(-highspy.kHighsInf); up.append(K[j])
hs.addRows(len(lo), np.array(lo,float), np.array(up,float), len(ci), np.array(st,dtype=np.int32), np.array(ci,dtype=np.int32), np.array(cv,float))
hs.run(); y = np.array(hs.getSolution().col_value)
print(f"분할 배정 LP = {hs.getInfo().objective_function_value + f[sites].sum():,.2f}")
frac = {}
for t, (i, j, k) in enumerate(keys):
    if y[t] > 1e-6: frac.setdefault((i, k), []).append((y[t], j))
nsplit = sum(1 for v in frac.values() if len(v) > 1)
print(f"분할된 항목 수 {nsplit} / {len(frac)}")
a = {ik: max(v)[1] for ik, v in frac.items()}
W = lambda i, k: V[k]*h[k][i]
load = {j: 0.0 for j in sites}
for (i, k), j in a.items(): load[j] += W(i, k)
# 수리: 초과 센터에서 (추가비용/부피)가 가장 작은 항목을 여유 센터로
for _ in range(10000):
    over = [j for j in sites if load[j] > K[j] + 1e-9]
    if not over: break
    j = max(over, key=lambda q: load[q] - K[q]); best = None
    for (i, k), jj in a.items():
        if jj != j: continue
        for j2 in sites:
            if j2 == j or (k == 2 and not cold[j2]) or load[j2] + W(i, k) > K[j2] + 1e-9: continue
            c = TAU[k]*h[k][i]*(d[i, j2] - d[i, j])
            if best is None or c < best[0]: best = (c, i, k, j2)
    if best is None: print("단순 수리 실패 → CP-SAT 로 수리"); break
    _, i, k, j2 = best; a[i, k] = j2; load[j] -= W(i, k); load[j2] += W(i, k)
json.dump(dict(split=[[int(i), int(k)] for (i, k), v in frac.items() if len(v) > 1]), open(f"split_items_{idx}.json", "w"))
tot = f[sites].sum() + sum(TAU[k]*h[k][i]*d[i, j] for (i, k), j in a.items())
print(f"반올림+수리 = {tot:,.2f}")
json.dump(dict(obj=tot, X=[int(j) for j in sites], assign=[[int(i), int(k), int(j)] for (i, k), j in a.items()]),
          open(f"rounded_{idx}.json", "w"))

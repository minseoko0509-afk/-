"""LP 완화 → 후보 축소 MIP → (열기/닫기/교체) 국소탐색. 매 단계 배정은 MIP(GAP)로 정확히 풂."""
import sys, time, json
import numpy as np, highspy
from solve_cfl import load, distances, TAU, V, evaluate

cand, cust = load("data/sites.csv", "data/customers.csv")
d = distances(cand, cust); h = cust["h"]; nI, nJ = d.shape
f, K, cold = cand["f"], cand["K"], cand["cold"]
inf = highspy.kHighsInf


def build(sites, integer=True, fixX=None, tl=60, gap=1e-5, start=None, quiet=True):
    """sites 에 있는 후보만 사용하는 모델. fixX 가 주어지면 X 를 그 집합으로 고정(=GAP)."""
    sites = list(sites); Jc = [j for j in sites if cold[j]]
    keys = [(i, j, 1) for i in range(nI) for j in sites] + [(i, j, 2) for i in range(nI) for j in Jc]
    nX = len(sites); pos = {j: t for t, j in enumerate(sites)}
    ncol = nX + len(keys)
    cost = np.concatenate([f[sites], [TAU[k] * h[k][i] * d[i, j] for i, j, k in keys]])
    lb = np.zeros(ncol); ub = np.ones(ncol)
    if fixX is not None:
        for j in sites: lb[pos[j]] = ub[pos[j]] = 1.0 if j in fixX else 0.0
    hs = highspy.Highs(); hs.setOptionValue("output_flag", not quiet)
    hs.setOptionValue("time_limit", float(tl)); hs.setOptionValue("mip_rel_gap", gap)
    hs.addVars(ncol, lb, ub); idx = np.arange(ncol, dtype=np.int32)
    hs.changeColsCost(ncol, idx, cost)
    if integer:
        hs.changeColsIntegrality(ncol, idx, np.array([highspy.HighsVarType.kInteger] * ncol))
    starts, ci, cv, lo, up = [], [], [], [], []
    by_ik, by_j = {}, {j: ([pos[j]], [-K[j]]) for j in sites}
    for t, (i, j, k) in enumerate(keys):
        c = nX + t; by_ik.setdefault((i, k), []).append(c)
        by_j[j][0].append(c); by_j[j][1].append(V[k] * h[k][i])
        starts.append(len(ci)); ci += [c, pos[j]]; cv += [1, -1]; lo.append(-inf); up.append(0)
    for cols in by_ik.values():
        starts.append(len(ci)); ci += cols; cv += [1] * len(cols); lo.append(1); up.append(1)
    for j in sites:
        starts.append(len(ci)); ci += by_j[j][0]; cv += by_j[j][1]; lo.append(-inf); up.append(0)
    hs.addRows(len(lo), np.array(lo, float), np.array(up, float), len(ci),
               np.array(starts, dtype=np.int32), np.array(ci, dtype=np.int32), np.array(cv, float))
    if start is not None:
        X0, a0 = start; v = np.zeros(ncol)
        for j in X0:
            if j in pos: v[pos[j]] = 1
        kpos = {key: nX + t for t, key in enumerate(keys)}
        for (i, k), j in a0.items():
            if (i, j, k) in kpos: v[kpos[i, j, k]] = 1
        s = highspy.HighsSolution(); s.col_value = list(v); s.value_valid = True; hs.setSolution(s)
    hs.run()
    st = hs.getModelStatus()
    info = hs.getInfo(); sol = np.array(hs.getSolution().col_value)
    if info.primal_solution_status != 2:
        return None
    X = [j for j in sites if sol[pos[j]] > 0.5] if integer else {j: sol[pos[j]] for j in sites}
    assign = {(i, k): j for t, (i, j, k) in enumerate(keys) if sol[nX + t] > 0.5} if integer else None
    return dict(obj=info.objective_function_value, bound=info.mip_dual_bound if integer else None,
                X=X, assign=assign)


def gap_cost(openset, tl=30, start=None):
    r = build(sorted(openset), fixX=set(openset), tl=tl, gap=1e-6, start=start)
    return (r["obj"], r) if r else (float("inf"), None)


if __name__ == "__main__":
    t0 = time.time()
    lp = build(range(nJ), integer=False)
    Xlp = lp["X"]; print(f"LP 완화 (strong) = {lp['obj']:,.2f}")
    supp = [j for j in range(nJ) if Xlp[j] > 1e-3]
    print(f"LP에서 X>0 인 후보 {len(supp)}곳: {[(cand['name'][j], round(Xlp[j],2)) for j in supp]}")
    # 후보 축소 MIP
    r = build(supp, tl=float(sys.argv[1]) if len(sys.argv) > 1 else 600, quiet=False)
    print(f"축소 MIP: {r['obj']:,.2f} (축소모형 내부 bound {r['bound']:,.2f})  {time.time()-t0:.0f}s")
    json.dump(dict(X=[int(j) for j in r["X"]], assign=[[i, k, int(j)] for (i, k), j in r["assign"].items()]),
              open("best.json", "w"))

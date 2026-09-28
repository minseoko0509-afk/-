"""개설 집합을 (거의) 고정하고 단일 배정을 CP-SAT 로 풂.
 - 개설 집합 S 의 센터는 X 자유(닫아도 됨), S 밖 후보도 'extra' 개까지 가까운 것 허용 가능.
사용법: python gap_cpsat.py split_sets.json idx time workers [hint.json]"""
import sys, json
import numpy as np
from solve_cfl import load, distances, TAU, V, evaluate
from ortools.sat.python import cp_model

SCALE = 1000
cand, cust = load("data/sites.csv", "data/customers.csv")
d = distances(cand, cust); h = cust["h"]; nI, nJ = d.shape
f, K, cold = cand["f"], cand["K"], cand["cold"]
w2 = {1: (2 * h[1]).astype(int), 2: (5 * h[2]).astype(int)}


def solve_gap(S, tl, workers, hint=None, tag=""):
    S = sorted(S)
    m = cp_model.CpModel()
    X = {j: m.NewBoolVar(f"X{j}") for j in S}
    Y = {(i, j, k): m.NewBoolVar("") for k in (1, 2) for i in range(nI) for j in S if k == 1 or cold[j]}
    for k in (1, 2):
        for i in range(nI):
            m.AddExactlyOne(Y[i, j, k] for j in S if (i, j, k) in Y)
    for j in S:
        m.Add(sum(w2[k][i] * Y[i, j, k] for k in (1, 2) for i in range(nI) if (i, j, k) in Y) <= int(2 * K[j]) * X[j])
    for (i, j, k), y in Y.items(): m.AddImplication(y, X[j])
    m.Minimize(sum(int(round(f[j] * SCALE)) * X[j] for j in S)
               + sum(int(round(TAU[k] * h[k][i] * d[i, j] * SCALE)) * y for (i, j, k), y in Y.items()))
    if hint:
        A = {(i, k): j for i, k, j in hint["assign"]}
        for j in S: m.AddHint(X[j], j in hint["X"])
        for (i, j, k), y in Y.items(): m.AddHint(y, A.get((i, k)) == j)
    sv = cp_model.CpSolver(); sv.parameters.max_time_in_seconds = tl; sv.parameters.num_workers = workers
    sv.parameters.relative_gap_limit = 1e-6
    st = sv.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE): return None
    Xo = [j for j in S if sv.Value(X[j])]
    a = {(i, k): j for (i, j, k), y in Y.items() if sv.Value(y)}
    fx, tr, _ = evaluate(cand, cust, Xo, a, d); tot = fx + tr[1] + tr[2]
    print(f"#GAP{tag} {sv.StatusName(st)} total {tot:,.2f} bound {sv.BestObjectiveBound()/SCALE:,.2f} open {len(Xo)}", flush=True)
    return dict(obj=tot, X=Xo, assign=[[i, k, j] for (i, k), j in a.items()])


if __name__ == "__main__":
    sets = json.load(open(sys.argv[1])); idx = int(sys.argv[2])
    tl = float(sys.argv[3]); wk = int(sys.argv[4])
    S = set(sets[idx]["X"])
    hint = json.load(open(sys.argv[5])) if len(sys.argv) > 5 else None
    r = solve_gap(S, tl, wk, hint, tag=str(idx))
    if r: json.dump(r, open(f"gap_{idx}.json", "w"))

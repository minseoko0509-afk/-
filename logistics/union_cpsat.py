"""후보를 (현재 최선해 ∪ 분할-MIP 개설 집합들)로, 각 항목의 배정 후보를 그 안에서 가까운 NN곳(+현재 배정)으로 줄여 CP-SAT.
사용법: python union_cpsat.py hint.json time workers NN [extra_sites_json ...]"""
import sys, json
import numpy as np
from solve_cfl import load, distances, TAU, V, evaluate
from ortools.sat.python import cp_model

SCALE = 1000
cand, cust = load("data/sites.csv", "data/customers.csv")
d = distances(cand, cust); h = cust["h"]; nI, nJ = d.shape
f, K, cold = cand["f"], cand["K"], cand["cold"]
w2 = {1: (2 * h[1]).astype(int), 2: (5 * h[2]).astype(int)}

hint = json.load(open(sys.argv[1])); tl = float(sys.argv[2]); wk = int(sys.argv[3]); NN = int(sys.argv[4])
U = set(hint["X"])
for s in json.load(open("split_sets.json")): U |= set(s["X"])
for fn in sys.argv[5:]: U |= set(json.load(open(fn))["X"])
U = sorted(U); A = {(i, k): j for i, k, j in hint["assign"]}
print(f"후보 {len(U)}곳", flush=True)

m = cp_model.CpModel()
X = {j: m.NewBoolVar("") for j in U}
Y = {}
for k in (1, 2):
    Uk = np.array([j for j in U if k == 1 or cold[j]])
    for i in range(nI):
        js = set(Uk[np.argsort(d[i, Uk])[:NN]].tolist()) | {A[i, k]}
        for j in js: Y[i, j, k] = m.NewBoolVar("")
by_ik, by_j = {}, {j: [] for j in U}
for (i, j, k), y in Y.items():
    by_ik.setdefault((i, k), []).append(y); by_j[j].append(w2[k][i] * y)
    m.AddImplication(y, X[j])
for ys in by_ik.values(): m.AddExactlyOne(ys)
for j in U: m.Add(sum(by_j[j]) <= int(2 * K[j]) * X[j])
m.Minimize(sum(int(round(f[j] * SCALE)) * X[j] for j in U)
           + sum(int(round(TAU[k] * h[k][i] * d[i, j] * SCALE)) * y for (i, j, k), y in Y.items()))
for j in U: m.AddHint(X[j], j in hint["X"])
for (i, j, k), y in Y.items(): m.AddHint(y, A[i, k] == j)


class CB(cp_model.CpSolverSolutionCallback):
    best = float("inf")
    def on_solution_callback(self):
        Xo = [j for j in U if self.Value(X[j])]
        a = {(i, k): j for (i, j, k), y in Y.items() if self.Value(y)}
        fx, tr, _ = evaluate(cand, cust, Xo, a, d); tot = fx + tr[1] + tr[2]
        if tot < CB.best:
            CB.best = tot
            print(f"#SOL {self.WallTime():7.1f}s total {tot:,.2f} bound {self.BestObjectiveBound()/SCALE:,.2f} open {len(Xo)}", flush=True)
            json.dump(dict(obj=tot, X=Xo, assign=[[i, k, j] for (i, k), j in a.items()]), open("best_union.json", "w"))


sv = cp_model.CpSolver(); sv.parameters.max_time_in_seconds = tl; sv.parameters.num_workers = wk
sv.parameters.relative_gap_limit = 1e-6
st = sv.Solve(m, CB())
print(sv.StatusName(st), sv.ObjectiveValue() / SCALE, sv.BestObjectiveBound() / SCALE)

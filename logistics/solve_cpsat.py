"""전체 모형을 OR-Tools CP-SAT 로 풂 (목적계수 ×SCALE 후 정수 반올림, 부피 ×2 로 정수화).
사용법: python solve_cpsat.py [hint.json] [time_limit] [workers]"""
import sys, json
import numpy as np
from solve_cfl import load, distances, TAU, V, evaluate, report   # highspy 를 ortools 보다 먼저 로드
from ortools.sat.python import cp_model

SCALE = 1000
OUT = sys.argv[5] if len(sys.argv) > 5 else "best_cpsat.json"
cand, cust = load("data/sites.csv", "data/customers.csv")
d = distances(cand, cust); h = cust["h"]; nI, nJ = d.shape
f, K, cold = cand["f"], cand["K"], cand["cold"]
J = {1: list(range(nJ)), 2: [j for j in range(nJ) if cold[j]]}
w2 = {1: (2 * h[1]).astype(int), 2: (5 * h[2]).astype(int)}          # 부피 × 2 (정수)

m = cp_model.CpModel()
X = [m.NewBoolVar(f"X{j}") for j in range(nJ)]
Y = {(i, j, k): m.NewBoolVar(f"Y{i}_{j}_{k}") for k in (1, 2) for i in range(nI) for j in J[k]}
for k in (1, 2):
    for i in range(nI):
        m.AddExactlyOne(Y[i, j, k] for j in J[k])
for j in range(nJ):
    terms = [w2[k][i] * Y[i, j, k] for k in (1, 2) for i in range(nI) if (i, j, k) in Y]
    m.Add(sum(terms) <= int(2 * K[j]) * X[j])
for (i, j, k), y in Y.items():
    m.AddImplication(y, X[j])
obj = [int(round(f[j] * SCALE)) * X[j] for j in range(nJ)]
obj += [int(round(TAU[k] * h[k][i] * d[i, j] * SCALE)) * y for (i, j, k), y in Y.items()]
m.Minimize(sum(obj))

if len(sys.argv) > 1 and sys.argv[1] != "-":
    B = json.load(open(sys.argv[1])); Xs = set(B["X"]); A = {(i, k): j for i, k, j in B["assign"]}
    for j in range(nJ): m.AddHint(X[j], j in Xs)
    for (i, j, k), y in Y.items(): m.AddHint(y, A[i, k] == j)

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 1200
solver.parameters.num_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
solver.parameters.log_search_progress = True
solver.parameters.relative_gap_limit = 1e-5
if len(sys.argv) > 4 and sys.argv[4] == "repair":
    solver.parameters.repair_hint = True
    solver.parameters.hint_conflict_limit = 100000


class CB(cp_model.CpSolverSolutionCallback):
    def on_solution_callback(self):
        Xo = [j for j in range(nJ) if self.Value(X[j])]
        a = {(i, k): j for (i, j, k), y in Y.items() if self.Value(y)}
        fx, tr, _ = evaluate(cand, cust, Xo, a, d)
        tot = fx + tr[1] + tr[2]
        if tot >= getattr(CB, "best", 1e18): return
        CB.best = tot
        print(f"#SOL {self.WallTime():7.1f}s  total {tot:,.2f}  bound {self.BestObjectiveBound()/SCALE:,.2f}  open {len(Xo)}", flush=True)
        json.dump(dict(obj=tot, X=Xo, assign=[[i, k, j] for (i, k), j in a.items()]), open(OUT, "w"))


st = solver.Solve(m, CB())
print(solver.StatusName(st), solver.ObjectiveValue() / SCALE, solver.BestObjectiveBound() / SCALE)

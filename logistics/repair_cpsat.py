"""반올림 해(용량 초과 가능)를 힌트로, 제한된 이웃에서 CP-SAT 로 실행가능·최소비용 단일배정을 찾음.
 - 후보: 반올림 해의 개설 집합 ∪ best_cpsat 의 개설 집합 ∪ 분할-MIP 집합들
 - 분할 항목: 가까운 NS곳 / 그 외 항목: 반올림 센터 + 가까운 NO곳
사용법: python repair_cpsat.py rounded_0.json split_items_0.json time workers NS NO"""
import sys, json
import numpy as np
from solve_cfl import load, distances, TAU, V, evaluate
from ortools.sat.python import cp_model

SCALE = 1000
cand, cust = load("data/sites.csv", "data/customers.csv")
d = distances(cand, cust); h = cust["h"]; nI, nJ = d.shape
f, K, cold = cand["f"], cand["K"], cand["cold"]
w2 = {1: (2 * h[1]).astype(int), 2: (5 * h[2]).astype(int)}
R = json.load(open(sys.argv[1])); split = {tuple(x) for x in json.load(open(sys.argv[2]))["split"]}
tl, wk, NS, NO = float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
U = set(R["X"]) | set(json.load(open("best_cpsat.json"))["X"])
for s in json.load(open("split_sets.json")): U |= set(s["X"])
U = sorted(U); A = {(i, k): j for i, k, j in R["assign"]}

m = cp_model.CpModel(); X = {j: m.NewBoolVar("") for j in U}; Y = {}
for k in (1, 2):
    Uk = np.array([j for j in U if k == 1 or cold[j]])
    for i in range(nI):
        near = Uk[np.argsort(d[i, Uk])].tolist()
        js = set(near[:NS]) if (i, k) in split else set(near[:NO]) | {A[i, k]}
        for j in js: Y[i, j, k] = m.NewBoolVar("")
by_ik, by_j = {}, {j: [] for j in U}
for (i, j, k), y in Y.items():
    by_ik.setdefault((i, k), []).append(y); by_j[j].append(w2[k][i] * y); m.AddImplication(y, X[j])
for ys in by_ik.values(): m.AddExactlyOne(ys)
for j in U: m.Add(sum(by_j[j]) <= int(2 * K[j]) * X[j])
m.Minimize(sum(int(round(f[j] * SCALE)) * X[j] for j in U)
           + sum(int(round(TAU[k] * h[k][i] * d[i, j] * SCALE)) * y for (i, j, k), y in Y.items()))
for j in U: m.AddHint(X[j], j in R["X"])
for (i, j, k), y in Y.items(): m.AddHint(y, A[i, k] == j)
print(f"후보 {len(U)}곳, Y 변수 {len(Y)}", flush=True)


class CB(cp_model.CpSolverSolutionCallback):
    best = float("inf")
    def on_solution_callback(self):
        Xo = [j for j in U if self.Value(X[j])]
        a = {(i, k): j for (i, j, k), y in Y.items() if self.Value(y)}
        fx, tr, _ = evaluate(cand, cust, Xo, a, d); tot = fx + tr[1] + tr[2]
        if tot < CB.best:
            CB.best = tot
            print(f"#SOL {self.WallTime():7.1f}s total {tot:,.2f} bound {self.BestObjectiveBound()/SCALE:,.2f} open {len(Xo)}", flush=True)
            json.dump(dict(obj=tot, X=Xo, assign=[[i, k, j] for (i, k), j in a.items()]), open("best_repair.json", "w"))


sv = cp_model.CpSolver(); sv.parameters.max_time_in_seconds = tl; sv.parameters.num_workers = wk
sv.parameters.relative_gap_limit = 1e-6
st = sv.Solve(m, CB()); print(sv.StatusName(st), sv.ObjectiveValue() / SCALE, sv.BestObjectiveBound() / SCALE)

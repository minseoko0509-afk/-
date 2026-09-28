"""CP-SAT 기반 LNS. 매 반복: 임의 중심 주변 고객 nc명의 두 품목 + 주변 후보 ns곳(+그 고객들의 현재 센터)을 풀어
나머지를 고정한 부분문제를 CP-SAT 로 (거의) 최적으로 풂. 개선되면 채택.
사용법: python lns_cpsat.py start.json time nc ns seed [out.json]"""
import sys, json, time, random
import numpy as np
from solve_cfl import load, distances, TAU, V, evaluate
from ortools.sat.python import cp_model

SCALE = 1000
cand, cust = load("data/sites.csv", "data/customers.csv")
d = distances(cand, cust); h = cust["h"]; nI, nJ = d.shape
f, K, cold = cand["f"], cand["K"], cand["cold"]
w2 = {1: (2 * h[1]).astype(int), 2: (5 * h[2]).astype(int)}
K2 = (2 * K).astype(int)
cst = lambda i, j, k: TAU[k] * h[k][i] * d[i, j]


def total(X, a): return f[list(X)].sum() + sum(cst(i, j, k) for (i, k), j in a.items())


def sub(X, a, I_s, J_s, tl, wk):
    Iset = set(I_s); fixed = np.zeros(nJ, int)
    for (i, k), j in a.items():
        if i not in Iset: fixed[j] += w2[k][i]
    m = cp_model.CpModel(); Xv = {}
    for j in J_s:
        Xv[j] = m.NewBoolVar("")
        if fixed[j] > 0: m.Add(Xv[j] == 1)
    Y = {(i, j, k): m.NewBoolVar("") for k in (1, 2) for i in I_s for j in J_s if k == 1 or cold[j]}
    by_ik, by_j = {}, {j: [] for j in J_s}
    for (i, j, k), y in Y.items():
        by_ik.setdefault((i, k), []).append(y); by_j[j].append(w2[k][i] * y); m.AddImplication(y, Xv[j])
    for ys in by_ik.values(): m.AddExactlyOne(ys)
    for j in J_s: m.Add(sum(by_j[j]) <= int(K2[j] - fixed[j]) * Xv[j])
    m.Minimize(sum(int(round(f[j] * SCALE)) * Xv[j] for j in J_s)
               + sum(int(round(cst(i, j, k) * SCALE)) * y for (i, j, k), y in Y.items()))
    for j in J_s: m.AddHint(Xv[j], j in X)
    for (i, j, k), y in Y.items(): m.AddHint(y, a[i, k] == j)
    sv = cp_model.CpSolver(); sv.parameters.max_time_in_seconds = tl; sv.parameters.num_workers = wk
    st = sv.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE): return None
    X2 = (set(X) - set(J_s)) | {j for j in J_s if sv.Value(Xv[j])}
    a2 = dict(a)
    for (i, j, k), y in Y.items():
        if sv.Value(y): a2[i, k] = j
    used = set(a2.values())
    return {j for j in X2 if j in used}, a2, sv.StatusName(st)


if __name__ == "__main__":
    src, T, nc, ns, seed = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    out = sys.argv[6] if len(sys.argv) > 6 else "best_lnscp.json"
    B = json.load(open(src)); X = set(B["X"]); a = {(i, k): j for i, k, j in B["assign"]}
    best = total(X, a); print(f"start {best:,.2f}", flush=True)
    rnd = random.Random(seed); t0 = time.time(); it = 0
    while time.time() - t0 < T:
        it += 1
        if rnd.random() < 0.5:   # 중심 = 임의 고객
            c = rnd.randrange(nI); cx, cy = cust["x"][c], cust["y"][c]
        else:                    # 중심 = 임의 후보(열림/닫힘 무관) → 개설 결정 바꾸기
            c = rnd.randrange(nJ); cx, cy = cand["x"][c], cand["y"][c]
        I_s = np.argsort(np.hypot(cust["x"] - cx, cust["y"] - cy))[:nc].tolist()
        J_s = set(np.argsort(np.hypot(cand["x"] - cx, cand["y"] - cy))[:ns].tolist())
        J_s |= {a[i, k] for i in I_s for k in (1, 2)}
        r = sub(X, a, I_s, sorted(J_s), tl=30, wk=4)
        if r is None: continue
        X2, a2, stn = r; c2 = total(X2, a2)
        if c2 < best - 1e-6:
            X, a, best = X2, a2, c2
            fx, tr, _ = evaluate(cand, cust, sorted(X), a, d)   # 실행가능성 검증
            json.dump(dict(obj=best, X=sorted(int(j) for j in X), assign=[[int(i), int(k), int(j)] for (i, k), j in a.items()]), open(out, "w"))
            print(f"{time.time()-t0:6.0f}s it{it:4d} {stn[:3]} -> {best:,.2f} (개설 {len(X)})", flush=True)
    print(f"final {best:,.2f} iters {it}")

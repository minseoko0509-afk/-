"""MIP 기반 LNS: 임의의 중심점 주변 고객 nc명·후보 ns곳만 자유 변수로 두고 나머지는 고정한 부분 MIP를 정확히 풂."""
import sys, json, time, random
import numpy as np, highspy
from solve_cfl import load, distances, TAU, V, evaluate

cand, cust = load("data/sites.csv", "data/customers.csv")
d = distances(cand, cust); h = cust["h"]; nI, nJ = d.shape
f, K, cold = cand["f"], cand["K"], cand["cold"]
inf = highspy.kHighsInf
W = {k: V[k] * h[k] for k in (1, 2)}


def total(X, a):
    return f[list(X)].sum() + sum(TAU[k] * h[k][i] * d[i, j] for (i, k), j in a.items())


def sub_mip(X, a, I_s, J_s, tl=60):
    X = set(X); I_s = list(I_s); J_s = list(J_s); Iset = set(I_s)
    fixed_load = np.zeros(nJ)
    for (i, k), j in a.items():
        if i not in Iset: fixed_load[j] += W[k][i]
    pos = {j: t for t, j in enumerate(J_s)}; nX = len(J_s)
    keys = [(i, j, k) for k in (1, 2) for i in I_s for j in J_s if k == 1 or cold[j]]
    # 고객의 품목이 J_s 밖에 붙어 있으면 J_s 안에 대안이 있어야 함: 없으면 그 품목은 고정
    has = {(i, k) for i, j, k in keys}
    ncol = nX + len(keys)
    cost = np.concatenate([f[J_s], [TAU[k] * h[k][i] * d[i, j] for i, j, k in keys]])
    lb = np.zeros(ncol); ub = np.ones(ncol)
    for j in J_s:
        if fixed_load[j] > 0: lb[pos[j]] = 1
    hs = highspy.Highs(); hs.setOptionValue("output_flag", False)
    hs.setOptionValue("time_limit", float(tl)); hs.setOptionValue("mip_rel_gap", 1e-6)
    hs.addVars(ncol, lb, ub); ix = np.arange(ncol, dtype=np.int32)
    hs.changeColsCost(ncol, ix, cost)
    hs.changeColsIntegrality(ncol, ix, np.array([highspy.HighsVarType.kInteger] * ncol))
    starts, ci, cv, lo, up = [], [], [], [], []
    by_ik, by_j = {}, {j: ([pos[j]], [-(K[j] - fixed_load[j])]) for j in J_s}
    for t, (i, j, k) in enumerate(keys):
        c = nX + t; by_ik.setdefault((i, k), []).append(c)
        by_j[j][0].append(c); by_j[j][1].append(W[k][i])
        starts.append(len(ci)); ci += [c, pos[j]]; cv += [1, -1]; lo.append(-inf); up.append(0)
    for cols in by_ik.values():
        starts.append(len(ci)); ci += cols; cv += [1] * len(cols); lo.append(1); up.append(1)
    for j in J_s:
        starts.append(len(ci)); ci += by_j[j][0]; cv += by_j[j][1]; lo.append(-inf); up.append(0)
    hs.addRows(len(lo), np.array(lo, float), np.array(up, float), len(ci),
               np.array(starts, dtype=np.int32), np.array(ci, dtype=np.int32), np.array(cv, float))
    # 부분문제에 포함되지 않는 (i,k) (냉장 후보가 J_s 에 없음) 는 원래 배정 유지 → 그 센터는 J_s 밖이어야 함
    # 초기해(현재 해) 주입
    v = np.zeros(ncol); kpos = {key: nX + t for t, key in enumerate(keys)}
    ok = True
    for j in J_s:
        if j in X: v[pos[j]] = 1
    for i in I_s:
        for k in (1, 2):
            j = a[i, k]
            if (i, k) in has:
                if (i, j, k) in kpos: v[kpos[i, j, k]] = 1
                else: ok = False
    if ok:
        s = highspy.HighsSolution(); s.col_value = list(v); s.value_valid = True; hs.setSolution(s)
    hs.run()
    if hs.getInfo().primal_solution_status != 2: return None
    sol = np.array(hs.getSolution().col_value)
    X2 = (X - set(J_s)) | {j for j in J_s if sol[pos[j]] > 0.5}
    a2 = dict(a)
    for t, (i, j, k) in enumerate(keys):
        if sol[nX + t] > 0.5: a2[i, k] = j
    # 배정이 없어진 센터는 닫음
    used = set(a2.values()); X2 = {j for j in X2 if j in used}
    return X2, a2


def check(X, a):
    ld = np.zeros(nJ)
    for (i, k), j in a.items():
        assert j in X and (k == 1 or cold[j]); ld[j] += W[k][i]
    assert (ld <= K + 1e-6).all()


def save(X, a, fn="best_lns.json"):
    json.dump(dict(obj=total(X, a), X=sorted(int(j) for j in X),
                   assign=[[int(i), int(k), int(j)] for (i, k), j in a.items()]), open(fn, "w"))


if __name__ == "__main__":
    src, T = sys.argv[1], float(sys.argv[2]); nc = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    ns = int(sys.argv[4]) if len(sys.argv) > 4 else 30; seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    B = json.load(open(src)); X = set(B["X"]); a = {(i, k): j for i, k, j in B["assign"]}
    check(X, a); best = total(X, a); print(f"start {best:,.2f}", flush=True)
    rnd = random.Random(seed); t0 = time.time(); it = 0
    while time.time() - t0 < T:
        it += 1
        c = rnd.randrange(nI); cx, cy = cust["x"][c], cust["y"][c]
        I_s = np.argsort(np.hypot(cust["x"] - cx, cust["y"] - cy))[:nc]
        # 부분 고객의 현재 센터 + 중심 근처 후보
        J_s = set(np.argsort(np.hypot(cand["x"] - cx, cand["y"] - cy))[:ns].tolist())
        J_s |= {a[i, k] for i in I_s for k in (1, 2)}
        r = sub_mip(X, a, I_s, sorted(J_s), tl=90)
        if r is None: continue
        X2, a2 = r; c2 = total(X2, a2)
        if c2 < best - 1e-6:
            check(X2, a2); X, a, best = X2, a2, c2; save(X, a)
            print(f"{time.time()-t0:6.0f}s it{it:4d} -> {best:,.2f} (개설 {len(X)})", flush=True)
    print(f"final {best:,.2f} (개설 {len(X)}) iters {it}")

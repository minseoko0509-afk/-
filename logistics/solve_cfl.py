"""IE325 Case: 상온·냉장 통합 물류센터 재설계
Multi-commodity Capacitated Fixed-Charge Location Problem (품목별 single sourcing).

min  Σj fj Xj + Σk τk Σi Σj hik dij Yijk
s.t. Σj Yijk = 1                      ∀i,k      (품목마다 한 센터)
     Σi Σk vk hik Yijk <= Kj Xj        ∀j        (두 품목이 용량을 함께 씀, 부피 기준)
     Yijk <= Xj                        ∀i,j,k    (strong linking — LP 완화를 강하게)
     Yij2 = 0  if aj2 = 0                        (냉장은 설비 있는 곳만: 변수를 아예 만들지 않음)

사용법: python solve_cfl.py [data/sites.csv] [data/customers.csv] [time_limit_sec]
"""
import sys
import numpy as np
import pandas as pd

TAU = {1: 0.14, 2: 0.35}   # 백만원/(천박스·격자)
V = {1: 1.0, 2: 2.5}       # 부피 환산


def load(sites_csv, cust_csv):
    S = pd.read_csv(sites_csv)
    C = pd.read_csv(cust_csv)
    cand = dict(name=S["name"].values, x=S["x"].values, y=S["y"].values,
                f=S["f"].values.astype(float), K=S["K"].values.astype(float),
                cold=S["a1"].values.astype(bool))
    # 고객 CSV: h0 = 상온(k=1), h1 = 냉장(k=2)
    cust = dict(name=C["name"].values, x=C["x"].values, y=C["y"].values,
                h={1: C["h0"].values.astype(float), 2: C["h1"].values.astype(float)})
    return cand, cust


def distances(cand, cust):
    return np.hypot(cust["x"][:, None] - cand["x"][None, :],
                    cust["y"][:, None] - cand["y"][None, :])


def solve(cand, cust, time_limit=900, gap=1e-5, nearest=None, start=None, threads=None):
    """nearest=n 이면 각 권역·품목을 가까운 후보 n곳에만 배정 가능하게 줄인 모델(휴리스틱)."""
    import highspy  # ortools 와 동시 로드 시 심볼 충돌 → 필요할 때만 로드
    nI, nJ = len(cust["x"]), len(cand["x"])
    d = distances(cand, cust)
    h = cust["h"]
    J = {1: np.arange(nJ), 2: np.flatnonzero(cand["cold"])}

    # ---- 변수: X(nJ) 다음에 Y(i,j,k) ----
    keys, cost = [], [*cand["f"]]
    for k in (1, 2):
        for i in range(nI):
            Jik = J[k] if nearest is None else J[k][np.argsort(d[i, J[k]])[:nearest]]
            for j in Jik:
                keys.append((i, j, k))
                cost.append(TAU[k] * h[k][i] * d[i, j])
    nY = len(keys)
    ncol = nJ + nY
    ycol = {key: nJ + t for t, key in enumerate(keys)}

    hs = highspy.Highs()
    hs.setOptionValue("time_limit", float(time_limit))
    hs.setOptionValue("mip_rel_gap", gap)
    hs.setOptionValue("output_flag", True)
    inf = highspy.kHighsInf
    hs.addVars(ncol, np.zeros(ncol), np.ones(ncol))
    hs.changeColsCost(ncol, np.arange(ncol, dtype=np.int32), np.array(cost))
    hs.changeColsIntegrality(ncol, np.arange(ncol, dtype=np.int32),
                             np.array([highspy.HighsVarType.kInteger] * ncol))

    lo, up, starts, idx, val = [], [], [], [], []

    def row(l, u, cols, vals):
        starts.append(len(idx)); lo.append(l); up.append(u)
        idx.extend(cols); val.extend(vals)

    by_ik, by_j = {}, {j: ([j], [-cand["K"][j]]) for j in range(nJ)}
    for (i, j, k), c in ycol.items():
        by_ik.setdefault((i, k), []).append(c)
        by_j[j][0].append(c); by_j[j][1].append(V[k] * h[k][i])
    for cols in by_ik.values():                        # 배정
        row(1, 1, cols, [1.0] * len(cols))
    for j in range(nJ):                                # 용량
        row(-inf, 0, *by_j[j])
    for (i, j, k), c in ycol.items():                  # Y <= X
        row(-inf, 0, [c, j], [1.0, -1.0])

    hs.addRows(len(lo), np.array(lo), np.array(up), len(idx),
               np.array(starts, dtype=np.int32), np.array(idx, dtype=np.int32), np.array(val))
    if threads:
        hs.setOptionValue("threads", int(threads))
    if start is not None:                              # 초기해 (X, assign)
        X0, a0 = start
        v = np.zeros(ncol); v[list(X0)] = 1
        for (i, k), j in a0.items():
            if (i, j, k) in ycol: v[ycol[i, j, k]] = 1
        sol0 = highspy.HighsSolution(); sol0.col_value = list(v); sol0.value_valid = True
        hs.setSolution(sol0)
    hs.run()
    sol = np.array(hs.getSolution().col_value)
    info = hs.getInfo()
    X = [j for j in range(nJ) if sol[j] > 0.5]
    assign = {(i, k): j for (i, j, k), c in ycol.items() if sol[c] > 0.5}
    return X, assign, d, info


def evaluate(cand, cust, X, assign, d):
    """해를 독립적으로 다시 검증하고 비용을 계산."""
    nI = len(cust["x"]); h = cust["h"]
    load = {j: 0.0 for j in X}
    for i in range(nI):
        for k in (1, 2):
            j = assign[i, k]
            assert j in load, "미개설 센터에 배정"
            if k == 2:
                assert cand["cold"][j], "냉장 설비 없는 센터에 냉장 배정"
            load[j] += V[k] * h[k][i]
    for j in X:
        assert load[j] <= cand["K"][j] + 1e-6, f"용량 초과 {cand['name'][j]}"
    fixed = sum(cand["f"][j] for j in X)
    tr = {k: sum(TAU[k] * h[k][i] * d[i, assign[i, k]] for i in range(nI)) for k in (1, 2)}
    return fixed, tr, load


def report(cand, cust, X, assign, d, out_prefix="solution"):
    nI = len(cust["x"]); h = cust["h"]
    fixed, tr, load = evaluate(cand, cust, X, assign, d)
    total = fixed + tr[1] + tr[2]
    print(f"\n개설 {len(X)}곳 (냉장 {sum(cand['cold'][j] for j in X)}곳)")
    print(f"고정비 {fixed:,.1f} | 상온 운송비 {tr[1]:,.1f} | 냉장 운송비 {tr[2]:,.1f} | 총비용 {total:,.1f} 백만원/년")
    print(f"{'센터':>5} {'냉장':>3} {'f':>6} {'K':>6} {'상온h':>6} {'냉장h':>6} {'부피':>7} {'가동률':>6}")
    rows = []
    for j in X:
        h1 = sum(h[1][i] for i in range(nI) if assign[i, 1] == j)
        h2 = sum(h[2][i] for i in range(nI) if assign[i, 2] == j)
        print(f"{cand['name'][j]:>5} {'○' if cand['cold'][j] else '—':>3} {cand['f'][j]:6.0f} {cand['K'][j]:6.0f} "
              f"{h1:6.0f} {h2:6.0f} {load[j]:7.1f} {load[j]/cand['K'][j]:6.1%}")
        rows.append(dict(j=j, name=cand["name"][j], cold=int(cand["cold"][j]), f=cand["f"][j], K=cand["K"][j],
                         h_amb=h1, h_cold=h2, volume=load[j], util=load[j] / cand["K"][j]))
    pd.DataFrame(rows).to_csv(f"{out_prefix}_open_sites.csv", index=False)
    split = [i for i in range(nI) if assign[i, 1] != assign[i, 2]]
    print(f"상온·냉장을 서로 다른 센터로 보낸 권역: {len(split)}곳")
    grid = 5.4
    pd.DataFrame([dict(i=i, name=cust["name"][i], h_amb=h[1][i], h_cold=h[2][i],
                       amb_center=cand["name"][assign[i, 1]], cold_center=cand["name"][assign[i, 2]],
                       amb_km=round(d[i, assign[i, 1]] * grid, 1), cold_km=round(d[i, assign[i, 2]] * grid, 1))
                  for i in range(nI)]).to_csv(f"{out_prefix}_assignment.csv", index=False)
    return total


if __name__ == "__main__":
    a = sys.argv[1:]
    sites = a[0] if len(a) > 0 else "data/sites.csv"
    custs = a[1] if len(a) > 1 else "data/customers.csv"
    tl = float(a[2]) if len(a) > 2 else 900
    nn = int(a[3]) if len(a) > 3 else None
    cand, cust = load(sites, custs)
    X, assign, d, info = solve(cand, cust, tl, nearest=nn)
    print(f"\nMIP obj {info.objective_function_value:,.2f}  dual bound {info.mip_dual_bound:,.2f}  gap {info.mip_gap:.4%}")
    report(cand, cust, X, assign, d)

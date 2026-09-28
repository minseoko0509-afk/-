"""IE325 Case: 상온·냉장 통합 물류센터 재설계
Multi-commodity Capacitated Fixed-Charge Location Problem (single sourcing per commodity).

min  Σj fj Xj + Σk τk Σi Σj hik dij Yijk
s.t. Σj Yijk = 1                      ∀i,k
     Σi Σk vk hik Yijk <= Kj Xj        ∀j
     Yijk <= Xj                        ∀i,j,k   (strong linking)
     Yij2 = 0  if j has no cold facility
사용법: python solve_cfl.py candidates.csv customers.csv [time_limit_sec]
"""
import sys, math
import numpy as np, pandas as pd, highspy

TAU = {1: 0.14, 2: 0.35}
V = {1: 1.0, 2: 2.5}


def pick(df, *names):
    cols = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n.lower() in cols:
            return df[cols[n.lower()]]
    raise KeyError(f"none of {names} in {list(df.columns)}")


def load(cand_csv, cust_csv):
    C = pd.read_csv(cand_csv); D = pd.read_csv(cust_csv)
    cand = dict(
        name=pick(C, "name", "후보지", "id", "site").astype(str).values,
        x=pick(C, "x").astype(float).values, y=pick(C, "y").astype(float).values,
        f=pick(C, "f", "fixed", "fj", "fixed_cost").astype(float).values,
        K=pick(C, "K", "cap", "capacity", "kj").astype(float).values,
        cold=pick(C, "cold", "refrig", "냉장", "a2", "a", "reefer").astype(str).str.strip()
            .isin(["1", "1.0", "True", "true", "○", "Y", "y", "yes"]).values,
    )
    cust = dict(
        name=pick(D, "name", "권역", "id").astype(str).values,
        x=pick(D, "x").astype(float).values, y=pick(D, "y").astype(float).values,
        h1=pick(D, "h1", "h_1", "amb", "ambient", "상온").astype(float).values,
        h2=pick(D, "h2", "h_2", "cold", "냉장").astype(float).values,
    )
    return cand, cust


def solve(cand, cust, time_limit=900):
    nI, nJ = len(cust["x"]), len(cand["x"])
    d = np.hypot(cust["x"][:, None] - cand["x"][None, :], cust["y"][:, None] - cand["y"][None, :])
    h = {1: cust["h1"], 2: cust["h2"]}
    J = {1: list(range(nJ)), 2: [j for j in range(nJ) if cand["cold"][j]]}

    hs = highspy.Highs(); hs.setOptionValue("time_limit", float(time_limit))
    hs.setOptionValue("mip_rel_gap", 1e-4)
    inf = highspy.kHighsInf
    # X vars
    for j in range(nJ):
        hs.addVar(0, 1); hs.changeColCost(j, cand["f"][j]); hs.changeColIntegrality(j, highspy.HighsVarType.kInteger)
    idx = {}
    col = nJ
    for k in (1, 2):
        for i in range(nI):
            for j in J[k]:
                hs.addVar(0, 1); hs.changeColCost(col, TAU[k] * h[k][i] * d[i, j])
                hs.changeColIntegrality(col, highspy.HighsVarType.kInteger)
                idx[i, j, k] = col; col += 1
    # assignment
    for k in (1, 2):
        for i in range(nI):
            cs = [idx[i, j, k] for j in J[k]]
            hs.addRow(1, 1, len(cs), cs, [1.0] * len(cs))
    # capacity
    for j in range(nJ):
        cs, vs = [j], [-cand["K"][j]]
        for k in (1, 2):
            if j in J[k] or k == 1:
                for i in range(nI):
                    if (i, j, k) in idx:
                        cs.append(idx[i, j, k]); vs.append(V[k] * h[k][i])
        hs.addRow(-inf, 0, len(cs), cs, vs)
    # strong linking Y <= X
    for (i, j, k), c in idx.items():
        hs.addRow(-inf, 0, 2, [c, j], [1.0, -1.0])
    hs.run()
    sol = hs.getSolution().col_value
    info = hs.getInfo()
    X = [j for j in range(nJ) if sol[j] > 0.5]
    assign = {}
    for (i, j, k), c in idx.items():
        if sol[c] > 0.5:
            assign[i, k] = j
    return X, assign, d, info


def report(cand, cust, X, assign, d):
    nI = len(cust["x"]); h = {1: cust["h1"], 2: cust["h2"]}
    fixed = sum(cand["f"][j] for j in X)
    tr = {k: sum(TAU[k] * h[k][i] * d[i, assign[i, k]] for i in range(nI)) for k in (1, 2)}
    print(f"개설 {len(X)}곳 : {[cand['name'][j] for j in X]}")
    print(f"고정비 {fixed:,.1f} | 상온 운송비 {tr[1]:,.1f} | 냉장 운송비 {tr[2]:,.1f} | 총비용 {fixed+tr[1]+tr[2]:,.1f}")
    print("센터별 적재(부피) / 용량")
    for j in X:
        load = sum(V[k] * h[k][i] for (i, k), jj in assign.items() if jj == j)
        print(f"  {cand['name'][j]:>5} {'냉장' if cand['cold'][j] else '상온'} f={cand['f'][j]:,.0f} {load:,.0f}/{cand['K'][j]:,.0f} ({load/cand['K'][j]:.0%})")
    split = sum(1 for i in range(nI) if assign[i, 1] != assign[i, 2])
    print(f"상온·냉장을 서로 다른 센터로 보낸 권역: {split}곳")
    rows = [dict(i=i, name=cust["name"][i], amb=cand["name"][assign[i, 1]], cold=cand["name"][assign[i, 2]]) for i in range(nI)]
    pd.DataFrame(rows).to_csv("assignment.csv", index=False)
    pd.DataFrame(dict(j=X, name=[cand["name"][j] for j in X])).to_csv("open_sites.csv", index=False)


if __name__ == "__main__":
    cand, cust = load(sys.argv[1], sys.argv[2])
    tl = float(sys.argv[3]) if len(sys.argv) > 3 else 900
    X, assign, d, info = solve(cand, cust, tl)
    print(f"MIP obj {info.objective_function_value:,.1f}  bound {info.mip_dual_bound:,.1f}  gap {info.mip_gap:.4%}")
    report(cand, cust, X, assign, d)

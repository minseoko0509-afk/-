"""손풀이 스타일 비교 기준.
규칙: (1) 권역(R1~R5)마다 f/K가 가장 싼 냉장 후보부터 열어 그 권역 냉장 부피를 덮고,
      (2) f/K가 싼 후보를 추가로 열어 권역 전체 부피의 110%를 덮는다.
      (3) 배정은 '가장 가까운 개설 센터'(냉장은 가까운 냉장 센터) — 용량은 무시 → 이후 검사.
      (3') 용량을 지키려면 물동량 큰 순서로 '남은 용량이 있는 가장 가까운 센터'에 배정.
"""
import numpy as np, pandas as pd
from solve_cfl import load, distances, TAU, V

cand, cust = load("data/sites.csv", "data/customers.csv")
d = distances(cand, cust); h = cust["h"]; nI, nJ = d.shape
S = pd.read_csv("data/sites.csv"); C = pd.read_csv("data/customers.csv")
f, K, cold = cand["f"], cand["K"], cand["cold"]

opened = []
for r in sorted(S.region.unique()):
    cs = C[C.region == r]; need_c = 2.5 * cs.h1.sum(); need = cs.h0.sum() + need_c
    g = S[S.region == r].assign(r=lambda t: t.f / t.K)
    got = 0
    for j in g[g.a1 == 1].sort_values("r").j:
        if got >= need_c: break
        opened.append(j); got += K[j]
    for j in g[~g.j.isin(opened)].sort_values("r").j:
        if got >= 1.1 * need: break
        opened.append(j); got += K[j]

def nearest(opened):
    op = np.array(opened); opc = op[cold[op]]
    a1 = op[np.argmin(d[:, op], 1)]; a2 = opc[np.argmin(d[:, opc], 1)]
    load = {j: 0.0 for j in opened}
    for i in range(nI): load[a1[i]] += h[1][i]; load[a2[i]] += 2.5 * h[2][i]
    over = {cand["name"][j]: round(load[j] - K[j]) for j in opened if load[j] > K[j]}
    cost = f[op].sum() + sum(TAU[1]*h[1][i]*d[i,a1[i]] + TAU[2]*h[2][i]*d[i,a2[i]] for i in range(nI))
    return cost, over

def greedy_cap(opened):
    rem = {j: K[j] for j in opened}; cost = f[list(opened)].sum()
    items = sorted([(i, k) for i in range(nI) for k in (1, 2)], key=lambda t: -V[t[1]] * h[t[1]][t[0]])
    for i, k in items:
        js = sorted([j for j in opened if (k == 1 or cold[j])], key=lambda j: d[i, j])
        for j in js:
            if rem[j] >= V[k] * h[k][i]:
                rem[j] -= V[k] * h[k][i]; cost += TAU[k] * h[k][i] * d[i, j]; break
        else:
            return None
    return cost

print(f"손풀이 개설 {len(opened)}곳: {[cand['name'][j] for j in opened]}")
c, over = nearest(opened)
print(f"가장 가까운 센터 배정: 비용 {c:,.0f}, 용량 초과 센터 {len(over)}곳 {over}")
print(f"용량 지키는 탐욕 배정: 비용 {greedy_cap(opened):,.0f}")

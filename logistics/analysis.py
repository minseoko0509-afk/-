"""「생각해 볼 것」 질문용 수치 계산 + 손풀이 스타일 휴리스틱(비교 기준)."""
import numpy as np, pandas as pd
from solve_cfl import load, distances, TAU, V

cand, cust = load("data/sites.csv", "data/customers.csv")
d = distances(cand, cust); h = cust["h"]; nI, nJ = d.shape
f, K, cold = cand["f"], cand["K"], cand["cold"]
vol = h[1] + 2.5 * h[2]; D = vol.sum()
print(f"부피수요 {D:,.0f}, 냉장부피 {2.5*h[2].sum():,.0f}, 냉장설비용량 {K[cold].sum():,.0f}")

# Q1: 고정비 싼 순서로 17곳
o = np.argsort(f)
print(f"f 싼 17곳 용량합 {K[o[:17]].sum():,.0f}  (필요 {D:,.0f}),  그중 냉장 {cold[o[:17]].sum()}곳")
cum = np.cumsum(K[o]); n = np.searchsorted(cum, D) + 1
print(f"f 싼 순서로 열면 용량이 수요를 넘는 데 {n}곳 필요, 고정비 합 {f[o[:n]].sum():,.0f}")
# 용량 단위당 고정비
r = f / K; o2 = np.argsort(r)
cum2 = np.cumsum(K[o2]); n2 = np.searchsorted(cum2, D) + 1
print(f"f/K 싼 순서로 열면 {n2}곳, 고정비 {f[o2[:n2]].sum():,.0f}  (고정비 하한 근사)")
# 순수 고정비 하한: 연속 knapsack  min Σ f x  s.t. Σ K x >= D, 냉장 Σ K x >= 냉장부피
# Q3: 냉장만
print(f"냉장 31곳 전부 열 때 용량 {K[cold].sum():,.0f} < {D:,.0f} → 부족 {D-K[cold].sum():,.0f}; 고정비 {f[cold].sum():,.0f}")
# Q5: 가장 큰 권역
i = np.argmax(vol); print(f"최대 권역 {cust['name'][i]} 부피 {vol[i]:.0f} = {vol[i]/D:.1%}")
reg = pd.read_csv("data/customers.csv").assign(vol=vol).groupby("region").vol.sum()
print("권역(Region)별 부피 비중:\n", (reg / D).round(3).to_string())
sreg = pd.read_csv("data/sites.csv").groupby("region").agg(K=("K","sum"), coldK=("K", lambda s: 0))
S = pd.read_csv("data/sites.csv")
print(S.groupby("region").apply(lambda g: pd.Series(dict(K=g.K.sum(), coldK=g.K[g.a1==1].sum(), maxK=g.K.max()))).to_string())

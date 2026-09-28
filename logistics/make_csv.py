"""best_*.json 중 실행가능한 최저 비용 해를 골라 제출 양식 CSV 로 쓰고, CSV 만으로 재검증."""
import json, csv, glob
import numpy as np, pandas as pd
S = pd.read_csv("data/sites.csv"); C = pd.read_csv("data/customers.csv")
d = np.hypot(C.x.values[:, None] - S.x.values[None, :], C.y.values[:, None] - S.y.values[None, :])


def check(x, y0, y1):
    ok = ((y0.sum(1) == 1).all() and (y1.sum(1) == 1).all() and (y0 <= x).all() and (y1 <= x).all()
          and (y1[:, S.a1.values == 0] == 0).all()
          and (C.h0.values @ y0 + 2.5 * C.h1.values @ y1 <= S.K.values * x + 1e-9).all())
    cost = S.f.values @ x + 0.14 * (C.h0.values[:, None] * d * y0).sum() + 0.35 * (C.h1.values[:, None] * d * y1).sum()
    return ok, cost


def arrays(B):
    x = np.zeros(160, int); x[B["X"]] = 1
    y = {1: np.zeros((600, 160), int), 2: np.zeros((600, 160), int)}
    for i, k, j in B["assign"]: y[k][i, j] = 1
    return x, y[1], y[2]


best = None
for fn in glob.glob("best_*.json") + glob.glob("start.json"):
    x, y0, y1 = arrays(json.load(open(fn)))
    ok, c = check(x, y0, y1)
    print(f"{fn:22s} {'OK ' if ok else 'BAD'} {c:,.2f}")
    if ok and (best is None or c < best[0]): best = (c, fn, x, y0, y1)
c, fn, x, y0, y1 = best
rows = [[""] + [str(j) for j in range(160)], ["x_j"] + list(map(str, x)), ["y_ij_k0"] + [str(j) for j in range(160)]]
rows += [[str(i)] + list(map(str, y0[i])) for i in range(600)]
rows += [["y_ij_k1"] + [str(j) for j in range(160)]] + [[str(i)] + list(map(str, y1[i])) for i in range(600)]
csv.writer(open("IE325_case_solution.csv", "w", newline="", encoding="utf-8-sig")).writerows(rows)
R = list(csv.reader(open("IE325_case_solution.csv", encoding="utf-8-sig")))
x2 = np.array(R[1][1:], int); a = np.array([r[1:] for r in R[3:603]], int); b = np.array([r[1:] for r in R[604:1204]], int)
ok, c2 = check(x2, a, b)
tmpl = list(csv.reader(open(encoding="utf-8-sig", file="/root/.claude/uploads/05174d7e-ea65-54b3-a0f3-ed71f78cc8d5/12cf44c8-IE325_case_solution.csv")))
assert len(tmpl) == len(R) and all(t[0] == r[0] for t, r in zip(tmpl, R)) and tmpl[0] == R[0]
print(f"\n선택: {fn}  → CSV 재검증 {'OK' if ok else 'BAD'}  총비용 {c2:,.2f}  개설 {x2.sum()}곳")
json.dump(json.load(open(fn)), open("final_solution.json", "w"))

"""국소탐색 (시작해: best.json).
 - shift : 항목(권역 i, 품목 k)을 다른 개설 센터로 이동
 - swap  : 서로 다른 센터의 두 항목을 교환 (용량 때문에 shift가 막힐 때)
 - close : 센터 하나를 닫고 그 항목들을 재배치
 - open  : 닫힌 후보를 열고 이득이 되는 항목을 끌어옴
 - (close+open 조합은 close 후 open 으로 자연히 탐색)
"""
import sys, json, time, random
import numpy as np
from solve_cfl import load, distances, TAU, V, evaluate

cand, cust = load("data/sites.csv", "data/customers.csv")
d = distances(cand, cust); nI, nJ = d.shape
f, K, cold = cand["f"], cand["K"], cand["cold"]
items = [(i, k) for i in range(nI) for k in (1, 2)]
n = len(items)
it_i = np.array([i for i, k in items]); it_k = np.array([k for i, k in items])
w = np.array([V[k] * cust["h"][k][i] for i, k in items])                  # 부피
C = np.array([TAU[k] * cust["h"][k][i] * d[i] for i, k in items])          # 항목별 센터별 운송비 (n x nJ)
elig = np.array([[(k == 1) or cold[j] for j in range(nJ)] for i, k in items])
C[~elig] = np.inf
NEAR = np.argsort(C, 1)[:, :25]                                             # 항목별 가까운 후보


class State:
    def __init__(s, openset, a):
        s.open = np.zeros(nJ, bool); s.open[list(openset)] = True
        s.a = np.array(a); s.load = np.bincount(s.a, weights=w, minlength=nJ)

    def cost(s):
        return f[s.open].sum() + C[np.arange(n), s.a].sum()

    def feasible(s):
        return (s.load <= K + 1e-9).all() and s.open[s.a].all() and np.isfinite(C[np.arange(n), s.a]).all()


def improve_assign(s, items_subset=None, max_pass=50):
    """shift + swap 를 더 이상 개선이 없을 때까지."""
    idx = range(n) if items_subset is None else items_subset
    for _ in range(max_pass):
        improved = False
        # shift
        for t in idx:
            j0 = s.a[t]; best, bj = 0, -1
            for j in NEAR[t]:
                if j != j0 and s.open[j] and s.load[j] + w[t] <= K[j] + 1e-9:
                    g = C[t, j0] - C[t, j]
                    if g > best + 1e-9: best, bj = g, j
            if bj >= 0:
                s.load[j0] -= w[t]; s.load[bj] += w[t]; s.a[t] = bj; improved = True
        # swap : t -> j (가까운 곳), 그 센터의 항목 u -> j0
        for t in idx:
            j0 = s.a[t]
            for j in NEAR[t][:10]:
                if j == j0 or not s.open[j] or C[t, j] >= C[t, j0]: continue
                gt = C[t, j0] - C[t, j]
                us = np.flatnonzero(s.a == j)
                for u in us:
                    if not np.isfinite(C[u, j0]): continue
                    g = gt + C[u, j] - C[u, j0]
                    if g > 1e-9 and s.load[j] - w[u] + w[t] <= K[j] + 1e-9 and s.load[j0] - w[t] + w[u] <= K[j0] + 1e-9:
                        s.a[t], s.a[u] = j, j0
                        s.load[j] += w[t] - w[u]; s.load[j0] += w[u] - w[t]
                        improved = True; break
                if s.a[t] != j0: break
        if not improved: break
    return s


def reinsert(s, ts):
    """항목들을 (부피 큰 순) 남은 용량 있는 가장 싼 센터에 넣음. 실패 시 False."""
    for t in sorted(ts, key=lambda t: -w[t]):
        order = np.argsort(C[t])
        for j in order:
            if not np.isfinite(C[t, j]): return False
            if s.open[j] and s.load[j] + w[t] <= K[j] + 1e-9:
                s.a[t] = j; s.load[j] += w[t]; break
        else:
            return False
    return True


def copy(s):
    c = State.__new__(State); c.open = s.open.copy(); c.a = s.a.copy(); c.load = s.load.copy(); return c


def try_close(s, j):
    c = copy(s); ts = np.flatnonzero(c.a == j)
    c.open[j] = False; c.load[j] = 0
    if not reinsert(c, ts): return None
    near = set(ts.tolist())
    for t in ts: near |= set(np.flatnonzero(np.isin(c.a, NEAR[t][:6])).tolist())
    improve_assign(c, sorted(near), max_pass=5)
    return c


def try_open(s, j):
    c = copy(s); c.open[j] = True
    gain = np.array([C[t, c.a[t]] - C[t, j] for t in range(n)])
    for t in np.argsort(-gain / w):
        if gain[t] <= 0: break
        if c.load[j] + w[t] <= K[j]:
            c.load[c.a[t]] -= w[t]; c.load[j] += w[t]; c.a[t] = j
    ts = sorted(set(np.flatnonzero(c.a == j).tolist()) | set(t for t in range(n) if j in NEAR[t][:6]))
    improve_assign(c, ts, max_pass=5)
    return c


def search(s, time_limit=900, seed=0):
    rnd = random.Random(seed); t0 = time.time()
    improve_assign(s); best = s.cost(); print(f"start LS {best:,.2f}", flush=True)
    while time.time() - t0 < time_limit:
        improved = False
        moves = [("close", j) for j in np.flatnonzero(s.open)] + [("open", j) for j in np.flatnonzero(~s.open)]
        rnd.shuffle(moves)
        for m, j in moves:
            c = try_close(s, j) if m == "close" else try_open(s, j)
            if c is None: continue
            if m == "open":   # 연 뒤 다른 센터를 닫아 보는 swap
                cc = c.cost()
                if cc >= best - 1e-6:
                    for j2 in sorted(np.flatnonzero(c.open), key=lambda q: np.hypot(cand['x'][q]-cand['x'][j], cand['y'][q]-cand['y'][j]))[1:6]:
                        c2 = try_close(c, j2)
                        if c2 is not None and c2.cost() < best - 1e-6:
                            c = c2; break
            if c.cost() < best - 1e-6:
                improve_assign(c)
                s, best = c, c.cost(); improved = True
                print(f"  {time.time()-t0:6.0f}s {m} {cand['name'][j]:>5} -> {best:,.2f}  (개설 {s.open.sum()})", flush=True)
                save(s)
            if time.time() - t0 > time_limit: break
        if not improved: break
    return s


def save(s, fn="best_ls.json"):
    assert s.feasible()
    json.dump(dict(obj=s.cost(), X=[int(j) for j in np.flatnonzero(s.open)],
                   assign=[[int(i), int(k), int(s.a[t])] for t, (i, k) in enumerate(items)]), open(fn, "w"))


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "best.json"
    B = json.load(open(src)); amap = {(i, k): j for i, k, j in B["assign"]}
    s = State(B["X"], [amap[i, k] for i, k in items])
    assert s.feasible(); print(f"시작해 {s.cost():,.2f}")
    s = search(s, float(sys.argv[2]) if len(sys.argv) > 2 else 900)
    improve_assign(s); save(s)
    X = list(np.flatnonzero(s.open)); assign = {(i, k): int(s.a[t]) for t, (i, k) in enumerate(items)}
    fx, tr, _ = evaluate(cand, cust, X, assign, d)
    print(f"최종 {fx + tr[1] + tr[2]:,.2f}  (고정비 {fx:,.0f}, 상온 {tr[1]:,.1f}, 냉장 {tr[2]:,.1f}, 개설 {len(X)})")

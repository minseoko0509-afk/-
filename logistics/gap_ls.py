"""GAP 벌점 국소탐색: 개설 집합 고정, 목적 = 운송비 + λ·Σ 용량초과. shift/swap, 막히면 λ 증가.
사용법: python gap_ls.py start.json out.json"""
import sys, json
import numpy as np
from solve_cfl import load, distances, TAU, V, evaluate

cand, cust = load("data/sites.csv", "data/customers.csv")
d = distances(cand, cust); h = cust["h"]; nI, nJ = d.shape
f, K, cold = cand["f"], cand["K"], cand["cold"]


def run(Xopen, a0, lam0=1.0, max_rounds=60, verbose=True):
    S = np.array(sorted(Xopen)); m = len(S); pos = {j: t for t, j in enumerate(S)}
    items = [(i, k) for i in range(nI) for k in (1, 2)]; n = len(items)
    w = np.array([V[k] * h[k][i] for i, k in items])
    C = np.array([[TAU[k] * h[k][i] * d[i, j] if (k == 1 or cold[j]) else np.inf for j in S] for i, k in items])
    Kc = K[S].astype(float)
    a = np.array([pos[a0[i, k]] for i, k in items])
    load = np.bincount(a, weights=w, minlength=m)
    lam = lam0; best = (np.inf, None)
    over = lambda L: np.maximum(0, L - Kc)
    for rnd in range(max_rounds):
        changed = True
        while changed:
            changed = False
            # shift
            for t in range(n):
                j0 = a[t]
                newload_j = load + w[t]
                dpen = lam * (np.maximum(0, newload_j - Kc) - over(load))
                dpen0 = lam * (max(0, load[j0] - w[t] - Kc[j0]) - max(0, load[j0] - Kc[j0]))
                delta = C[t] - C[t, j0] + dpen + dpen0
                delta[j0] = 0
                j = int(np.argmin(delta))
                if delta[j] < -1e-9:
                    load[j0] -= w[t]; load[j] += w[t]; a[t] = j; changed = True
            # swap (t in j0) <-> (u in j1)
            for t in range(n):
                j0 = a[t]
                u = np.arange(n); j1 = a
                mask = j1 != j0
                dw = w[t] - w  # j1 gains dw, j0 loses dw
                L0 = load[j0] - dw; L1 = load[j1] + dw
                dpen = lam * (np.maximum(0, L0 - Kc[j0]) - max(0, load[j0] - Kc[j0])
                              + np.maximum(0, L1 - Kc[j1]) - np.maximum(0, load[j1] - Kc[j1]))
                with np.errstate(invalid="ignore"):
                    delta = C[t, j1] - C[t, j0] + C[u, j0] - C[u, j1] + dpen
                delta[~mask | ~np.isfinite(delta)] = np.inf
                uu = int(np.argmin(delta))
                if delta[uu] < -1e-9:
                    j1u = a[uu]
                    load[j0] += w[uu] - w[t]; load[j1u] += w[t] - w[uu]
                    a[t], a[uu] = j1u, j0; changed = True
        ov = over(load).sum(); tc = C[np.arange(n), a].sum()
        if ov <= 1e-9:
            if tc < best[0]: best = (tc, a.copy())
            if verbose: print(f"  round {rnd} λ={lam:.3g} 실행가능 운송비 {tc:,.2f} → 총 {tc + f[S].sum():,.2f}", flush=True)
            lam *= 0.7   # 실행가능이면 벌점을 조금 낮춰 다른 영역 탐색
            if rnd > 10 and lam < 1e-3: break
        else:
            if verbose: print(f"  round {rnd} λ={lam:.3g} 초과 {ov:,.1f} 운송비 {tc:,.2f}", flush=True)
            lam *= 2
    if best[1] is None: return None
    ab = best[1]
    return best[0] + f[S].sum(), {items[t]: int(S[ab[t]]) for t in range(n)}


if __name__ == "__main__":
    B = json.load(open(sys.argv[1])); A = {(i, k): j for i, k, j in B["assign"]}
    r = run(B["X"], A)
    if r:
        tot, a = r; Xo = sorted({j for j in a.values()})
        fx, tr, _ = evaluate(cand, cust, Xo, a, d)
        print(f"최종 {fx + tr[1] + tr[2]:,.2f} (개설 {len(Xo)})")
        json.dump(dict(obj=fx + tr[1] + tr[2], X=Xo, assign=[[i, k, j] for (i, k), j in a.items()]), open(sys.argv[2], "w"))

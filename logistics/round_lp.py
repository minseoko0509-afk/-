import sys, json, time
from improve import *
lp = build(range(nJ), integer=False); Xlp = lp["X"]
for thr in [0.5, 0.3, 0.2]:
    S = [j for j in range(nJ) if Xlp[j] >= thr]
    t = time.time(); c, r = gap_cost(S, tl=150)
    print(f"thr={thr}: {len(S)}곳 개설, 고정비 {f[S].sum():,.0f}, GAP 비용 {c:,.2f}  ({time.time()-t:.0f}s)", flush=True)
    if r:
        json.dump(dict(X=[int(j) for j in r["X"]], assign=[[i,k,int(j)] for (i,k),j in r["assign"].items()], obj=c),
                  open(f"round_{thr}.json","w"))

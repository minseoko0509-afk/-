"""HiGHS 전체 MIP 에 현재 최선해를 초기해로 주입 (RINS 등 개선 휴리스틱이 작동하도록)."""
import sys, json
from solve_cfl import load, solve, report
cand, cust = load("data/sites.csv", "data/customers.csv")
B = json.load(open(sys.argv[1])); A = {(i, k): j for i, k, j in B["assign"]}
X, assign, d, info = solve(cand, cust, float(sys.argv[2]), start=(B["X"], A), threads=1)
print(f"MIP obj {info.objective_function_value:,.2f} bound {info.mip_dual_bound:,.2f}")
json.dump(dict(obj=info.objective_function_value, X=[int(j) for j in X], assign=[[i, k, int(j)] for (i, k), j in assign.items()]), open("best_warm.json", "w"))

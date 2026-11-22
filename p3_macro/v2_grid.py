"""손절(1차 매수가 -10%)을 고정하고, 매매를 늘리는 조건 조합 96개를 비교.

앞 기간(~2004)과 뒤 기간(2005~)의 누적 배수를 따로 봐서, 한쪽에서만 좋은 조합을 거름.
"""
import itertools
import os
import numpy as np
import pandas as pd
from backtest_v2 import setup, run, stats, trade_table, V2

HERE = os.path.dirname(os.path.abspath(__file__))
df, fed, reentry, rf = setup()
grid = dict(dd=[0.20, 0.15], band_hi=[0.03, 0.05], hy_drop1=[0.3, 0.0], rebound=[0.06, 0.04],
            trend_ma=[None, 150, 200], credit=[False, True])
rows = []
for vals in itertools.product(*grid.values()):
    o = dict(zip(grid, vals))
    P = {**V2, **o, "trend": o["trend_ma"] is not None, "stop": 0.10, "stop_ref": "entry"}
    t, e, eq, s = run(df, P, fed, reentry, rf)
    tt = trade_table(t)
    tt = tt[tt["종료"] != "보유 중"]
    yr = pd.to_datetime(tt["1차 매수"]).dt.year
    m = lambda x: round(float(np.prod(1 + x / 100)), 2)
    rows.append({**{k: ("-" if v is None else v) for k, v in o.items()}, **stats(t, eq),
                 "~2004 배수": m(tt.loc[yr < 2005, "P3 자금%"]), "2005~ 배수": m(tt.loc[yr >= 2005, "P3 자금%"])})
out = pd.DataFrame(rows).sort_values("누적 배수", ascending=False)
out.to_csv(os.path.join(HERE, "v2_grid.csv"), index=False)
pd.set_option("display.width", 250, "display.max_rows", 200)
print(out.head(30).to_string(index=False))
print("\n매매 수별 중앙값 누적 배수:\n", out.groupby(pd.cut(out["매매"], [0, 8, 12, 16, 40]), observed=True)["누적 배수"].median())
for k in grid:
    print(k, out.groupby(k)["누적 배수"].median().to_dict())

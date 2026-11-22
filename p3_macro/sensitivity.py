"""P3 기준값을 하나씩 바꿔 보는 민감도 표, HY 대체값 크기 ±15%.

열: 매수 에피소드 수, 각 에피소드의 P3 자금 수익률(1차·2차 반반, 안 쓴 반은 현금),
전체 누적 배수(에피소드 수익률을 곱한 것), 첫 반등에 샀을 때 누적 배수.
"""
import numpy as np
import pandas as pd
from backtest import load, run, summarize, BASE, fed_actions, p1_reentry

df, _ = load()
fed, reentry = fed_actions(df.index), p1_reentry(df)


def one(label, P=BASE, d=df):
    s = summarize(d, run(d, P, fed, reentry))
    s = s[s["종료"] != "진행 중"]
    bought = s[s["1차 매수"] != "-"]
    per = ", ".join(f"{str(r['1차 매수'])[:4]}:{r['P3 자금 수익%']:+.0f}" for _, r in bought.iterrows())
    mult = np.prod(1 + bought["P3 자금 수익%"] / 100)
    reb = s[s["첫 반등"] != "-"]
    return {"변형": label, "감시": len(s), "매수": len(bought), "에피소드별 P3 수익%": per or "-",
            "누적 배수": round(mult, 2),
            "첫 반등 매수 누적": round(np.prod(1 + reb["첫 반등 매수%"] / 100), 2)}


rows = [one("기준 규칙")]
for k, vals in {"dd": [0.15, 0.25], "rebound": [0.04, 0.08, 0.10], "band_hi": [0.02, 0.05],
                "band_lo": [0.05, 0.10], "hy_drop1": [0.0, 0.6, 1.0], "hy_drop2": [0.5, 2.0],
                "hy_alarm": [4.0, 6.0]}.items():
    for v in vals:
        rows.append(one(f"{k}={v}", {**BASE, k: v}))
for scale in (0.85, 1.15):
    d = df.copy()
    d.loc[d.hy_is_proxy, "hy"] *= scale
    rows.append(one(f"HY 대체값 x{scale}", BASE, d))

out = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_colwidth", 80)
print(out.to_string(index=False))
out.to_csv("sensitivity.csv", index=False)

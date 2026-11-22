"""역P3 규칙을 VIX 현물(1990~)에 적용한 매매 시점 검증.

VIX 현물은 살 수 없으므로 금액 수익이 아니라 '진입 VIX → 종료 VIX'와 보유 중 최고 VIX만 봅니다.
숏볼이므로 종료 VIX가 진입보다 낮으면 방향이 맞은 것입니다.
"""
import os
import numpy as np
import pandas as pd
from states import p3_states, HERE
from mirror import run, BASE

if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 30)
    s, _ = p3_states()
    v = pd.read_csv(os.path.join(HERE, "data", "vix.csv"), index_col=0, parse_dates=True)["^VIX"]
    j = s.join(v.rename("vix"), how="inner")["1990-01-02":]
    for name, opt in [("기본", {}), ("손절 없음", dict(stop=None, p3stop=False))]:
        w, trades = run(j, {**BASE, **opt})
        rows = []
        for t in trades:
            seg = j.vix.loc[t["entry"]:(t["exit"] or j.index[-1])]
            rows.append({"진입": t["entry"].date(), "진입 VIX": round(t["entry_vix"], 1),
                         "추가": t["add"].date() if t["add"] is not None else "-",
                         "종료": t["exit"].date() if t["exit"] is not None else "보유 중", "방식": t["how"],
                         "종료 VIX": round(t["exit_vix"], 1), "보유 중 최고": round(seg.max(), 1),
                         "VIX 변화%": round((t["exit_vix"] / t["entry_vix"] - 1) * 100, 1),
                         "최대 역행%": round((seg.max() / t["entry_vix"] - 1) * 100, 1)})
        t = pd.DataFrame(rows)
        print(f"\n== {name}: {len(t)}번, VIX 하락으로 끝난 매매 {(t['VIX 변화%'] < 0).sum()}번")
        print(t.to_string(index=False))
        t.to_csv(os.path.join(HERE, f"spot_trades_{'base' if name == '기본' else 'nostop'}.csv"), index=False)

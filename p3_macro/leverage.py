"""P3를 QQQ 대신 QLD(2배)·TQQQ(3배)로 매매하면? 신호(감시·L·이중바닥·50일선)는 QQQ로 판단.

- QLD: 실제(2006-06~), 그 전은 합성 2 x QQQ - 1 x T-bill - 0.95%/년.
- TQQQ: 실제(2010-02~), 그 전은 합성 3 x QQQ - 2 x T-bill - 0.95%/년 (backtest.py와 같음).
- 손절은 세 가지로 비교:
  QQQ 기준   : QQQ 종가 < 1차 매수 시점 QQQ x (1 - s)   -> 매매 시점이 QQQ판과 같음
  자산 기준   : QLD/TQQQ 종가 < 1차 매수가 x (1 - s)
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf
from backtest import ER, DATA
from backtest_v2 import setup, run, stats, trade_table, V2, FINAL

HERE = os.path.dirname(os.path.abspath(__file__))


def add_qld(df):
    path = os.path.join(DATA, "qld.csv")
    if not os.path.exists(path):
        yf.download("QLD", start="2006-01-01", auto_adjust=True, progress=False,
                    threads=False)["Close"].to_csv(path)
    real = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].dropna()
    y = pd.read_csv(os.path.join(DATA, "yahoo.csv"), index_col=0, parse_dates=True)
    rf = (y["^IRX"].ffill() / 100 / 252).reindex(df.index).ffill()
    r = df.qqq.pct_change().fillna(0.0)
    syn = (1 + 2 * r - rf - ER).cumprod()
    out = syn.copy()
    t0 = real.index[0]
    out.loc[t0:] = real.reindex(df.loc[t0:].index).ffill() / real.iloc[0] * syn.loc[t0]
    df["qld"] = out
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 30)
    df, fed, reentry, rf = setup()
    df = add_qld(df)
    rows, keep = [], {}
    cases = [("기준 규칙(v1)", {}, None)]
    for s in (0.05, 0.07, 0.10):
        cases.append((f"개선안, QQQ 기준 손절 {int(s*100)}%", {**FINAL, "stop": s}, s))
    cases += [("개선안, 자산 기준 손절 10%", {**FINAL, "stop_ref": "asset"}, None),
              ("개선안, 자산 기준 손절 20%", {**FINAL, "stop_ref": "asset", "stop": 0.20}, None),
              ("개선안, 자산 기준 손절 30%", {**FINAL, "stop_ref": "asset", "stop": 0.30}, None)]
    for asset in ("qqq", "qld", "tqqq"):
        for name, opt, _ in cases:
            t, e, eq, st = run(df, {**V2, **opt, "asset": asset}, fed, reentry, rf)
            rows.append({"자산": asset.upper(), "규칙": name, **stats(t, eq)})
            keep[(asset, name)] = t
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(os.path.join(HERE, "leverage_results.csv"), index=False)

    # 개선안(QQQ 기준 손절 10%) 매매별 비교: 시점은 같고 수익만 다름
    base = "개선안, QQQ 기준 손절 10%"
    cmp_ = trade_table(keep[("qqq", base)])[["1차 매수", "2차 매수", "종료", "방식", "P3 자금%"]]
    cmp_ = cmp_.rename(columns={"P3 자금%": "QQQ%"})
    cmp_["QLD%"] = trade_table(keep[("qld", base)])["P3 자금%"].values
    cmp_["TQQQ%"] = trade_table(keep[("tqqq", base)])["P3 자금%"].values
    print("\n" + base + " 매매별 P3 자금 수익률")
    print(cmp_.round(1).to_string(index=False))
    cmp_.round(2).to_csv(os.path.join(HERE, "leverage_trades.csv"), index=False)

"""역P3 VIX 규칙: P3 단계를 VIX에 거울처럼 대응시킨 숏볼 규칙.

P3 (QQQ)                         역P3 (VIX, 숏볼)
1 감시 시작 (−15% + 배관 경보)   →  같음. 감시 시작 이후 VIX 최고 종가 Vmax 갱신
2 연준 항복                      →  숏볼 1차(반) 진입 관문 ①
3 첫 반등 (L×1.06)               →  'VIX 정점 확인' VIX ≤ Vmax×(1−peak_off) : 숏볼 1차 관문 ②
4 1차 매수 (이중바닥)            →  (VIX 재상승 구간. 신규 진입 없음)
5 2차 매수 (신용 정상화)         →  숏볼 2차(나머지 반)
손절 (QQQ < 1차가×0.90)          →  VIX 종가 > 진입 VIX×(1+stop) 또는 P3 손절일 → 전량 환매, 재시도
매도 (P1 TQQQ 재진입)            →  전량 환매 (또는 VIX ≤ exit_vix)
"""
import os
import numpy as np
import pandas as pd
from vix_rules import load_all, stats, HERE

BASE = dict(gate="fed_or_peak", peak_off=0.20, add="buy2", stop=0.25, p3stop=True,
            exit="p3", exit_vix=None, size1=0.5)


def run(j, P):
    st, ev, vix = j.state.values, j.event.values, j.vix.values
    w = np.zeros(len(j))        # 숏볼 비중 (0~1)
    trades = []
    pos, vmax, fed_ok, entry, tr = 0.0, None, False, None, None

    def close(i, how):
        nonlocal pos, tr
        tr.update(exit=j.index[i], exit_vix=vix[i], how=how)
        trades.append(tr); tr = None; pos = 0.0

    for i in range(len(j)):
        x, e, v = st[i], ev[i], vix[i]
        if "watch" in e:
            vmax, fed_ok = v, False
        if x == "IDLE":
            if pos > 0:
                close(i, "종료(P1)")
            vmax = None
            w[i] = 0; continue
        if vmax is None:          # 2011 이전에 시작한 감시
            vmax = v
        if "fed" in e:
            fed_ok = True
        if pos > 0:
            hit = v > entry * (1 + P["stop"]) if P["stop"] else False
            if P["p3stop"] and e == "stop":
                hit = True
            if hit:
                close(i, "손절")
                vmax = max(vmax, v)
                w[i] = 0; continue
            if P["exit_vix"] and v <= P["exit_vix"]:
                close(i, f"VIX≤{P['exit_vix']}")
                w[i] = 0; continue
            if pos < 1 and ((P["add"] == "buy2" and x == "HOLD2") or
                            (P["add"] == "half_more" and v <= vmax * (1 - 2 * P["peak_off"]))):
                pos = 1.0; tr["add"] = j.index[i]
        else:
            vmax = max(vmax, v)
            g = P["gate"]
            peak = v <= vmax * (1 - P["peak_off"])
            ok = {"watch": True, "fed": fed_ok, "peak": peak,
                  "fed_or_peak": fed_ok or peak, "fed_and_peak": fed_ok and peak}[g]
            if ok and not (P["exit_vix"] and v <= P["exit_vix"]):
                pos = P["size1"] if P["add"] else 1.0
                if x == "HOLD2":
                    pos = 1.0
                entry = v
                tr = dict(entry=j.index[i], entry_vix=v, vmax=vmax, add=None)
        w[i] = pos
    if tr is not None:
        tr.update(exit=None, exit_vix=vix[-1], how="보유 중"); trades.append(tr)
    return pd.Series(w, index=j.index), trades


def equity(w, j):
    p = w.shift(1).fillna(0)
    daily = p * j.short + (1 - p) * j.rf
    return pd.Series(np.cumprod(1 + daily), index=j.index), -p


VARIANTS = {
    "기본: 연준 또는 VIX 정점 확인(−20%) → 반, 2차매수 → 전부, 진입VIX +25% 손절, P1 종료": {},
    "관문=감시 시작 즉시": dict(gate="watch"),
    "관문=연준만": dict(gate="fed"),
    "관문=VIX 정점 확인만": dict(gate="peak"),
    "정점 확인 −15%": dict(peak_off=0.15),
    "정점 확인 −30%": dict(peak_off=0.30),
    "처음부터 전부": dict(add=None),
    "추가=VIX가 정점 −40%": dict(add="half_more"),
    "손절 없음": dict(stop=None, p3stop=False),
    "손절 +15%": dict(stop=0.15),
    "손절 +40%": dict(stop=0.40),
    "P3 손절 연동 안 함": dict(p3stop=False),
    "종료 VIX≤20": dict(exit_vix=20),
    "종료 VIX≤18": dict(exit_vix=18),
}


def trade_table(trades, j):
    rows = []
    for t in trades:
        a, b = t["entry"], t["exit"] or j.index[-1]
        seg = j.loc[a:b]
        rows.append({"진입": a.date(), "진입 VIX": round(t["entry_vix"], 1), "Vmax": round(t["vmax"], 1),
                     "추가": t["add"].date() if t["add"] is not None else "-",
                     "종료": t["exit"].date() if t["exit"] is not None else "보유 중", "방식": t["how"],
                     "종료 VIX": round(t["exit_vix"], 1), "보유 중 VIX 최고": round(seg.vix.max(), 1)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 80)
    j, _ = load_all()
    real = j["2011-01-03":]
    rows = []
    for name, opt in VARIANTS.items():
        w, trades = run(real, {**BASE, **opt})
        eq, p = equity(w, real)
        tt = trade_table(trades, real)
        # 매매별 수익
        rets = [eq.loc[t["entry"]:(t["exit"] or eq.index[-1])].pipe(lambda s: s.iloc[-1] / s.iloc[0] - 1) for t in trades]
        rows.append({"변형": name, "매매": len(trades), "승": int(sum(r > 0 for r in rets)),
                     "최악 매매%": round(min(rets) * 100, 1) if rets else np.nan, **stats(eq, p)})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(os.path.join(HERE, "mirror_variants.csv"), index=False)
    w, trades = run(real, BASE)
    eq, p = equity(w, real)
    tt = trade_table(trades, real)
    tt["수익%"] = [round((eq.loc[t["entry"]:(t["exit"] or eq.index[-1])].iloc[-1] /
                       eq.loc[t["entry"]:(t["exit"] or eq.index[-1])].iloc[0] - 1) * 100, 1) for t in trades]
    print(tt.to_string(index=False))

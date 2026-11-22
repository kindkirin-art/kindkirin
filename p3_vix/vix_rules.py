"""P3 신호를 거꾸로 써서 VIX(변동성)를 매매하는 규칙 비교, 2006-07 ~ 현재.

매매 대상 (매일 리밸런싱, 쉬는 돈은 T-bill)
- 롱볼: VIXY (S&P 500 VIX 단기선물 지수 1배, 보수 0.85%)
- 숏볼: SVXY 방식 (같은 지수 −0.5배, 보수 0.95%)
VIXY는 2011-01부터 실제 값. 그 전(2006-07~2010-12)은 VIX·VIX3M으로 만든 대체값:
  지수 초과수익 = a + b1·ΔlnVIX + b2·ΔlnVIX3M + c·ln(VIX3M/VIX)[전날]
  계수는 2011~ 실제 값으로 추정 (2011~2018-06으로 추정해 2018-07~ 에 적용하면 연 −49% vs 실제 −56%).
P3 단계는 states.py (P3 v2 채택안 그대로).
"""
import os
import numpy as np
import pandas as pd
from states import p3_states, HERE

FEE_L, FEE_S, K_S = 0.0085, 0.0095, 0.5


def vol_returns():
    v = pd.read_csv(os.path.join(HERE, "data", "vix.csv"), index_col=0, parse_dates=True)
    y = pd.read_csv(os.path.join(HERE, "..", "p3_macro", "data", "yahoo.csv"), index_col=0, parse_dates=True)
    rf = (y["^IRX"].ffill() / 100 / 252).reindex(v.index).ffill()
    vix, v3 = v["^VIX"], v["^VIX3M"]
    er_real = v["VIXY"].pct_change() - rf + FEE_L / 252
    X = pd.DataFrame({"dl": np.log(vix).diff(), "dl3": np.log(v3).diff(),
                      "sl": np.log(v3 / vix).shift(1)})
    fit = pd.concat([er_real, X], axis=1, keys=["er", "x"]).dropna()["2011-01-05":]
    A = np.column_stack([np.ones(len(fit)), fit["x"].values])
    coef = np.linalg.lstsq(A, fit["er"].values.ravel(), rcond=None)[0]
    er_syn = coef[0] + X.values @ coef[1:]
    er = pd.Series(er_syn, index=v.index)
    er.loc["2011-01-05":] = er_real.loc["2011-01-05":]
    out = pd.DataFrame({"er": er, "rf": rf, "vix": vix, "vix3m": v3,
                        "svxy": v["SVXY"], "vixy": v["VIXY"]}).dropna(subset=["er", "rf"])
    out["long"] = out.er + out.rf - FEE_L / 252
    out["short"] = -K_S * out.er + out.rf - FEE_S / 252
    return out, coef


def positions(s, rule):
    """날짜별 포지션: +1 롱볼, −1 숏볼, 0 현금. 그날 종가에 정하고 다음 날 수익에 적용."""
    st, ev = s.state.values, s.event.values
    contango = (s.vix < s.vix3m).values
    alarm = s.alarm.values
    pos = np.zeros(len(s))
    for i in range(len(s)):
        x, e = st[i], ev[i]
        if rule == "숏볼 항상":
            p = -1
        elif rule == "롱볼 항상":
            p = 1
        elif rule == "감시~종료 숏":
            p = -1 if x != "IDLE" else 0
        elif rule == "연준~종료 숏":
            p = -1 if x in ("FED", "REB", "HOLD1", "HOLD2") else 0
        elif rule == "연준~종료 숏, 1차 후 현금":
            p = -1 if x in ("FED", "REB", "HOLD2") else 0
        elif rule == "연준~종료 숏, 1차 후 롱":
            p = -1 if x in ("FED", "REB", "HOLD2") else (1 if x == "HOLD1" else 0)
        elif rule == "1차 후 롱만":
            p = 1 if x == "HOLD1" else 0
        elif rule == "평시 캐리":           # P3 없이: 경보 없고 콘탱고면 숏
            p = -1 if (not alarm[i] and contango[i]) else 0
        elif rule == "평시 캐리 + P3":       # 평시 캐리, 감시~연준 전 현금, 연준 후 P3 숏, 1차 후 현금
            if x == "IDLE":
                p = -1 if (not alarm[i] and contango[i]) else 0
            elif x == "WATCH" or x == "HOLD1":
                p = 0
            else:
                p = -1
        elif rule == "평시 캐리 + P3 (1차 후 롱)":
            if x == "IDLE":
                p = -1 if (not alarm[i] and contango[i]) else 0
            elif x == "WATCH":
                p = 0
            elif x == "HOLD1":
                p = 1
            else:
                p = -1
        else:
            raise ValueError(rule)
        pos[i] = p
    return pd.Series(pos, index=s.index)


RULES = ["숏볼 항상", "롱볼 항상", "평시 캐리", "감시~종료 숏", "연준~종료 숏",
         "연준~종료 숏, 1차 후 현금", "연준~종료 숏, 1차 후 롱", "1차 후 롱만",
         "평시 캐리 + P3", "평시 캐리 + P3 (1차 후 롱)"]


def equity(pos, r):
    p = pos.shift(1).fillna(0)
    daily = np.where(p > 0, r.long, np.where(p < 0, r.short, r.rf))
    return pd.Series(np.cumprod(1 + daily), index=r.index), p


def stats(eq, p):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    d = eq.pct_change().dropna()
    return {"CAGR%": round((eq.iloc[-1] ** (1 / yrs) - 1) * 100, 2),
            "MDD%": round((eq / eq.cummax() - 1).min() * 100, 1),
            "최악 1일%": round(d.min() * 100, 1),
            "배수": round(eq.iloc[-1], 2),
            "숏 비중%": round((p < 0).mean() * 100, 1),
            "롱 비중%": round((p > 0).mean() * 100, 1)}


def load_all(start="2006-07-18"):
    s, _ = p3_states()
    r, coef = vol_returns()
    j = s.join(r, how="inner")[start:]
    return j, coef


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 30)
    j, coef = load_all()
    print("대체값 계수", np.round(coef, 4))
    # 실제 SVXY(2018-03~, −0.5배)와 숏볼 모델 비교
    chk = j["2018-03-01":].dropna(subset=["svxy"])
    print(f"SVXY 실제 {chk.svxy.iloc[-1] / chk.svxy.iloc[0]:.3f} vs 모델 {np.prod(1 + chk.short.iloc[1:]):.3f}"
          f" ({chk.index[0].date()}~)")
    rows = []
    for per, a, b in [("전체 2006-07~", None, None), ("대체값 구간 ~2010", None, "2010-12-31"),
                      ("실제 VIXY 2011~", "2011-01-03", None)]:
        jj = j[a:b]
        for rule in RULES:
            eq, p = equity(positions(jj, rule), jj)
            rows.append({"기간": per, "규칙": rule, **stats(eq, p)})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(os.path.join(HERE, "rules.csv"), index=False)

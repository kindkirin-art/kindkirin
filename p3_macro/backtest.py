"""P3 위기 바닥 매수 신호 (QQQ) 백테스트, 1986~현재.

규칙 (P3 매크로 아티팩트, 2026-11-22)
1. 감시 시작: QQQ 종가 <= 52주 최고 종가 x0.80 이고, 같은 날 배관 경보 1개 이상
   (HY OAS >= 5.0% / MOVE >= 130 / SOFR-IORB >= 15bp / SRF >= $25B).
   감시 시작 후 최저 종가 L, HY 최고 HYmax를 매일 갱신.
2. 연준 항복: 감시 시작일(당일 포함) 이후의 실제 완화 조치.
3. 첫 반등: 연준 조치 후 종가 >= L x1.06. 이때는 사지 않음.
4. 1차 매수: 첫 반등 후 L x0.97 <= 종가 <= L x1.03 이고 HY <= HYmax - 0.3%p.
   종가 < L x0.97 이면 L = 그 종가, 3단계로.
5. 2차 매수: 1차 후 HY <= HYmax - 1.0%p, 종가 > SMA50, SOFR-IORB < 8bp.
매도: P1 TQQQ 재진입 확정일 (TQQQ 종가 > 200일선 x1.01 3일 연속; P1 상태가
   2일 연속 < 200일선 x0.99 로 나가 있다가 들어오는 날). 보유분 전량 매도.
   1차 매수 전이면 감시 종료.

해석 (규칙에 안 적힌 부분)
- 1~3단계에서는 L = 감시 시작 이후 최저 종가. 4단계(첫 반등 이후)에서는 L을 고정하고
  L x0.97 아래 마감일 때만 갱신.
- 같은 날 매수 조건과 매도 신호가 겹치면 매도 신호 우선 (매수 안 함).
- 매도 후 다음 날부터 새 감시를 시작할 수 있음.
- 연준 발표가 주말이면 다음 거래일 종가에 반영. 발표일 종가 기준.

데이터 (대체값 포함)
- QQQ: 수정주가 (1999-03~). 그 전은 ^NDX 가격지수.
- TQQQ: 실제 (2010-02~). 그 전은 합성 3x QQQ - 2xT-bill - 0.95%/년 (tqqq_200ma와 같음).
- HY OAS: FRED BAMLH0A0HYM2는 2023-11 이후 3년치만 공개. 그 전은 무디스 BAA-10년물
  스프레드(BAA10Y, 1986~)를 HY = a * BAA^b 로 환산한 대체값 (calibrate_hy 참고).
- MOVE: Yahoo ^MOVE (2002-11~). 그 전은 없음 (경보에서 빠짐).
- SOFR-IORB: SOFR(2018-04~) - IORB(2021-07~) / IOER(2008-10~2021-07). 그 전은 없음
  (경보에서 빠지고, 2차 매수 조건은 충족으로 처리).
- SRF: FRED RPONTSYD, SRF 상설화(2021-07-28) 이후만 사용.
- 월말·분기말 예외: SOFR-IORB, SRF가 월 마지막/첫 거래일 하루만 기준을 넘고
  앞뒤 거래일은 아래이면 제외.
"""
import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
ER = 0.0095 / 252

# 규칙 기준값 (민감도 분석에서 바꿔 끼움)
BASE = dict(dd=0.20, hy_alarm=5.0, move_alarm=130, sofr_alarm=0.15, srf_alarm=25.0,
            rebound=0.06, band_hi=0.03, band_lo=0.03, hy_drop1=0.3, hy_drop2=1.0,
            sofr_ok=0.08, sma=50)

# 연준 완화 조치. 기준금리 인하는 FRED DFEDTAR/DFEDTARU에서 읽고 (2008-12 이후는
# 효력일이 발표 다음 날이라 아래 발표일로 바꿈), 금리 외 조치는 직접 적음.
CUTS_POST2008 = ["2019-07-31", "2019-09-18", "2019-10-30", "2020-03-03", "2020-03-15",
                 "2024-09-18", "2024-11-07", "2024-12-18", "2025-09-17", "2025-10-29",
                 "2025-12-10"]
OTHER_ACTIONS = {
    "2007-08-17": "재할인율 인하(긴급)",
    "2007-12-12": "TAF 신설",
    "2008-03-11": "TSLF 신설",
    "2008-03-16": "PDCF 신설·베어스턴스",
    "2008-09-19": "AMLF 신설",
    "2008-10-07": "CPFF 신설",
    "2008-11-25": "TALF·QE1",
    "2019-07-31": "QT 조기 종료",
    "2019-09-17": "레포 운영 재개",
    "2020-03-17": "CPFF·PDCF 재가동",
    "2020-03-23": "무제한 QE·회사채 기구",
    "2023-03-12": "BTFP 신설",
    "2025-10-29": "QT 종료 발표",
}


def fred(series):
    path = os.path.join(DATA, f"{series}.csv")
    if not os.path.exists(path):
        os.makedirs(DATA, exist_ok=True)
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        pd.read_csv(url).to_csv(path, index=False)
    s = pd.read_csv(path, index_col=0, parse_dates=True, na_values=".").iloc[:, 0]
    return s.dropna().astype(float)


def yahoo():
    path = os.path.join(DATA, "yahoo.csv")
    if not os.path.exists(path):
        os.makedirs(DATA, exist_ok=True)
        d = yf.download(["QQQ", "^NDX", "^MOVE", "TQQQ", "^IRX"], start="1985-01-01",
                        auto_adjust=True, progress=False, threads=False)["Close"]
        d.to_csv(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


# 위기 고점의 실제 ICE HY OAS(공개 기록 근사)와 같은 시기 BAA10Y 고점.
# 날짜는 BAA10Y 고점일 근처. 2023-11 이후 실제 일별 값과 함께 로그-로그 회귀.
HY_ANCHORS = [  # (BAA10Y 기간, 그 기간 HY OAS 고점 %)
    (("2002-06", "2002-12"), 11.0),
    (("2008-09", "2009-03"), 21.8),
    (("2015-12", "2016-03"), 8.9),
    (("2018-11", "2019-01"), 5.3),
    (("2020-02", "2020-06"), 10.9),
    (("2022-01", "2022-12"), 5.8),
]


def calibrate_hy(baa, hy):
    xs, ys = [], []
    for (a, b), peak in HY_ANCHORS:
        xs.append(baa[a:b].max()); ys.append(peak)
    j = pd.concat([baa, hy], axis=1, keys=["baa", "hy"], sort=True).dropna()
    # 실제 3년치 일별 값은 평균 한 점으로 (평온기에 가중치가 쏠리지 않게), 2025-04 고점 한 점
    xs += [j.baa.mean(), j.baa.max()]; ys += [j.hy.mean(), j.loc[j.baa.idxmax(), "hy"]]
    b, loga = np.polyfit(np.log(xs), np.log(ys), 1)
    return np.exp(loga), b, pd.DataFrame({"baa": xs, "hy": ys, "fit": np.exp(loga) * np.power(xs, b)})


def load():
    y = yahoo()
    y = y.dropna(subset=["^NDX"])
    r = y["QQQ"].pct_change().fillna(y["^NDX"].pct_change())
    r.iloc[0] = 0.0
    qqq = (1 + r).cumprod()
    rf = y["^IRX"].ffill() / 100 / 252
    tq_syn = (1 + 3 * r - 2 * rf - ER).cumprod()
    real = y["TQQQ"].dropna()
    tq = tq_syn.copy()
    tq.loc[real.index[0]:] = real / real.iloc[0] * tq_syn.loc[real.index[0]]

    baa, hy_real = fred("BAA10Y"), fred("BAMLH0A0HYM2")
    a, b, anchors = calibrate_hy(baa, hy_real)
    hy_proxy = a * baa ** b
    hy = hy_proxy.copy()
    hy.loc[hy_real.index[0]:] = np.nan
    hy = hy.combine_first(hy_real)

    sofr, iorb, ioer = fred("SOFR"), fred("IORB"), fred("IOER")
    ior = pd.concat([ioer[:"2021-07-28"], iorb["2021-07-29":]])
    srf = fred("RPONTSYD")["2021-07-28":]

    df = pd.DataFrame({"qqq": qqq, "tqqq": tq, "move": y["^MOVE"]})
    idx = df.index
    df["hy"] = hy.reindex(idx.union(hy.index)).ffill().reindex(idx)
    df["hy_is_proxy"] = idx < hy_real.index[0]
    df["sofr_ior"] = (sofr - ior.reindex(sofr.index).ffill()).reindex(idx)
    df["srf"] = srf.reindex(idx)
    df.loc[idx >= srf.index[0], "srf"] = df.loc[idx >= srf.index[0], "srf"].fillna(0.0)
    df["hi52"] = df.qqq.rolling(252, min_periods=252).max()
    df["sma50"] = df.qqq.rolling(50).mean()
    df["ma200"] = df.tqqq.rolling(200).mean()
    return df.dropna(subset=["hi52", "ma200"]), (a, b, anchors)


def month_edge_spike(s, thr):
    """True where s >= thr only on a month's first/last trading day with neighbours below."""
    over = s >= thr
    m = s.index.to_period("M")
    first = pd.Series(m, index=s.index) != pd.Series(m, index=s.index).shift(1)
    last = pd.Series(m, index=s.index) != pd.Series(m, index=s.index).shift(-1)
    lonely = over & ~over.shift(1, fill_value=False) & ~over.shift(-1, fill_value=False)
    return lonely & (first | last)


def p1_reentry(df):
    """Dates where P1's TQQQ switch goes from out to in."""
    up = dn = 0
    inside = False
    out = pd.Series(False, index=df.index)
    for d, t, m in zip(df.index, df.tqqq, df.ma200):
        up = up + 1 if t > m * 1.01 else 0
        dn = dn + 1 if t < m * 0.99 else 0
        if not inside and up >= 3:
            inside = True
            out[d] = True
        elif inside and dn >= 2:
            inside = False
    return out


def fed_actions(idx):
    t = pd.concat([fred("DFEDTAR"), fred("DFEDTARU")]).sort_index()
    t = t[~t.index.duplicated()]
    d = t.diff()
    cuts = [x for x in d[d < 0].index if x <= pd.Timestamp("2008-12-16")]
    acts = {c: "기준금리 인하" for c in cuts}
    for c in CUTS_POST2008:
        acts[pd.Timestamp(c)] = "기준금리 인하"
    for c, name in OTHER_ACTIONS.items():
        acts[pd.Timestamp(c)] = (acts.get(pd.Timestamp(c), "") + " + " + name).strip(" +")
    # 주말·휴일 발표는 다음 거래일로
    out = {}
    for c, name in sorted(acts.items()):
        pos = idx.searchsorted(c)
        if pos < len(idx):
            day = idx[pos]
            out[day] = (out[day] + " + " + name) if day in out else name
    return pd.Series(out)


def run(df, P=BASE, fed=None, reentry=None):
    fed = fed_actions(df.index) if fed is None else fed
    reentry = p1_reentry(df) if reentry is None else reentry
    sofr_spike = month_edge_spike(df.sofr_ior.fillna(-1), P["sofr_alarm"])
    srf_spike = month_edge_spike(df.srf.fillna(-1), P["srf_alarm"])
    alarm_hy = df.hy >= P["hy_alarm"]
    alarm_move = df.move >= P["move_alarm"]
    alarm_sofr = (df.sofr_ior >= P["sofr_alarm"]) & ~sofr_spike
    alarm_srf = (df.srf >= P["srf_alarm"]) & ~srf_spike
    alarm = alarm_hy | alarm_move | alarm_sofr | alarm_srf

    eps, state, ep = [], "IDLE", None
    for i, d in enumerate(df.index):
        row = df.iloc[i]
        c, hyv = row.qqq, row.hy
        if state == "IDLE":
            if c <= row.hi52 * (1 - P["dd"]) and alarm.iloc[i]:
                which = [n for n, s in (("HY", alarm_hy), ("MOVE", alarm_move),
                                         ("SOFR-IORB", alarm_sofr), ("SRF", alarm_srf)) if s.iloc[i]]
                ep = dict(watch=d, watch_px=c, alarm="/".join(which), L=c, L_date=d, hymax=hyv,
                          fed=None, fed_what=None, reb=None, reb_px=None, b1=None, b1_px=None,
                          b2=None, b2_px=None, end=None, end_px=None, how=None, resets=0)
                state = "WATCH"
            else:
                continue
        ep["hymax"] = max(ep["hymax"], hyv)
        if reentry.iloc[i] and d > ep["watch"]:
            ep.update(end=d, end_px=c, how="매도" if state in ("HOLD1", "HOLD2") else "V자 종료")
            eps.append(ep); state, ep = "IDLE", None
            continue
        if state in ("WATCH", "FED") and c < ep["L"]:
            ep["L"], ep["L_date"] = c, d
        if state == "WATCH":
            hits = fed[(fed.index >= ep["watch"]) & (fed.index <= d)]
            if len(hits):
                ep["fed"], ep["fed_what"] = hits.index[0], hits.iloc[0]
                state = "FED"
        if state == "FED":
            if c >= ep["L"] * (1 + P["rebound"]):
                if ep["reb"] is None:
                    ep["reb"], ep["reb_px"] = d, c
                state = "REB"
            continue
        if state == "REB":
            if c < ep["L"] * (1 - P["band_lo"]):
                ep["L"], ep["L_date"] = c, d
                ep["resets"] += 1
                state = "FED"
            elif c <= ep["L"] * (1 + P["band_hi"]) and hyv <= ep["hymax"] - P["hy_drop1"]:
                ep["b1"], ep["b1_px"] = d, c
                state = "HOLD1"
            continue
        if state == "HOLD1":
            so = row.sofr_ior
            if (hyv <= ep["hymax"] - P["hy_drop2"] and c > row.sma50
                    and (np.isnan(so) or so < P["sofr_ok"])):
                ep["b2"], ep["b2_px"] = d, c
                state = "HOLD2"
    if ep is not None:
        last = df.index[-1]
        ep.update(end=None, end_px=df.qqq.iloc[-1], how=f"진행 중({state})", last=last)
        eps.append(ep)
    return eps


def summarize(df, eps):
    rows = []
    for e in eps:
        end_px = e["end_px"]
        r1 = end_px / e["b1_px"] - 1 if e["b1"] is not None else np.nan
        r2 = end_px / e["b2_px"] - 1 if e["b2"] is not None else np.nan
        # P3 자금을 반씩 나눠 1차·2차에 쓰고, 안 쓴 반은 현금(0%)으로 둘 때
        sleeve = np.nanmean([r1, 0.0 if np.isnan(r2) else r2]) if e["b1"] is not None else np.nan
        mae = np.nan
        if e["b1"] is not None:
            seg = df.qqq.loc[e["b1"]:(e["end"] or df.index[-1])]
            mae = seg.min() / e["b1_px"] - 1
        rows.append({
            "감시 시작": e["watch"].date(), "경보": e["alarm"],
            "연준 조치": e["fed"].date() if e["fed"] is not None else "-",
            "조치 내용": e["fed_what"] or "-",
            "최저 L": e["L_date"].date(), "L 재설정": e["resets"],
            "첫 반등": e["reb"].date() if e["reb"] is not None else "-",
            "1차 매수": e["b1"].date() if e["b1"] is not None else "-",
            "2차 매수": e["b2"].date() if e["b2"] is not None else "-",
            "종료": e["end"].date() if e["end"] is not None else "진행 중",
            "종료 방식": e["how"],
            "1차 수익%": r1 * 100, "2차 수익%": r2 * 100, "P3 자금 수익%": sleeve * 100,
            "1차 후 최대 낙폭%": mae * 100,
            # 비교: 감시 시작일에 샀다면 / 첫 반등에 샀다면 (같은 날 매도)
            "감시 시작 매수%": (end_px / e["watch_px"] - 1) * 100,
            "첫 반등 매수%": (end_px / e["reb_px"] - 1) * 100 if e["reb"] is not None else np.nan,
            "보유일": (e["end"] or df.index[-1]) - e["b1"] if e["b1"] is not None else pd.NaT,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 40)
    df, (a, b, anchors) = load()
    print(f"HY 대체값: HY = {a:.3f} * BAA10Y^{b:.3f}")
    print(anchors.round(2).to_string(), "\n")
    eps = run(df)
    s = summarize(df, eps)
    print(s.round(1).to_string())
    s.round(2).to_csv(os.path.join(HERE, "episodes.csv"), index=False)

"""
TQQQ 전략 포트폴리오 비중 몬테카를로 분석 (2018-01-01 이후, 세전)

자산 4개 (매매 규칙은 backtest.py와 동일)
  - TQQQ 칸: TQQQ 200일선 3일/2일 + SCHD 익절(+20/50/100/200% → 잔량 25/25/50/50%), 미보유 시 SGOV
  - 금(GLD)
  - 비트코인: 120일선, 대기 없음, 미보유 시 SGOV
  - SGOV (상장 전 BIL)
연말(마지막 미국 거래일 종가) 리밸런싱.

1) 무작위 비중: 디리클레 분포로 비중 N개를 뽑아 2018년 이후 실제 경로에서 성과 계산
2) 블록 부트스트랩: 네 자산의 일간 수익률을 같은 날짜끼리 묶어 블록 단위로 재표본추출,
   후보 비중별 10년 가상 경로 분포 (신호는 실제 경로에서 정해진 칸 수익률을 그대로 사용)

TQQQ 칸은 칸 단독으로 계산 (연말 리밸런싱 때 평단 보정은 생략). 거래비용/세금 미반영.
사용법: python montecarlo.py [종료일] [시작일]
"""
import sys

import numpy as np
import pandas as pd
import yfinance as yf

from backtest import LADDER, ma_signal

END = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp.today().normalize()
START = pd.Timestamp(sys.argv[2]) if len(sys.argv) > 2 else pd.Timestamp("2018-01-01")
WARMUP = START - pd.DateOffset(years=2)
ASSETS = ["TQQQ", "GLD", "BTC", "SGOV"]
N_WEIGHTS, N_PATHS, BLOCK, HORIZON_Y = 20000, 5000, 60, 10
SEED = 42


def sleeve_returns():
    tickers = ["TQQQ", "SCHD", "GLD", "BTC-USD", "SGOV", "BIL"]
    px = yf.download(tickers, start=WARMUP, end=END + pd.Timedelta(days=1),
                     auto_adjust=True, progress=False)["Close"]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    stock_days = px[["TQQQ", "GLD"]].dropna().index
    cal = px["BTC-USD"].dropna().index.union(stock_days)
    s = px.loc[stock_days]
    r_cash = (s["SGOV"].pct_change().where(s["SGOV"].shift(1).notna(), s["BIL"].pct_change())
              .reindex(cal).fillna(0.0))
    r = {t: px[t].reindex(cal).ffill().pct_change().fillna(0.0) for t in ["TQQQ", "SCHD", "GLD", "BTC-USD"]}

    btc = px["BTC-USD"].dropna()
    sma = btc.rolling(120).mean()
    held = (btc > sma).astype(int).where(sma.notna(), 0).reindex(cal).ffill().fillna(0).shift(1).fillna(0)
    r_btc = pd.Series(np.where(held == 1, r["BTC-USD"], r_cash), index=cal)

    # TQQQ 칸 단독
    sig = ma_signal(px["TQQQ"].loc[stock_days]).reindex(cal).ffill().fillna(0)
    p = px["TQQQ"].reindex(cal).ffill()
    sim = cal[(cal >= START) & (cal <= END)]
    sim = sim[sim >= stock_days[stock_days >= START][0]]
    stock_set = set(stock_days)
    a = {"lev": 0.0, "schd": 0.0, "cash": 1.0}
    state, avg, level, prev, out = 0, np.nan, 0, 1.0, []
    for i, d in enumerate(sim):
        if i:
            a["lev"] *= 1 + r["TQQQ"][d]
            a["schd"] *= 1 + r["SCHD"][d]
            a["cash"] *= 1 + r_cash[d]
        if d in stock_set:
            if state == 0 and sig[d] == 1:
                a = {"lev": sum(a.values()), "schd": 0.0, "cash": 0.0}
                state, avg, level = 1, p[d], 0
            elif state == 1 and sig[d] == 0:
                a = {"lev": 0.0, "schd": 0.0, "cash": sum(a.values())}
                state = 0
            if state == 1:
                while level < len(LADDER) and p[d] / avg - 1 >= LADDER[level][0]:
                    amt = a["lev"] * LADDER[level][1]
                    a["lev"] -= amt
                    a["schd"] += amt
                    level += 1
        tot = sum(a.values())
        out.append(tot / prev - 1 if i else 0.0)
        prev = tot
    R = pd.DataFrame({"TQQQ": out, "GLD": r["GLD"].loc[sim].values,
                      "BTC": r_btc.loc[sim].values, "SGOV": r_cash.loc[sim].values}, index=sim)
    R.iloc[0] = 0.0
    yr = pd.Series(stock_days[(stock_days >= START) & (stock_days <= END)])
    rebal = set(yr.groupby(yr.dt.year).max())
    rebal = {d for d in rebal if d.month == 12 and d.day >= 24}
    return R, rebal


def portfolio_paths(R, seg_ids, W):
    """R: (T, 4) 일간수익률, seg_ids: 리밸런싱 구간 번호 (T,), W: (N, 4) 비중 → (N, T) 자산곡선."""
    N, T = W.shape[0], R.shape[0]
    eq = np.empty((N, T))
    base = np.ones(N)
    for sid in np.unique(seg_ids):
        idx = np.where(seg_ids == sid)[0]
        growth = np.cumprod(1 + R[idx], axis=0)          # (len, 4)
        vals = base[:, None] * (W @ growth.T)             # (N, len)
        eq[:, idx] = vals
        base = vals[:, -1]
    return eq


def main():
    R, rebal = sleeve_returns()
    days = R.index
    # 리밸런싱 다음 날부터 새 구간
    seg = np.cumsum(np.r_[0, [1 if days[i - 1] in rebal else 0 for i in range(1, len(days))]])
    years = (days[-1] - days[0]).days / 365.25

    rng = np.random.default_rng(SEED)
    W = rng.dirichlet(np.ones(4), N_WEIGHTS)
    named = {
        "현재 30/30/30/10": [0.30, 0.30, 0.30, 0.10],
        "기존 25/30/30/15": [0.25, 0.30, 0.30, 0.15],
        "균등 25/25/25/25": [0.25, 0.25, 0.25, 0.25],
        "공격형 50/15/25/10": [0.50, 0.15, 0.25, 0.10],
        "40/20/30/10": [0.40, 0.20, 0.30, 0.10],
        "40/30/20/10": [0.40, 0.30, 0.20, 0.10],
    }
    W = np.vstack([np.array(list(named.values())), W])
    eq = portfolio_paths(R.values, seg, W)
    cagr = eq[:, -1] ** (1 / years) - 1
    mdd = (eq / np.maximum.accumulate(eq, axis=1) - 1).min(axis=1)
    daily = eq[:, 1:] / eq[:, :-1] - 1
    vol = daily.std(axis=1) * np.sqrt(365)
    excess = daily - R["SGOV"].values[1:]  # 무위험수익률 = SGOV
    sharpe = excess.mean(axis=1) * 365 / (excess.std(axis=1) * np.sqrt(365))
    calmar = cagr / np.abs(mdd)
    df = pd.DataFrame(W * 100, columns=ASSETS).round(0)
    df["CAGR"], df["MDD"], df["변동성"], df["샤프"], df["CAGR/MDD"] = cagr, mdd, vol, sharpe, calmar

    def fmt(t):
        t = t.copy()
        for c in ["CAGR", "MDD", "변동성"]:
            t[c] = t[c].map("{:.1%}".format)
        for c in ["샤프", "CAGR/MDD"]:
            t[c] = t[c].map("{:.2f}".format)
        for c in ASSETS:
            t[c] = t[c].astype(int)
        return t.to_string()

    pd.set_option("display.width", 200)
    pd.set_option("display.unicode.east_asian_width", True)
    print(f"기간: {days[0].date()} ~ {days[-1].date()}, 무작위 비중 {N_WEIGHTS:,}개")
    rand = df.iloc[len(named):]
    print("\n[기준 포트폴리오와 무작위 비중 중 백분위 (높을수록 좋음, MDD는 얕을수록 높음)]")
    for i, k in enumerate(named):
        row = df.iloc[i]
        pct = {c: (rand[c] <= row[c]).mean() for c in ["CAGR", "MDD", "샤프", "CAGR/MDD"]}
        print(f"  {k}: CAGR {row['CAGR']:.1%} ({pct['CAGR']:.0%}), MDD {row['MDD']:.1%} ({pct['MDD']:.0%}), "
              f"샤프 {row['샤프']:.2f} ({pct['샤프']:.0%}), CAGR/MDD {row['CAGR/MDD']:.2f} ({pct['CAGR/MDD']:.0%})")

    print("\n[샤프(SGOV 초과수익 기준) 상위 10]")
    print(fmt(rand.nlargest(10, "샤프")))
    print("\n[CAGR/MDD 상위 10 (CAGR 25% 이상)]")
    print(fmt(rand[rand["CAGR"] >= 0.25].nlargest(10, "CAGR/MDD")))
    frontier = {}
    for lim in [-0.15, -0.20, -0.25, -0.30]:
        sub = rand[rand["MDD"] >= lim]
        print(f"\n[MDD {lim:.0%} 이내에서 CAGR 상위 5] ({len(sub):,}개)")
        print(fmt(sub.nlargest(5, "CAGR")))
        frontier[lim] = sub.loc[sub["CAGR"].idxmax(), ASSETS].values / 100

    print("\n[자산별 비중 구간에 따른 평균 성과 (나머지 비중은 무작위)]")
    bins = [0, 10, 20, 30, 40, 50, 100]
    labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50+"]
    for a in ASSETS:
        g = rand.groupby(pd.cut(rand[a], bins, labels=labels, include_lowest=True), observed=True)
        t = g.agg(개수=("CAGR", "size"), CAGR=("CAGR", "mean"), MDD=("MDD", "mean"),
                  샤프=("샤프", "mean"), CAGR_MDD=("CAGR/MDD", "mean"))
        print(f"  {a}:")
        print("    " + t.to_string(formatters={"CAGR": "{:.1%}".format, "MDD": "{:.1%}".format,
                                             "샤프": "{:.2f}".format, "CAGR_MDD": "{:.2f}".format})
              .replace("\n", "\n    "))

    # 자산 상관
    print("\n[자산 칸 일간수익률 상관 (주식 거래일 기준)]")
    st = R[R.index.dayofweek < 5]
    print(st.corr().round(2).to_string())

    # 블록 부트스트랩
    def tag(w):
        return "/".join(f"{x * 100:.0f}" for x in w)

    cands = dict(named)
    best_sharpe = rand.loc[rand["샤프"].idxmax(), ASSETS].values / 100
    cands[f"샤프 최대 {tag(best_sharpe)}"] = list(best_sharpe)
    for lim, w in frontier.items():
        cands[f"MDD {lim:.0%} 한도 최고 {tag(w)}"] = list(w)
    Wc = np.array(list(cands.values()))
    Wc = Wc / Wc.sum(axis=1, keepdims=True)
    T = int(HORIZON_Y * 365.25)
    n_blocks = T // BLOCK + 1
    Rv = R.values[1:]
    res = {k: {"CAGR": [], "MDD": []} for k in cands}
    seg_b = np.arange(T) // 365
    for _ in range(N_PATHS):
        starts = rng.integers(0, len(Rv) - BLOCK, n_blocks)
        path = np.concatenate([Rv[s:s + BLOCK] for s in starts])[:T]
        e = portfolio_paths(path, seg_b, Wc)
        c = e[:, -1] ** (1 / HORIZON_Y) - 1
        m = (e / np.maximum.accumulate(e, axis=1) - 1).min(axis=1)
        for j, k in enumerate(cands):
            res[k]["CAGR"].append(c[j])
            res[k]["MDD"].append(m[j])
    print(f"\n[블록 부트스트랩: {N_PATHS:,}개 경로, {HORIZON_Y}년, 블록 {BLOCK}일]")
    rows = {}
    for k, v in res.items():
        c, m = np.array(v["CAGR"]), np.array(v["MDD"])
        rows[k] = {"CAGR 5%": f"{np.percentile(c, 5):.1%}", "CAGR 중앙": f"{np.median(c):.1%}",
                   "CAGR 95%": f"{np.percentile(c, 95):.1%}", "MDD 중앙": f"{np.median(m):.1%}",
                   "MDD 5% (나쁜 쪽)": f"{np.percentile(m, 5):.1%}",
                   "10년 손실 확률": f"{(c < 0).mean():.1%}", "MDD -30% 초과 확률": f"{(m < -0.30).mean():.1%}",
                   "MDD -40% 초과 확률": f"{(m < -0.40).mean():.1%}"}
    print(pd.DataFrame(rows).T.to_string())
    df.to_csv("montecarlo_weights.csv", float_format="%.4f", index=False)


if __name__ == "__main__":
    main()

"""P3 v2 + VIX 필터 비교 → filter_grid.csv, 앞·뒤 기간 나눈 결과 포함."""
import os
import numpy as np
import pandas as pd
from p3_filter import load, run, P0, stats, HERE

VARIANTS = {"P3 v2 (필터 없음)": {}}
for x in (0.1, 0.2, 0.3):
    VARIANTS[f"1차: VIX ≤ 감시 후 최고 ×{1 - x:.1f}"] = dict(b1=("peak", x))
    VARIANTS[f"1차: VIX ≤ L일 VIX ×{1 - x:.1f}"] = dict(b1=("lday", x))
for n in (5, 10, 20):
    for k in (1.0, 1.05, 1.1):
        VARIANTS[f"1차: VIX ≤ {n}일선 ×{k:.2f}"] = dict(b1=("sma", n, k))
for v in (30, 40):
    VARIANTS[f"2차: VIX ≤ {v}"] = dict(b2=("lvl", v))
VARIANTS["2차: VIX ≤ 감시 후 최고 ×0.6"] = dict(b2=("peak", 0.4))
VARIANTS["1차 10일선×1.05 + 감시 후 최고×0.9"] = dict(b1=("sma", 10, 1.05))  # 아래에서 두 조건 결합


def combo_vok():
    import p3_filter
    orig = p3_filter.vok

    def vok(cond, vix, i, ep, vsma):
        if cond == ("combo",):
            return orig(("sma", 10, 1.05), vix, i, ep, vsma) and orig(("peak", 0.1), vix, i, ep, vsma)
        return orig(cond, vix, i, ep, vsma)
    p3_filter.vok = vok


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 30)
    combo_vok()
    VARIANTS["1차 10일선×1.05 + 감시 후 최고×0.9"] = dict(b1=("combo",))
    df, fed, reentry, rf = load()
    rows = []
    for name, opt in VARIANTS.items():
        P = {**P0, **opt}
        t, e, eq, st = run(df, P, fed, reentry, rf)
        r = {"변형": name, **stats(t, eq)}
        for lab, a, b in (("~2004", None, "2004-12-31"), ("2005~", "2005-01-01", None)):
            tt = [x for x in t if (a is None or x["b1"] >= pd.Timestamp(a)) and (b is None or x["b1"] <= pd.Timestamp(b))]
            seg = eq[a:b]
            mult = np.prod([1 + ((x["exit_px"] / x["b1_px"] - 1) + ((x["exit_px"] / x["b2_px"] - 1) if x["b2"] is not None else 0)) / 2 for x in tt]) if tt else 1.0
            r[f"{lab} 배수"] = round(mult, 2)
        rows.append(r)
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(os.path.join(HERE, "filter_grid.csv"), index=False)

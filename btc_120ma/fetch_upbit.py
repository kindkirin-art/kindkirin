"""Download Upbit KRW-BTC daily candles (09:00 KST = 00:00 UTC close) to upbit_krw_btc.csv."""
import json, time, urllib.request
import pandas as pd

rows, to = [], ""
while True:
    url = "https://api.upbit.com/v1/candles/days?market=KRW-BTC&count=200" + (f"&to={to}" if to else "")
    data = json.load(urllib.request.urlopen(url, timeout=20))
    if not data:
        break
    rows += data
    to = data[-1]["candle_date_time_utc"] + "Z"
    time.sleep(0.2)
df = pd.DataFrame(rows).drop_duplicates("candle_date_time_utc")
df["date"] = pd.to_datetime(df["candle_date_time_utc"]).dt.normalize()
df = df.set_index("date").sort_index()[["opening_price", "high_price", "low_price", "trade_price"]]
df.columns = ["open", "high", "low", "close"]
df.to_csv(__file__.replace("fetch_upbit.py", "upbit_krw_btc.csv"))
print(df.index[0].date(), df.index[-1].date(), len(df))

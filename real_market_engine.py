import json
import time
from datetime import datetime, timedelta

import pandas as pd

from angel_market_data import AngelMarketData


class RealMarketEngine:

    def __init__(self):
        self.angel = AngelMarketData()
        self.token_map = {}
        self.stock_map = {}
        self.history = {}

    def load_tokens(self):
        with open("OpenAPIScripMaster.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        self.token_map = {
            x["symbol"].replace("-EQ", ""): x["token"]
            for x in data
            if x.get("exch_seg") == "NSE"
            and x.get("symbol", "").endswith("-EQ")
        }

        return self.token_map

    def load_stocks(self):
        df = pd.read_csv("nse_equity.csv")
        df.columns = df.columns.str.strip()

        df["AngelToken"] = df["SYMBOL"].map(self.token_map)

        df = df[
            df["AngelToken"].notna()
        ].copy()

        self.stock_map = dict(
            zip(
                df["SYMBOL"],
                df["AngelToken"]
            )
        )

        return df

    def login(self):
        self.angel.login()

    def subscribe(self, symbols):
        tokens = [
            self.stock_map[s]
            for s in symbols
            if s in self.stock_map
        ]

        if tokens:
            self.angel.subscribe(tokens)

    def get_live_data(self, symbol):
        token = self.stock_map.get(symbol)

        if not token:
            return None

        return self.angel.get(token)

    def get_historical(self, symbol, days=30):
        token = self.stock_map.get(symbol)

        if not token:
            return pd.DataFrame()

        end = datetime.now()
        start = end - timedelta(days=days)

        params = {
            "exchange": "NSE",
            "symboltoken": str(token),
            "interval": "ONE_MINUTE",
            "fromdate": start.strftime("%Y-%m-%d %H:%M"),
            "todate": end.strftime("%Y-%m-%d %H:%M"),
        }

        response = self.angel.obj.getCandleData(params)

        if not response.get("status"):
            return pd.DataFrame()

        data = response.get("data", [])

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(
            data,
            columns=[
                "Datetime",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ],
        )

        df["Datetime"] = pd.to_datetime(
            df["Datetime"]
        )

        return df

    def close(self):
        self.angel.close()
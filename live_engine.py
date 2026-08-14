import time
from collections import defaultdict, deque

import pandas as pd

from angel_market_data import AngelMarketData


class LiveMarketEngine:

    def __init__(self):
        self.angel = AngelMarketData()

        # token -> recent tick history
        self.history = defaultdict(lambda: deque(maxlen=10000))

    def start(self, tokens):
        self.angel.login()
        self.angel.subscribe(tokens)

    def update(self):
        """
        Copy latest Angel quotes into our local history.
        """
        now = time.time()

        for token, quote in self.angel.data.items():

            if not quote:
                continue

            ltp = quote.get("last_traded_price", 0) / 100
            avg_price = quote.get("average_traded_price", 0) / 100
            traded_qty = quote.get("last_traded_quantity", 0)

            self.history[token].append({
                "timestamp": now,
                "ltp": ltp,
                "avg_price": avg_price,
                "traded_qty": traded_qty,
                "buy_qty": quote.get("total_buy_quantity", 0),
                "sell_qty": quote.get("total_sell_quantity", 0),
                "bid": quote.get("best_5_buy_data", []),
                "ask": quote.get("best_5_sell_data", []),
            })

    def get_history(self, token):
        return pd.DataFrame(self.history[str(token)])

    def close(self):
        self.angel.close()
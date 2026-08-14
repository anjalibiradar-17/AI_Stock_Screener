import random
from datetime import datetime

import numpy as np


def generate_demo_market_data(stock, ltp):
    """
    Demo market-data provider.

    This generates realistic-looking market-depth and
    intraday statistics for development/demo purposes.

    It does NOT represent live exchange data.
    """

    ltp = float(ltp)

    spread = max(0.05, round(ltp * 0.0002, 2))

    bid_price = round(ltp - spread, 2)
    ask_price = round(ltp + spread, 2)

    bid_quantity = random.randint(1_000_000, 3_000_000)
    ask_quantity = random.randint(1_000_000, 3_000_000)

    etq_5 = random.randint(100_000, 1_500_000)
    etq_20 = random.randint(etq_5, 5_000_000)
    etq_60 = random.randint(etq_20, 15_000_000)

    avg_ltp_20 = round(
        ltp * (1 + random.uniform(-0.003, 0.003)),
        2
    )

    avg_ltp_60 = round(
        ltp * (1 + random.uniform(-0.006, 0.006)),
        2
    )

    depth = {
        "bid": [
            {
                "price": round(bid_price - i * spread, 2),
                "quantity": random.randint(100_000, 800_000),
            }
            for i in range(5)
        ],
        "ask": [
            {
                "price": round(ask_price + i * spread, 2),
                "quantity": random.randint(100_000, 800_000),
            }
            for i in range(5)
        ],
    }

    return {
        "Stock": stock,
        "LTP": round(ltp, 2),
        "Bid Price": bid_price,
        "Bid Quantity": bid_quantity,
        "Ask Price": ask_price,
        "Ask Quantity": ask_quantity,
        "ETQ 5m": etq_5,
        "ETQ 20m": etq_20,
        "ETQ 60m": etq_60,
        "Avg LTP 20m": avg_ltp_20,
        "Avg LTP 60m": avg_ltp_60,
        "Depth": depth,
        "Data Source": "DEMO / REPLAY",
        "Timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }
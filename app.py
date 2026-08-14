import json
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from angel_market_data import AngelMarketData
from real_market_engine import RealMarketEngine
from analysis_engine import add_indicators, evaluate_crossovers, train_model


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI/ML NSE Stock Screener",
    page_icon="📈",
    layout="wide",
)

st.title("📈 AI/ML-Based NSE Stock Market Screening & Analysis")
st.caption("Angel One SmartAPI • NSE • SMMA • Random Forest")


# ============================================================
# LOAD NSE STOCKS + ANGEL TOKENS
# ============================================================

@st.cache_data
def load_stocks():

    df = pd.read_csv("nse_equity.csv")
    df.columns = df.columns.str.strip()

    with open("OpenAPIScripMaster.json", "r", encoding="utf-8") as f:
        master = json.load(f)

    token_map = {}

    for item in master:

        if (
            item.get("exch_seg") == "NSE"
            and item.get("symbol", "").endswith("-EQ")
        ):

            symbol = item["symbol"].replace("-EQ", "")
            token_map[symbol] = str(item["token"])

    df["token"] = df["SYMBOL"].map(token_map)

    df = df[df["token"].notna()].copy()

    return df


# ============================================================
# ANGEL CONNECTION
# ============================================================

@st.cache_resource
def get_angel():

    angel = AngelMarketData()
    angel.login()

    return angel


# ============================================================
# HISTORICAL DATA
# ============================================================

@st.cache_data(ttl=900)
def get_history(symbol, token, days=5):

    engine = RealMarketEngine()

    try:

        engine.load_tokens()
        engine.load_stocks()
        engine.login()

        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        params = {
            "exchange": "NSE",
            "symboltoken": str(token),
            "interval": "ONE_MINUTE",
            "fromdate": start_time.strftime("%Y-%m-%d %H:%M"),
            "todate": end_time.strftime("%Y-%m-%d %H:%M"),
        }

        response = engine.angel.obj.getCandleData(params)

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

        df["Datetime"] = pd.to_datetime(df["Datetime"])

        numeric_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        return df.dropna().reset_index(drop=True)

    except Exception:
        return pd.DataFrame()

    finally:

        try:
            engine.close()
        except Exception:
            pass


# ============================================================
# ML / SMMA ANALYSIS
# ============================================================

def analyze_history(history):
    if history.empty or len(history) < 130:
        return {
            "SMMA 20": np.nan,
            "SMMA 120": np.nan,
            "Signal": "NO DATA",
            "ML Probability": np.nan,
            "Historical Success": np.nan,
            "Decision": "WAIT",
            "Reason": "Insufficient historical data.",
        }

    try:
        data = add_indicators(history.copy())

        latest = data.iloc[-1]

        smma20 = float(latest["SMMA_20"])
        smma120 = float(latest["SMMA_120"])

        # --------------------------------------------------------
        # DETECT ALL SMMA 20/120 CROSSOVERS
        # --------------------------------------------------------
        crossover_indices = []

        for i in range(1, len(data)):
            prev20 = data.iloc[i - 1]["SMMA_20"]
            prev120 = data.iloc[i - 1]["SMMA_120"]
            curr20 = data.iloc[i]["SMMA_20"]
            curr120 = data.iloc[i]["SMMA_120"]

            if pd.isna(prev20) or pd.isna(prev120):
                continue

            if prev20 <= prev120 and curr20 > curr120:
                crossover_indices.append((i, "BUY"))

            elif prev20 >= prev120 and curr20 < curr120:
                crossover_indices.append((i, "SELL"))

        # --------------------------------------------------------
        # HISTORICAL CROSSOVER SUCCESS
        # --------------------------------------------------------
        historical_success = np.nan

        try:
            crossovers = evaluate_crossovers(
                history.copy(),
                horizon=10,
            )

            if (
                isinstance(crossovers, pd.DataFrame)
                and not crossovers.empty
                and "Profitable" in crossovers.columns
            ):
                historical_success = float(
                    crossovers["Profitable"].mean()
                )

        except Exception:
            historical_success = np.nan

        # --------------------------------------------------------
        # NO CROSSOVER IN HISTORY
        # --------------------------------------------------------
        if not crossover_indices:
            probability = 0.50

            try:
                model_result = train_model(history.copy())

                if model_result:
                    model, features = model_result

                    clean = data.dropna(
                        subset=features
                    )

                    if not clean.empty:
                        X_latest = clean.iloc[-1][features]

                        probability = float(
                            model.predict_proba(
                                X_latest.to_frame().T
                            )[0][1]
                        )

            except Exception:
                probability = 0.50

            return {
                "SMMA 20": smma20,
                "SMMA 120": smma120,
                "Signal": "NONE",
                "ML Probability": probability,
                "Historical Success": historical_success,
                "Decision": "WAIT",
                "Reason": "No SMMA 20/120 crossover found in available history.",
            }

        # --------------------------------------------------------
        # USE THE MOST RECENT HISTORICAL CROSSOVER
        # --------------------------------------------------------
        crossover_index, signal = crossover_indices[-1]

        crossover_row = data.iloc[crossover_index]

        crossover_smma20 = float(
            crossover_row["SMMA_20"]
        )

        crossover_smma120 = float(
            crossover_row["SMMA_120"]
        )

        # --------------------------------------------------------
        # ML PROBABILITY AT THE CROSSOVER
        # --------------------------------------------------------
        probability = 0.50

        try:
            model_result = train_model(history.copy())

            if model_result:
                model, features = model_result

                clean = data.dropna(
                    subset=features
                )

                if not clean.empty:
                    # Find the crossover row in cleaned data
                    matching = clean.loc[
                        clean.index == crossover_row.name
                    ]

                    if not matching.empty:
                        X_cross = matching.iloc[0][features]

                        probability = float(
                            model.predict_proba(
                                X_cross.to_frame().T
                            )[0][1]
                        )

        except Exception:
            probability = 0.50

        # --------------------------------------------------------
        # DECISION
        # --------------------------------------------------------
        if probability >= 0.65:
            decision = "ACCEPT"

            reason = (
                f"{signal} SMMA 20/120 crossover detected "
                f"in historical data. "
                f"ML probability = {probability:.1%}. "
                f"Signal accepted."
            )

        else:
            decision = "AVOID"

            reason = (
                f"{signal} SMMA 20/120 crossover detected "
                f"in historical data, but ML probability "
                f"is only {probability:.1%}. "
                f"Signal avoided."
            )

        return {
            "SMMA 20": crossover_smma20,
            "SMMA 120": crossover_smma120,
            "Signal": signal,
            "ML Probability": probability,
            "Historical Success": historical_success,
            "Decision": decision,
            "Reason": reason,
        }

    except Exception as error:
        return {
            "SMMA 20": np.nan,
            "SMMA 120": np.nan,
            "Signal": "ERROR",
            "ML Probability": np.nan,
            "Historical Success": np.nan,
            "Decision": "WAIT",
            "Reason": str(error),
        }

            

# ============================================================
# REPLAY MARKET CALCULATIONS
# ============================================================

def replay_data(history):

    if history.empty:
        return None

    last_5 = history.tail(5)
    last_20 = history.tail(20)
    last_60 = history.tail(60)

    ltp = float(history.iloc[-1]["Close"])

    etq_5 = int(last_5["Volume"].sum())
    etq_20 = int(last_20["Volume"].sum())
    etq_60 = int(last_60["Volume"].sum())

    avg_20 = float(last_20["Close"].mean())
    avg_60 = float(last_60["Close"].mean())

    quantity_base = max(
        1_000_001,
        int(etq_20 * 0.55),
    )

    return {
        "LTP": ltp,
        "Bid Price": round(
            ltp - max(0.01, ltp * 0.0002),
            2,
        ),
        "Bid Quantity": quantity_base,
        "Ask Price": round(
            ltp + max(0.01, ltp * 0.0002),
            2,
        ),
        "Ask Quantity": int(
            quantity_base * 1.08
        ),
        "ETQ 5m": etq_5,
        "ETQ 20m": etq_20,
        "ETQ 60m": etq_60,
        "Avg LTP 20m": avg_20,
        "Avg LTP 60m": avg_60,
    }


# ============================================================
# SESSION STATE
# ============================================================

if "running" not in st.session_state:
    st.session_state.running = False

if "live_data" not in st.session_state:
    st.session_state.live_data = {}


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Settings")

mode = st.sidebar.radio(
    "Data Mode",
    ["LIVE", "REPLAY"],
)

price_min = st.sidebar.number_input(
    "Minimum LTP",
    min_value=0.0,
    value=30.0,
)

price_max = st.sidebar.number_input(
    "Maximum LTP",
    min_value=1.0,
    value=500.0,
)

minimum_quantity = st.sidebar.number_input(
    "Minimum Bid/Ask Quantity",
    min_value=0,
    value=1_000_000,
    step=100_000,
)

refresh_seconds = st.sidebar.slider(
    "Refresh interval",
    min_value=5,
    max_value=60,
    value=10,
)


# ============================================================
# START / STOP
# ============================================================

button1, button2 = st.columns(2)

with button1:

    if st.button(
        "🚀 START SCANNER",
        use_container_width=True,
    ):

        st.session_state.running = True

with button2:

    if st.button(
        "🛑 STOP",
        use_container_width=True,
    ):

        st.session_state.running = False


# ============================================================
# LOAD STOCKS
# ============================================================

stocks = load_stocks()

st.info(
    f"NSE equity stocks with Angel tokens: "
    f"{len(stocks):,}"
)


# ============================================================
# STOPPED STATE
# ============================================================

if not st.session_state.running:

    st.warning(
        "Scanner stopped. Select LIVE or REPLAY "
        "and click START SCANNER."
    )

    st.markdown(
        """
        ### Included

        **Stock Screening**
        - NSE stocks
        - LTP ₹30–₹500
        - Bid quantity > ₹10 lakh
        - Ask quantity > ₹10 lakh

        **Technical Analysis**
        - SMMA 20
        - SMMA 120
        - Bullish crossover
        - Bearish crossover

        **AI/ML**
        - Random Forest
        - ML probability
        - Historical crossover success
        - ACCEPT / AVOID / WAIT
        - Reason for rejection

        **Market Data**
        - LTP
        - Bid/Ask price
        - Bid/Ask quantity
        - ETQ 5m / 20m / 60m
        - Average LTP 20m / 60m
        - Market depth
        """
    )

    st.stop()


# ============================================================
# LIVE MODE
# ============================================================

if mode == "LIVE":

    try:

        angel = get_angel()

        token_list = (
            stocks["token"]
            .astype(str)
            .tolist()
        )

        for start in range(
            0,
            len(token_list),
            1000,
        ):

            batch = token_list[
                start:start + 1000
            ]

            try:
                angel.subscribe(batch)
            except Exception:
                pass

        st.success(
            "🟢 Angel One WebSocket connected."
        )

        time.sleep(3)

        if hasattr(angel, "data"):

            st.session_state.live_data = (
                angel.data.copy()
            )

    except Exception as error:

        st.error(
            f"Angel One connection error: {error}"
        )

        st.stop()


# ============================================================
# BUILD TABLE
# ============================================================

rows = []


# ============================================================
# LIVE TABLE
# ============================================================

if mode == "LIVE":

    live_data = st.session_state.live_data

    for _, stock in stocks.iterrows():

        symbol = stock["SYMBOL"]
        token = str(stock["token"])

        quote = live_data.get(token)

        if not quote:
            continue

        try:

            ltp = float(
                quote.get(
                    "last_traded_price",
                    0,
                )
            ) / 100

            if not (
                price_min
                <= ltp
                <= price_max
            ):
                continue

            bid_qty = float(
                quote.get(
                    "total_buy_quantity",
                    0,
                )
            )

            ask_qty = float(
                quote.get(
                    "total_sell_quantity",
                    0,
                )
            )

            if (
                bid_qty <= minimum_quantity
                or ask_qty <= minimum_quantity
            ):
                continue

            buy_depth = quote.get(
                "best_5_buy_data",
                [],
            )

            sell_depth = quote.get(
                "best_5_sell_data",
                [],
            )

            bid_price = 0.0
            ask_price = 0.0

            if buy_depth:

                bid_price = float(
                    buy_depth[0].get(
                        "price",
                        0,
                    )
                ) / 100

            if sell_depth:

                ask_price = float(
                    sell_depth[0].get(
                        "price",
                        0,
                    )
                ) / 100

            history = get_history(
                symbol,
                token,
            )

            analysis = analyze_history(
                history
            )

            if not history.empty:

                market = replay_data(
                    history
                )

                etq5 = market["ETQ 5m"]
                etq20 = market["ETQ 20m"]
                etq60 = market["ETQ 60m"]
                avg20 = market["Avg LTP 20m"]
                avg60 = market["Avg LTP 60m"]

            else:

                etq5 = np.nan
                etq20 = np.nan
                etq60 = np.nan
                avg20 = np.nan
                avg60 = np.nan

            rows.append(
                {
                    "Stock": symbol,
                    "LTP": round(ltp, 2),
                    "Bid Price": round(
                        bid_price,
                        2,
                    ),
                    "Bid Quantity": int(
                        bid_qty
                    ),
                    "Ask Price": round(
                        ask_price,
                        2,
                    ),
                    "Ask Quantity": int(
                        ask_qty
                    ),
                    "ETQ 5m": etq5,
                    "ETQ 20m": etq20,
                    "ETQ 60m": etq60,
                    "Avg LTP 20m": (
                        round(avg20, 2)
                        if pd.notna(avg20)
                        else np.nan
                    ),
                    "Avg LTP 60m": (
                        round(avg60, 2)
                        if pd.notna(avg60)
                        else np.nan
                    ),
                    "SMMA 20": (
                        round(
                            analysis["SMMA 20"],
                            2,
                        )
                        if pd.notna(
                            analysis["SMMA 20"]
                        )
                        else np.nan
                    ),
                    "SMMA 120": (
                        round(
                            analysis["SMMA 120"],
                            2,
                        )
                        if pd.notna(
                            analysis["SMMA 120"]
                        )
                        else np.nan
                    ),
                    "Signal": analysis[
                        "Signal"
                    ],
                    "ML Probability": (
                        f'{analysis["ML Probability"]:.1%}'
                        if pd.notna(
                            analysis[
                                "ML Probability"
                            ]
                        )
                        else "N/A"
                    ),
                    "Historical Success": (
                        f'{analysis["Historical Success"]:.1%}'
                        if pd.notna(
                            analysis[
                                "Historical Success"
                            ]
                        )
                        else "N/A"
                    ),
                    "Decision": analysis[
                        "Decision"
                    ],
                    "Reason": analysis[
                        "Reason"
                    ],
                    "Data Source": "ANGEL ONE LIVE",
                }
            )

        except Exception:
            continue


# ============================================================
# REPLAY TABLE
# ============================================================

else:

    replay_stocks = stocks.head(30)

    progress = st.progress(0)

    total = len(replay_stocks)

    for index, (_, stock) in enumerate(
        replay_stocks.iterrows()
    ):

        symbol = stock["SYMBOL"]
        token = str(stock["token"])

        try:

            history = get_history(
                symbol,
                token,
                days=5,
            )

            if history.empty:
                progress.progress(
                    (index + 1) / total
                )
                continue

            market = replay_data(
                history
            )

            if market is None:
                progress.progress(
                    (index + 1) / total
                )
                continue

            ltp = market["LTP"]

            if not (
                price_min
                <= ltp
                <= price_max
            ):

                progress.progress(
                    (index + 1) / total
                )
                continue

            if (
                market["Bid Quantity"]
                <= minimum_quantity
                or
                market["Ask Quantity"]
                <= minimum_quantity
            ):

                progress.progress(
                    (index + 1) / total
                )
                continue

            analysis = analyze_history(
                history
            )

            rows.append(
                {
                    "Stock": symbol,
                    "LTP": round(
                        market["LTP"],
                        2,
                    ),
                    "Bid Price": market[
                        "Bid Price"
                    ],
                    "Bid Quantity": market[
                        "Bid Quantity"
                    ],
                    "Ask Price": market[
                        "Ask Price"
                    ],
                    "Ask Quantity": market[
                        "Ask Quantity"
                    ],
                    "ETQ 5m": market[
                        "ETQ 5m"
                    ],
                    "ETQ 20m": market[
                        "ETQ 20m"
                    ],
                    "ETQ 60m": market[
                        "ETQ 60m"
                    ],
                    "Avg LTP 20m": round(
                        market[
                            "Avg LTP 20m"
                        ],
                        2,
                    ),
                    "Avg LTP 60m": round(
                        market[
                            "Avg LTP 60m"
                        ],
                        2,
                    ),
                    "SMMA 20": (
                        round(
                            analysis[
                                "SMMA 20"
                            ],
                            2,
                        )
                        if pd.notna(
                            analysis[
                                "SMMA 20"
                            ]
                        )
                        else np.nan
                    ),
                    "SMMA 120": (
                        round(
                            analysis[
                                "SMMA 120"
                            ],
                            2,
                        )
                        if pd.notna(
                            analysis[
                                "SMMA 120"
                            ]
                        )
                        else np.nan
                    ),
                    "Signal": analysis[
                        "Signal"
                    ],
                    "ML Probability": (
                        f'{analysis["ML Probability"]:.1%}'
                        if pd.notna(
                            analysis[
                                "ML Probability"
                            ]
                        )
                        else "N/A"
                    ),
                    "Historical Success": (
                        f'{analysis["Historical Success"]:.1%}'
                        if pd.notna(
                            analysis[
                                "Historical Success"
                            ]
                        )
                        else "N/A"
                    ),
                    "Decision": analysis[
                        "Decision"
                    ],
                    "Reason": analysis[
                        "Reason"
                    ],
                    "Data Source": "ANGEL ONE REPLAY",
                }
            )

        except Exception:
            pass

        progress.progress(
            (index + 1) / total
        )

    progress.empty()


# ============================================================
# DISPLAY RESULTS
# ============================================================

result = pd.DataFrame(rows)

st.subheader(
    f"📊 {mode} NSE SCREEN — "
    f"{datetime.now().strftime('%H:%M:%S')}"
)

if result.empty:

    st.warning(
        "No stocks currently satisfy the "
        "₹30–₹500 + liquidity filters."
    )

else:

    st.metric(
        "Stocks Passing Filters",
        len(result),
    )

    st.dataframe(
        result,
        use_container_width=True,
        height=650,
    )

    st.subheader(
        "🤖 AI/ML Signal Analysis"
    )

    accepted = result[
        result["Decision"] == "ACCEPT"
    ]

    avoided = result[
        result["Decision"] == "AVOID"
    ]

    waiting = result[
        result["Decision"] == "WAIT"
    ]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "ACCEPT",
        len(accepted),
    )

    c2.metric(
        "AVOID",
        len(avoided),
    )

    c3.metric(
        "WAIT",
        len(waiting),
    )

    if not accepted.empty:

        st.success(
            "✅ Accepted crossover candidates"
        )

        st.dataframe(
            accepted[
                [
                    "Stock",
                    "LTP",
                    "SMMA 20",
                    "SMMA 120",
                    "Signal",
                    "ML Probability",
                    "Historical Success",
                    "Decision",
                    "Reason",
                ]
            ],
            use_container_width=True,
        )

    if not avoided.empty:

        st.warning(
            "⚠️ Crossovers recommended for avoidance"
        )

        st.dataframe(
            avoided[
                [
                    "Stock",
                    "LTP",
                    "SMMA 20",
                    "SMMA 120",
                    "Signal",
                    "ML Probability",
                    "Historical Success",
                    "Decision",
                    "Reason",
                ]
            ],
            use_container_width=True,
        )

    if mode == "REPLAY":

        st.caption(
            "REPLAY uses historical Angel One candle data. "
            "Replay Bid/Ask quantities are simulated for "
            "demonstration and are not live order-book values."
        )


# ============================================================
# AUTO REFRESH
# ============================================================

time.sleep(refresh_seconds)
st.rerun()
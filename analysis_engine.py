import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# SMMA
# ============================================================

def smma(series, period):
    return series.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    df["SMMA_20"] = smma(
        df["Close"],
        20
    )

    df["SMMA_120"] = smma(
        df["Close"],
        120
    )

    df["Return_1"] = (
        df["Close"]
        .pct_change(1)
    )

    df["Return_5"] = (
        df["Close"]
        .pct_change(5)
    )

    df["Volatility"] = (
        df["Return_1"]
        .rolling(20)
        .std()
    )

    df["Volume_Change"] = (
        df["Volume"]
        .pct_change()
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    df["SMMA_Gap"] = (
        df["SMMA_20"]
        - df["SMMA_120"]
    )

    df["SMMA_Gap_Pct"] = (
        df["SMMA_Gap"]
        / df["SMMA_120"]
    )

    return df


# ============================================================
# DETECT EVERY CROSSOVER
# ============================================================

def detect_crossovers(df):

    df = add_indicators(df)

    df["Signal"] = "NONE"

    bullish = (
        (df["SMMA_20"] > df["SMMA_120"])
        &
        (
            df["SMMA_20"].shift(1)
            <=
            df["SMMA_120"].shift(1)
        )
    )

    bearish = (
        (df["SMMA_20"] < df["SMMA_120"])
        &
        (
            df["SMMA_20"].shift(1)
            >=
            df["SMMA_120"].shift(1)
        )
    )

    df.loc[
        bullish,
        "Signal"
    ] = "BUY"

    df.loc[
        bearish,
        "Signal"
    ] = "SELL"

    return df


# ============================================================
# PROFITABILITY OF EACH CROSSOVER
# ============================================================

def evaluate_crossovers(
    df,
    horizon=10
):

    df = detect_crossovers(df)

    records = []

    for i in range(len(df) - horizon):

        signal = df.iloc[i]["Signal"]

        if signal not in [
            "BUY",
            "SELL"
        ]:
            continue

        entry = float(
            df.iloc[i]["Close"]
        )

        future = float(
            df.iloc[
                i + horizon
            ]["Close"]
        )

        if signal == "BUY":

            profitable = (
                future > entry
            )

        else:

            profitable = (
                future < entry
            )

        records.append(
            {
                "Index": i,
                "Datetime":
                    df.iloc[i]["Datetime"]
                    if "Datetime" in df.columns
                    else i,
                "Signal": signal,
                "Entry": entry,
                "Exit": future,
                "Profit": (
                    future - entry
                    if signal == "BUY"
                    else entry - future
                ),
                "Profitable":
                    int(profitable),
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# MACHINE LEARNING DATASET
# ============================================================

FEATURES = [
    "Return_1",
    "Return_5",
    "Volatility",
    "Volume_Change",
    "SMMA_Gap_Pct",
]


def build_ml_dataset(df):

    data = add_indicators(df)

    horizon = 10

    data["Future_Return"] = (
        data["Close"]
        .shift(-horizon)
        /
        data["Close"]
        - 1
    )

    data["Target"] = (
        data["Future_Return"] > 0
    ).astype(int)

    data = data.dropna(
        subset=FEATURES + ["Target"]
    )

    return data


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

def train_model(df):

    data = build_ml_dataset(df)

    if len(data) < 100:

        return None

    X = data[FEATURES]
    y = data["Target"]

    if y.nunique() < 2:

        return None

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=4,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(
        X,
        y
    )

    return model, FEATURES


# ============================================================
# ANALYSE CURRENT SIGNAL
# ============================================================

def analyze_signal(df):

    data = detect_crossovers(df)

    if len(data) < 130:

        return {
            "SMMA 20": np.nan,
            "SMMA 120": np.nan,
            "Signal": "NO DATA",
            "ML Probability": np.nan,
            "Historical Success": np.nan,
            "Decision": "WAIT",
            "Reason":
                "Insufficient historical data.",
        }

    latest = data.iloc[-1]
    previous = data.iloc[-2]

    smma20 = float(
        latest["SMMA_20"]
    )

    smma120 = float(
        latest["SMMA_120"]
    )

    # --------------------------------------------------------
    # CURRENT CROSSOVER
    # --------------------------------------------------------

    signal = "NONE"

    if (
        previous["SMMA_20"]
        <=
        previous["SMMA_120"]
        and
        latest["SMMA_20"]
        >
        latest["SMMA_120"]
    ):

        signal = "BUY"

    elif (
        previous["SMMA_20"]
        >=
        previous["SMMA_120"]
        and
        latest["SMMA_20"]
        <
        latest["SMMA_120"]
    ):

        signal = "SELL"

    # --------------------------------------------------------
    # HISTORICAL CROSSOVERS
    # --------------------------------------------------------

    historical = evaluate_crossovers(
        df,
        horizon=10
    )

    historical_success = np.nan

    if not historical.empty:

        if signal == "BUY":

            subset = historical[
                historical["Signal"]
                == "BUY"
            ]

        elif signal == "SELL":

            subset = historical[
                historical["Signal"]
                == "SELL"
            ]

        else:

            subset = historical

        if not subset.empty:

            historical_success = float(
                subset["Profitable"].mean()
            )

    # --------------------------------------------------------
    # MACHINE LEARNING
    # --------------------------------------------------------

    probability = np.nan

    model_result = train_model(
        df
    )

    if model_result is not None:

        model, features = model_result

        latest_features = (
            data.iloc[-1:][features]
        )

        if not latest_features.isna().any(
            axis=None
        ):

            probability = float(
                model.predict_proba(
                    latest_features
                )[0][1]
            )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    if signal == "NONE":

        decision = "WAIT"

        reason = (
            "No new SMMA 20/120 crossover "
            "on the latest candle."
        )

    else:

        # For SELL, probability of rising price
        # means probability against the trade.
        if signal == "SELL" and not np.isnan(
            probability
        ):

            trade_probability = (
                1 - probability
            )

        else:

            trade_probability = probability

        reasons = []

        if not np.isnan(
            historical_success
        ):

            if historical_success < 0.50:

                reasons.append(
                    "Historical crossover "
                    "success is below 50%."
                )

            else:

                reasons.append(
                    f"Historical success "
                    f"{historical_success:.1%}."
                )

        if not np.isnan(
            trade_probability
        ):

            if trade_probability < 0.55:

                reasons.append(
                    "ML probability is weak."
                )

            else:

                reasons.append(
                    f"ML probability "
                    f"{trade_probability:.1%}."
                )

        if (
            not np.isnan(
                historical_success
            )
            and
            not np.isnan(
                trade_probability
            )
            and
            historical_success >= 0.55
            and
            trade_probability >= 0.60
        ):

            decision = "ACCEPT"

            reasons.append(
                "Both historical and ML "
                "evidence support the signal."
            )

        else:

            decision = "AVOID"

            reasons.append(
                "Risk/reward evidence is "
                "not strong enough."
            )

        reason = " ".join(
            reasons
        )

    return {
        "SMMA 20": smma20,
        "SMMA 120": smma120,
        "Signal": signal,
        "ML Probability": probability,
        "Historical Success":
            historical_success,
        "Decision": decision,
        "Reason": reason,
    }


# ============================================================
# COMPATIBILITY FUNCTION USED BY app.py
# ============================================================

def analyze_history(df):

    return analyze_signal(df)
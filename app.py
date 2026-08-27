
import math
import random
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# ============================================================
# 易經 × 台股量化研究 V3
# ------------------------------------------------------------
# V3 重點：
# 1. 64 卦標準上下卦矩陣
# 2. 時間起卦 / 數字起卦 / 三枚銅錢
# 3. 技術指標 + 大盤環境
# 4. 歷史逐日時間卦
# 5. Walk-forward / out-of-sample 思路
# 6. 訓練期估計「卦 + 動爻 + 技術條件」的歷史條件統計
# 7. 測試期完全使用訓練期統計，避免把未來資料混入模型
# 8. 手續費 / 滑價參數
# 9. Benchmark 比較
#
# 注意：
# 這不是投資建議。易經因子屬文化／研究變數，
# 不代表已證明具備金融市場預測能力。
# ============================================================


st.set_page_config(
    page_title="易經 × 台股量化研究 V3",
    page_icon="☯️",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1050px;
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .signal {
        border: 1px solid rgba(128,128,128,.35);
        border-radius: 16px;
        padding: 18px;
        text-align: center;
        margin: 8px 0 16px 0;
    }
    .signal-title {
        font-size: 30px;
        font-weight: 700;
    }
    .muted {
        color: #777;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 64 卦資料
# ============================================================

TRIGRAM_BITS = {
    "乾": "111",
    "兌": "110",
    "離": "101",
    "震": "100",
    "巽": "011",
    "坎": "010",
    "艮": "001",
    "坤": "000",
}

TRIGRAM_ELEMENT = {
    "乾": "天",
    "兌": "澤",
    "離": "火",
    "震": "雷",
    "巽": "風",
    "坎": "水",
    "艮": "山",
    "坤": "地",
}

TRIGRAM_ORDER = ["乾", "兌", "離", "震", "巽", "坎", "艮", "坤"]

KING_WEN_NAMES = [
    "乾為天", "坤為地", "水雷屯", "山水蒙", "水天需", "天水訟",
    "地水師", "水地比", "風天小畜", "天澤履", "地天泰", "天地否",
    "天火同人", "火天大有", "地山謙", "雷地豫", "澤雷隨", "山風蠱",
    "地澤臨", "風地觀", "火雷噬嗑", "山火賁", "山地剝", "地雷復",
    "天雷无妄", "山天大畜", "山雷頤", "澤風大過", "坎為水", "離為火",
    "澤山咸", "雷風恆", "天山遯", "雷天大壯", "火地晉", "地火明夷",
    "風火家人", "火澤睽", "水山蹇", "雷水解", "山澤損", "風雷益",
    "澤天夬", "天風姤", "澤地萃", "地風升", "澤水困", "水風井",
    "澤火革", "火風鼎", "震為雷", "艮為山", "風山漸", "雷澤歸妹",
    "雷火豐", "火山旅", "巽為風", "兌為澤", "風水渙", "水澤節",
    "風澤中孚", "雷山小過", "水火既濟", "火水未濟",
]

# row = lower trigram, col = upper trigram
KING_WEN_MATRIX = [
    [1, 43, 14, 34, 9, 5, 26, 11],
    [10, 58, 38, 54, 61, 60, 41, 19],
    [13, 49, 30, 55, 37, 63, 22, 36],
    [25, 17, 21, 51, 42, 3, 27, 24],
    [44, 28, 50, 32, 57, 48, 18, 46],
    [6, 47, 64, 40, 59, 29, 4, 7],
    [33, 31, 56, 62, 53, 39, 52, 15],
    [12, 45, 35, 16, 20, 8, 23, 2],
]

KW_BY_BITS = {}
HEXAGRAM_BY_NO = {}

for li, lower in enumerate(TRIGRAM_ORDER):
    for ui, upper in enumerate(TRIGRAM_ORDER):
        number = KING_WEN_MATRIX[li][ui]
        bits = TRIGRAM_BITS[lower] + TRIGRAM_BITS[upper]
        KW_BY_BITS[bits] = number
        HEXAGRAM_BY_NO[number] = {
            "name": KING_WEN_NAMES[number - 1],
            "lower": lower,
            "upper": upper,
            "bits": bits,
        }


# ============================================================
# Utility
# ============================================================

def clamp(x, lo=-1.0, hi=1.0):
    return float(max(lo, min(hi, x)))


def ticker_symbol(raw):
    raw = str(raw).strip().upper()
    if raw.endswith(".TW") or raw.endswith(".TWO") or raw.startswith("^"):
        return raw
    if raw.isdigit():
        return raw + ".TW"
    return raw


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def bits_to_number(bits):
    return KW_BY_BITS.get("".join(str(int(x)) for x in bits))


def bits_to_name(bits):
    no = bits_to_number(bits)
    return HEXAGRAM_BY_NO[no]["name"] if no else "未知卦"


def line_display(bits, moving_lines=None):
    moving_lines = set(moving_lines or [])
    rows = []
    for line_no in range(6, 0, -1):
        glyph = "━━━━━━" if bits[line_no - 1] else "━━  ━━"
        mark = "  ← 動爻" if line_no in moving_lines else ""
        rows.append(f"第{line_no}爻  {glyph}{mark}")
    return "\n".join(rows)


def transform_bits(bits, moving_lines):
    out = list(bits)
    for line_no in moving_lines:
        out[line_no - 1] = 1 - out[line_no - 1]
    return out


def mutual_bits(bits):
    # 下互卦：2、3、4爻；上互卦：3、4、5爻
    return [bits[1], bits[2], bits[3], bits[2], bits[3], bits[4]]


def opposite_bits(bits):
    return [1 - x for x in bits]


def inverse_bits(bits):
    return list(reversed(bits))


# ============================================================
# 起卦
# ============================================================

def time_cast(dt):
    """
    一種可重現的時間起卦實作：
    上卦 = 年 + 月 + 日
    下卦 = 年 + 月 + 日 + 時
    動爻 = 年 + 月 + 日 + 時
    """
    y, m, d, h = dt.year, dt.month, dt.day, dt.hour
    upper_idx = (y + m + d) % 8
    lower_idx = (y + m + d + h) % 8
    moving = (y + m + d + h) % 6 + 1

    upper = TRIGRAM_ORDER[upper_idx]
    lower = TRIGRAM_ORDER[lower_idx]
    bits = list(map(int, TRIGRAM_BITS[lower] + TRIGRAM_BITS[upper]))

    return bits, [moving], {"upper": upper, "lower": lower}


def number_cast(a, b):
    upper = TRIGRAM_ORDER[int(a) % 8]
    lower = TRIGRAM_ORDER[int(b) % 8]
    moving = (int(a) + int(b)) % 6 + 1
    bits = list(map(int, TRIGRAM_BITS[lower] + TRIGRAM_BITS[upper]))

    return bits, [moving], {"upper": upper, "lower": lower}


def coin_cast(seed=None):
    rng = random.Random(seed)
    lines = []
    moving = []
    values = []

    for line_no in range(1, 7):
        # 三枚銅錢：正面=3、反面=2
        total = sum(rng.choice([2, 3]) for _ in range(3))
        values.append(total)

        if total in (7, 9):
            lines.append(1)
        else:
            lines.append(0)

        if total in (6, 9):
            moving.append(line_no)

    return lines, moving, {"coin_values": values}


def historical_time_cast(ts, hour=13):
    dt = datetime(
        int(ts.year),
        int(ts.month),
        int(ts.day),
        int(hour),
        0,
    )
    return time_cast(dt)


# ============================================================
# 行情
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def download_history(symbol, period="5y"):
    df = yf.download(
        symbol,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in required):
        return pd.DataFrame()

    df = df[required].copy()
    df = df.dropna()

    idx = pd.to_datetime(df.index)
    try:
        idx = idx.tz_localize(None)
    except TypeError:
        pass

    df.index = idx
    return df


# ============================================================
# 技術指標
# ============================================================

def indicators(df):
    x = df.copy()

    x["ret_1d"] = x["Close"].pct_change()
    x["ret_5d"] = x["Close"].pct_change(5)

    x["MA5"] = x["Close"].rolling(5).mean()
    x["MA20"] = x["Close"].rolling(20).mean()
    x["MA60"] = x["Close"].rolling(60).mean()

    delta = x["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    x["RSI14"] = 100 - (100 / (1 + rs))

    ema12 = x["Close"].ewm(span=12, adjust=False).mean()
    ema26 = x["Close"].ewm(span=26, adjust=False).mean()

    x["MACD"] = ema12 - ema26
    x["MACDSignal"] = x["MACD"].ewm(span=9, adjust=False).mean()
    x["MACDHist"] = x["MACD"] - x["MACDSignal"]

    tr = pd.concat(
        [
            x["High"] - x["Low"],
            (x["High"] - x["Close"].shift()).abs(),
            (x["Low"] - x["Close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    x["ATR14"] = tr.rolling(14).mean()
    x["ATR_pct"] = x["ATR14"] / x["Close"]

    x["VOL20"] = x["Volume"].rolling(20).mean()
    x["VOL_RATIO"] = x["Volume"] / x["VOL20"]

    x["BB_MID"] = x["Close"].rolling(20).mean()
    bb_std = x["Close"].rolling(20).std()
    x["BB_UPPER"] = x["BB_MID"] + 2 * bb_std
    x["BB_LOWER"] = x["BB_MID"] - 2 * bb_std

    return x


def technical_score(row):
    vals = [
        row.get("MA20"),
        row.get("MA60"),
        row.get("RSI14"),
        row.get("MACDHist"),
        row.get("VOL_RATIO"),
        row.get("BB_MID"),
        row.get("Close"),
    ]

    if any(pd.isna(v) for v in vals):
        return 0.0

    score = 0.0

    score += 0.22 if row["Close"] > row["MA20"] else -0.22
    score += 0.22 if row["MA20"] > row["MA60"] else -0.22

    if row["RSI14"] >= 55:
        score += 0.18
    elif row["RSI14"] <= 45:
        score -= 0.18

    score += 0.18 if row["MACDHist"] > 0 else -0.18

    if row["VOL_RATIO"] >= 1.1:
        score += 0.10
    elif row["VOL_RATIO"] <= 0.8:
        score -= 0.05

    score += 0.10 if row["Close"] > row["BB_MID"] else -0.10

    return clamp(score)


def market_score(row):
    vals = [row.get("Close"), row.get("MA20"), row.get("MA60")]
    if any(pd.isna(v) for v in vals):
        return 0.0

    score = 0.0
    score += 0.5 if row["Close"] > row["MA20"] else -0.5
    score += 0.5 if row["MA20"] > row["MA60"] else -0.5
    return clamp(score)


# ============================================================
# Feature engineering
# ============================================================

def make_daily_dataset(stock_df, market_df, cast_hour=13):
    s = indicators(stock_df)
    m = indicators(market_df)

    mkt_cols = [
        "Close",
        "MA20",
        "MA60",
        "RSI14",
        "MACDHist",
    ]

    joined = s.join(
        m[mkt_cols].add_prefix("MKT_"),
        how="inner",
    )

    rows = []

    for dt, row in joined.iterrows():
        if pd.isna(row["MA60"]) or pd.isna(row["MKT_MA60"]):
            continue

        bits, moving, _ = historical_time_cast(
            dt,
            hour=cast_hour,
        )

        hex_no = bits_to_number(bits)
        if hex_no is None:
            continue

        future_1 = s["Close"].shift(-1).reindex([dt]).iloc[0]
        future_5 = s["Close"].shift(-5).reindex([dt]).iloc[0]

        if pd.isna(future_1) or pd.isna(future_5):
            continue

        tech = technical_score(row)

        mkt_row = pd.Series(
            {
                "Close": row["MKT_Close"],
                "MA20": row["MKT_MA20"],
                "MA60": row["MKT_MA60"],
            }
        )

        mkt = market_score(mkt_row)

        rows.append(
            {
                "Date": dt,
                "HexNo": int(hex_no),
                "Hexagram": HEXAGRAM_BY_NO[hex_no]["name"],
                "MovingLine": int(moving[0]) if moving else 0,
                "TechScore": tech,
                "MarketScore": mkt,
                "Close": row["Close"],
                "RSI14": row["RSI14"],
                "VOL_RATIO": row["VOL_RATIO"],
                "ATR_pct": row["ATR_pct"],
                "Ret1": row["ret_1d"],
                "Ret5": row["ret_5d"],
                "Future1": future_1 / row["Close"] - 1,
                "Future5": future_5 / row["Close"] - 1,
            }
        )

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    # 基礎易經先驗：只是一個研究 baseline。
    out["IChingPrior"] = out["HexNo"].map(
        {
            1: 0.30, 2: 0.00, 3: -0.10, 4: 0.00,
            5: 0.05, 6: -0.10, 7: 0.00, 8: 0.05,
            9: 0.10, 10: 0.05, 11: 0.20, 12: -0.20,
            13: 0.15, 14: 0.25, 15: 0.05, 16: 0.10,
            17: 0.10, 18: -0.05, 19: 0.15, 20: 0.05,
            21: 0.05, 22: 0.05, 23: -0.20, 24: 0.15,
            25: 0.05, 26: 0.15, 27: 0.05, 28: -0.05,
            29: -0.15, 30: 0.15, 31: 0.10, 32: 0.10,
            33: -0.10, 34: 0.15, 35: 0.15, 36: -0.15,
            37: 0.05, 38: -0.05, 39: -0.15, 40: 0.10,
            41: -0.05, 42: 0.15, 43: 0.05, 44: -0.05,
            45: 0.10, 46: 0.15, 47: -0.15, 48: 0.05,
            49: 0.10, 50: 0.15, 51: 0.05, 52: 0.00,
            53: 0.10, 54: -0.05, 55: 0.15, 56: -0.05,
            57: 0.05, 58: 0.05, 59: 0.00, 60: 0.00,
            61: 0.10, 62: -0.05, 63: 0.10, 64: -0.05,
        }
    ).fillna(0.0)

    # moving line 的小幅研究修正
    out["LinePrior"] = (3.5 - out["MovingLine"]) / 20.0
    out["IChingBaseline"] = (
        0.85 * out["IChingPrior"] +
        0.15 * out["LinePrior"]
    ).clip(-1, 1)

    return out.reset_index(drop=True)


# ============================================================
# Walk-forward model
# ------------------------------------------------------------
# 核心：
# 每個 test 日只使用它之前的 train 資料。
#
# 分組鍵：
# HexNo + MovingLine + 技術 regime + 大盤 regime
# ============================================================

def add_regimes(df):
    x = df.copy()

    x["TechRegime"] = np.select(
        [
            x["TechScore"] >= 0.35,
            x["TechScore"] <= -0.35,
        ],
        [
            "bull",
            "bear",
        ],
        default="neutral",
    )

    x["MarketRegime"] = np.select(
        [
            x["MarketScore"] >= 0.5,
            x["MarketScore"] <= -0.5,
        ],
        [
            "bull",
            "bear",
        ],
        default="neutral",
    )

    x["Key"] = (
        x["HexNo"].astype(str)
        + "_L"
        + x["MovingLine"].astype(str)
        + "_T"
        + x["TechRegime"]
        + "_M"
        + x["MarketRegime"]
    )

    return x


def fit_history_stats(train):
    """
    使用 train 期估計每個 regime/key 的方向與報酬。
    smoothing 使用 Beta(1,1)，避免小樣本 100% / 0% 過度自信。
    """
    if train.empty:
        return {}, {}

    work = add_regimes(train)

    global_up = (work["Future1"] > 0).mean()
    global_avg = work["Future1"].mean()

    global_stats = {
        "up": float(global_up),
        "avg": float(global_avg),
        "n": int(len(work)),
    }

    grouped = (
        work.groupby("Key")
        .agg(
            n=("Future1", "size"),
            up=("Future1", lambda s: (s > 0).mean()),
            avg=("Future1", "mean"),
            avg5=("Future5", "mean"),
            median=("Future1", "median"),
        )
        .to_dict("index")
    )

    return grouped, global_stats


def score_one_from_history(row, stats, global_stats, min_group_samples=15):
    """
    用歷史條件統計轉成 [-1,1]：
    2 * P(up) - 1，再與 I Ching baseline / technical / market 做小幅融合。
    """
    temp = pd.DataFrame([row])
    temp = add_regimes(temp).iloc[0]

    key = temp["Key"]
    s = stats.get(key)

    if s is not None and s["n"] >= min_group_samples:
        p_up = (s["up"] * s["n"] + global_stats["up"] * 5) / (
            s["n"] + 5
        )
        expected = s["avg"]
        n = s["n"]
    else:
        p_up = global_stats["up"]
        expected = global_stats["avg"]
        n = 0

    empirical = clamp(2 * p_up - 1)

    # V3：讓資料驅動的 empirical factor 為主，
    # 易經 baseline 僅保留為小權重，避免主觀先驗支配結果。
    model_score = clamp(
        0.65 * empirical
        + 0.15 * row["IChingBaseline"]
        + 0.15 * row["TechScore"]
        + 0.05 * row["MarketScore"]
    )

    return {
        "score": model_score,
        "p_up": p_up,
        "expected_1d": expected,
        "history_n": n,
        "key": key,
    }


def walk_forward(
    df,
    train_days=504,
    min_group_samples=15,
    step=1,
):
    if len(df) <= train_days + 10:
        return pd.DataFrame()

    results = []

    for i in range(train_days, len(df), step):
        train = df.iloc[:i].copy()
        test = df.iloc[i].copy()

        stats, global_stats = fit_history_stats(train)

        pred = score_one_from_history(
            test,
            stats,
            global_stats,
            min_group_samples=min_group_samples,
        )

        results.append(
            {
                "Date": test["Date"],
                "HexNo": test["HexNo"],
                "Hexagram": test["Hexagram"],
                "MovingLine": test["MovingLine"],
                "TechScore": test["TechScore"],
                "MarketScore": test["MarketScore"],
                "IChingBaseline": test["IChingBaseline"],
                "Score": pred["score"],
                "PUp": pred["p_up"],
                "Expected1D": pred["expected_1d"],
                "HistoryN": pred["history_n"],
                "Actual1D": test["Future1"],
                "Actual5D": test["Future5"],
            }
        )

    return pd.DataFrame(results)


# ============================================================
# 回測統計
# ============================================================

def add_backtest_returns(result, min_score=0.10, cost_bps=10.0):
    if result.empty:
        return result

    x = result.copy()

    x["Direction"] = np.where(
        x["Score"] > min_score,
        1,
        np.where(x["Score"] < -min_score, -1, 0),
    )

    # 每次有方向訊號，假設進出一次。
    cost = cost_bps / 10000.0

    x["Strategy1D"] = np.where(
        x["Direction"] == 0,
        0.0,
        x["Direction"] * x["Actual1D"] - cost,
    )

    # 5日策略：用同一方向持有五個交易日。
    x["Strategy5D"] = np.where(
        x["Direction"] == 0,
        0.0,
        x["Direction"] * x["Actual5D"] - cost,
    )

    x["BuyHold1D"] = x["Actual1D"]
    x["BuyHold5D"] = x["Actual5D"]

    x["CumStrategy"] = (1 + x["Strategy1D"]).cumprod() - 1
    x["CumBuyHold"] = (1 + x["BuyHold1D"]).cumprod() - 1

    x["Win1D"] = np.where(
        x["Direction"] == 0,
        np.nan,
        (
            (x["Direction"] > 0) & (x["Actual1D"] > 0)
        ) | (
            (x["Direction"] < 0) & (x["Actual1D"] < 0)
        ),
    )

    x["Win5D"] = np.where(
        x["Direction"] == 0,
        np.nan,
        (
            (x["Direction"] > 0) & (x["Actual5D"] > 0)
        ) | (
            (x["Direction"] < 0) & (x["Actual5D"] < 0)
        ),
    )

    return x


def performance_summary(bt):
    if bt.empty:
        return {}

    active = bt[bt["Direction"] != 0].copy()

    if active.empty:
        return {
            "samples": 0,
            "trades": 0,
            "win1": np.nan,
            "win5": np.nan,
            "avg1": np.nan,
            "avg5": np.nan,
            "strategy_total": 0.0,
            "buyhold_total": 0.0,
            "max_dd": np.nan,
        }

    equity = (1 + active["Strategy1D"]).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1

    return {
        "samples": len(bt),
        "trades": len(active),
        "win1": active["Win1D"].mean(),
        "win5": active["Win5D"].mean(),
        "avg1": active["Strategy1D"].mean(),
        "avg5": active["Strategy5D"].mean(),
        "strategy_total": (1 + active["Strategy1D"]).prod() - 1,
        "buyhold_total": (1 + active["BuyHold1D"]).prod() - 1,
        "max_dd": drawdown.min(),
    }


def hex_stats(bt):
    if bt.empty:
        return pd.DataFrame()

    x = bt[bt["Direction"] != 0].copy()

    if x.empty:
        return pd.DataFrame()

    g = x.groupby(
        ["HexNo", "Hexagram"],
        as_index=False,
    ).agg(
        samples=("Actual1D", "size"),
        p_up=("Actual1D", lambda s: (s > 0).mean()),
        avg1=("Actual1D", "mean"),
        avg5=("Actual5D", "mean"),
        win_model=("Win1D", "mean"),
    )

    return g.sort_values(
        ["samples", "win_model"],
        ascending=[False, False],
    )


def line_stats(bt):
    if bt.empty:
        return pd.DataFrame()

    x = bt[bt["Direction"] != 0].copy()

    if x.empty:
        return pd.DataFrame()

    g = x.groupby(
        "MovingLine",
        as_index=False,
    ).agg(
        samples=("Actual1D", "size"),
        p_up=("Actual1D", lambda s: (s > 0).mean()),
        avg1=("Actual1D", "mean"),
        win_model=("Win1D", "mean"),
    )

    return g.sort_values(
        "win_model",
        ascending=False,
    )


# ============================================================
# 圖表
# ============================================================

def price_chart(df, days=220):
    x = df.tail(days)

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=x.index,
            open=x["Open"],
            high=x["High"],
            low=x["Low"],
            close=x["Close"],
            name="K線",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x.index,
            y=x["MA20"],
            name="MA20",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x.index,
            y=x["MA60"],
            name="MA60",
        )
    )

    fig.update_layout(
        height=480,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
    )

    return fig


def equity_chart(bt):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=bt["Date"],
            y=bt["CumStrategy"] * 100,
            mode="lines",
            name="V3 模型",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=bt["Date"],
            y=bt["CumBuyHold"] * 100,
            mode="lines",
            name="Buy & Hold",
        )
    )

    fig.update_layout(
        height=400,
        yaxis_title="累積報酬 (%)",
        margin=dict(l=10, r=10, t=30, b=10),
    )

    return fig


# ============================================================
# UI：Sidebar
# ============================================================

with st.sidebar:
    st.header("⚙️ V3 分析設定")

    ticker_raw = st.text_input(
        "台股代號",
        value="2330",
    )

    period = st.selectbox(
        "歷史資料",
        ["2y", "5y", "10y", "max"],
        index=1,
    )

    cast_method = st.selectbox(
        "目前卜卦方式",
        ["時間起卦", "數字起卦", "三枚銅錢"],
    )

    if cast_method == "數字起卦":
        num1 = st.number_input(
            "數字一",
            value=2330,
            step=1,
        )
        num2 = st.number_input(
            "數字二",
            value=2026,
            step=1,
        )
    else:
        num1, num2 = 2330, 2026

    if cast_method == "三枚銅錢":
        seed = st.number_input(
            "隨機種子",
            value=0,
            step=1,
        )
    else:
        seed = 0

    st.divider()

    cast_hour = st.slider(
        "歷史時間起卦小時",
        0,
        23,
        13,
    )

    train_days = st.slider(
        "Walk-forward 訓練天數",
        252,
        1000,
        504,
        21,
    )

    min_group_samples = st.slider(
        "條件樣本最低數",
        5,
        50,
        15,
        5,
    )

    signal_threshold = st.slider(
        "訊號門檻",
        0.0,
        0.8,
        0.10,
        0.05,
    )

    cost_bps = st.number_input(
        "單次交易成本（bps）",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        help="研究用簡化假設；台股實際成本應再依券商、稅費與交易方向細化。",
    )

    run = st.button(
        "🔮 開始 V3 分析",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# Main
# ============================================================

st.title("☯️ 易經 × 台股量化研究 V3")
st.caption(
    "資料驅動的 Walk-forward 研究版："
    "易經卦象 + 技術面 + 大盤環境 + 歷史條件統計"
)

if not run:
    st.info(
        "請在左側輸入股票代號後按「開始 V3 分析」。"
        "例如 2330。"
    )

    st.markdown(
        """
        ### V3 與 V2 最大差異

        **V2：**
        > 卦象先驗 + 技術指標 → 人工加權分數

        **V3：**
        > 歷史資料 → 條件統計 → Walk-forward → 樣本外預測

        也就是說，V3 不再直接假設「某卦一定偏多」，
        而是讓模型只使用過去資料來估計：

        `某卦 + 某動爻 + 技術狀態 + 大盤狀態`

        在歷史上出現時，隔日上漲機率與報酬是多少。

        再把這個結果拿到未來尚未看過的資料測試。
        """
    )

    st.warning(
        "本工具僅供研究、學習與娛樂；不構成投資建議。"
    )
    st.stop()


symbol = ticker_symbol(ticker_raw)

with st.spinner("下載行情與建立研究資料集..."):
    stock = download_history(symbol, period)
    market = download_history("^TWII", period)

if stock.empty:
    st.error(
        f"無法取得 {symbol} 行情。"
        "請確認台股代號，例如 2330。"
    )
    st.stop()

if market.empty:
    st.error("無法取得台灣加權指數 ^TWII。")
    st.stop()

stock_i = indicators(stock)
market_i = indicators(market)

latest = stock_i.iloc[-1]
latest_mkt = market_i.iloc[-1]


# ============================================================
# 目前卦象
# ============================================================

if cast_method == "時間起卦":
    bits, moving, cast_meta = time_cast(datetime.now())
elif cast_method == "數字起卦":
    bits, moving, cast_meta = number_cast(
        int(num1),
        int(num2),
    )
else:
    bits, moving, cast_meta = coin_cast(
        seed if seed != 0 else None
    )

base_no = bits_to_number(bits)
base_name = HEXAGRAM_BY_NO[base_no]["name"]

changed_bits = transform_bits(
    bits,
    moving,
)

changed_no = bits_to_number(changed_bits)
changed_name = HEXAGRAM_BY_NO[changed_no]["name"]

mutual_no = bits_to_number(
    mutual_bits(bits)
)
mutual_name = HEXAGRAM_BY_NO[mutual_no]["name"]

opposite_no = bits_to_number(
    opposite_bits(bits)
)
opposite_name = HEXAGRAM_BY_NO[opposite_no]["name"]

inverse_no = bits_to_number(
    inverse_bits(bits)
)
inverse_name = HEXAGRAM_BY_NO[inverse_no]["name"]


# ============================================================
# 建立歷史 dataset
# ============================================================

with st.spinner("建立歷史時間卦與技術 regime..."):
    dataset = make_daily_dataset(
        stock,
        market,
        cast_hour=cast_hour,
    )

if dataset.empty or len(dataset) <= train_days + 10:
    st.error(
        "資料不足以進行 V3 Walk-forward。"
        "請把歷史資料改成 5y / 10y / max，"
        "或降低訓練天數。"
    )
    st.stop()


# ============================================================
# 最新日模型預測
# ------------------------------------------------------------
# 這裡只使用「截至最新資料之前」的 dataset。
# 最新資料本身沒有未來報酬，因此不放進 train。
# ============================================================

latest_row = dataset.iloc[-1].copy()

historical_train = dataset.iloc[:-1].copy()

if len(historical_train) < train_days:
    st.error("最新資料之前的歷史資料不足。")
    st.stop()

train_for_current = historical_train.iloc[-train_days:].copy()

stats, global_stats = fit_history_stats(
    train_for_current
)

current_pred = score_one_from_history(
    latest_row,
    stats,
    global_stats,
    min_group_samples=min_group_samples,
)

current_score = current_pred["score"]

if current_score >= 0.60:
    current_signal = "🚀 強勢偏多"
elif current_score >= signal_threshold:
    current_signal = "↗️ 偏多"
elif current_score <= -0.60:
    current_signal = "🔻 強勢偏空"
elif current_score <= -signal_threshold:
    current_signal = "↘️ 偏空"
else:
    current_signal = "➡️ 觀望 / 震盪"


# ============================================================
# Walk-forward
# ============================================================

with st.spinner("執行 Walk-forward 樣本外回測..."):
    wf = walk_forward(
        dataset,
        train_days=train_days,
        min_group_samples=min_group_samples,
    )

if wf.empty:
    st.error("Walk-forward 沒有產生有效測試樣本。")
    st.stop()

wf = add_backtest_returns(
    wf,
    min_score=signal_threshold,
    cost_bps=cost_bps,
)

summary = performance_summary(wf)


# ============================================================
# Header
# ============================================================

st.success(
    f"{symbol}｜行情截至 {stock.index[-1].date()}｜"
    f"訓練窗 {train_days} 交易日"
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "最新價",
        f"{latest['Close']:.2f}",
    )

with c2:
    st.metric(
        "V3 Score",
        f"{current_score:+.2f}",
    )

with c3:
    st.metric(
        "歷史條件 P(↑)",
        f"{current_pred['p_up'] * 100:.1f}%",
    )

with c4:
    st.metric(
        "條件樣本",
        f"{current_pred['history_n']}",
    )

st.markdown(
    f"""
    <div class="signal">
        <div class="muted">V3 模型方向</div>
        <div class="signal-title">{current_signal}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "V3 的方向不是「卦象保證漲跌」，"
    "而是把歷史條件統計轉換成研究型機率分數。"
)


# ============================================================
# Tabs
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "☯️ 目前卦象",
        "📈 技術面",
        "🧪 V3 Walk-forward",
        "🔬 卦象統計",
        "📋 原始資料",
    ]
)


# ============================================================
# Tab 1
# ============================================================

with tab1:
    st.subheader(
        f"本卦：{base_name}"
    )

    st.code(
        line_display(
            bits,
            moving,
        )
    )

    a, b, c, d = st.columns(4)

    with a:
        st.metric(
            "本卦",
            base_name,
        )

    with b:
        st.metric(
            "動爻",
            "、".join(
                str(x)
                for x in moving
            ) if moving else "無",
        )

    with c:
        st.metric(
            "變卦",
            changed_name,
        )

    with d:
        st.metric(
            "互卦",
            mutual_name,
        )

    st.write(
        f"錯卦：**{opposite_name}**"
    )

    st.write(
        f"綜卦：**{inverse_name}**"
    )

    if cast_method == "三枚銅錢":
        st.write(
            f"銅錢結果：`{cast_meta['coin_values']}`"
        )

    st.divider()

    st.subheader("V3 最新預測分解")

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.metric(
            "歷史條件 P(↑)",
            f"{current_pred['p_up'] * 100:.1f}%",
        )

    with p2:
        st.metric(
            "歷史平均隔日",
            f"{current_pred['expected_1d'] * 100:+.2f}%",
        )

    with p3:
        st.metric(
            "易經 baseline",
            f"{latest_row['IChingBaseline']:+.2f}",
        )

    with p4:
        st.metric(
            "技術分數",
            f"{latest_row['TechScore']:+.2f}",
        )

    st.info(
        "如果條件樣本數很少，V3 會回退到較寬的歷史統計，"
        "並使用平滑，避免單一小樣本造成 100% / 0% 的假象。"
    )


# ============================================================
# Tab 2
# ============================================================

with tab2:
    st.subheader("價格與均線")

    st.plotly_chart(
        price_chart(stock_i),
        use_container_width=True,
    )

    metric_df = pd.DataFrame(
        {
            "指標": [
                "MA5",
                "MA20",
                "MA60",
                "RSI14",
                "MACD",
                "MACD Signal",
                "ATR14",
                "ATR%",
                "成交量比",
            ],
            "數值": [
                latest["MA5"],
                latest["MA20"],
                latest["MA60"],
                latest["RSI14"],
                latest["MACD"],
                latest["MACDSignal"],
                latest["ATR14"],
                latest["ATR_pct"] * 100,
                latest["VOL_RATIO"],
            ],
        }
    )

    st.dataframe(
        metric_df,
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("大盤環境")

    st.write(
        f"加權指數：**{latest_mkt['Close']:.2f}**｜"
        f"MA20：**{latest_mkt['MA20']:.2f}**｜"
        f"MA60：**{latest_mkt['MA60']:.2f}**"
    )

    st.write(
        f"大盤分數：**{market_score(latest_mkt):+.2f}**"
    )


# ============================================================
# Tab 3
# ============================================================

with tab3:
    st.subheader(
        "Walk-forward 樣本外回測"
    )

    st.caption(
        "每個測試日期只使用該日期以前的訓練資料；"
        "模型不讀取測試日期之後的條件統計。"
    )

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.metric(
            "測試樣本",
            summary["samples"],
        )

    with s2:
        st.metric(
            "實際交易數",
            summary["trades"],
        )

    with s3:
        value = summary["win1"]
        st.metric(
            "隔日命中",
            "N/A" if pd.isna(value)
            else f"{value * 100:.1f}%",
        )

    with s4:
        value = summary["win5"]
        st.metric(
            "5日命中",
            "N/A" if pd.isna(value)
            else f"{value * 100:.1f}%",
        )

    p1, p2, p3 = st.columns(3)

    with p1:
        st.metric(
            "模型累積報酬",
            f"{summary['strategy_total'] * 100:+.2f}%",
        )

    with p2:
        st.metric(
            "Buy & Hold",
            f"{summary['buyhold_total'] * 100:+.2f}%",
        )

    with p3:
        dd = summary["max_dd"]
        st.metric(
            "最大回撤",
            "N/A" if pd.isna(dd)
            else f"{dd * 100:.2f}%",
        )

    st.plotly_chart(
        equity_chart(wf),
        use_container_width=True,
    )

    st.subheader("最近測試結果")

    recent = wf.tail(100).copy()

    recent["Score"] = recent["Score"].round(3)
    recent["PUp"] = (
        recent["PUp"] * 100
    ).round(1)

    recent["Actual1D"] = (
        recent["Actual1D"] * 100
    ).round(2)

    recent["Actual5D"] = (
        recent["Actual5D"] * 100
    ).round(2)

    recent["Strategy1D"] = (
        recent["Strategy1D"] * 100
    ).round(2)

    st.dataframe(
        recent[
            [
                "Date",
                "Hexagram",
                "MovingLine",
                "Score",
                "PUp",
                "HistoryN",
                "Actual1D",
                "Actual5D",
                "Strategy1D",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "⬇️ 匯出 V3 Walk-forward CSV",
        wf.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name=(
            f"{symbol.replace('.', '_')}"
            "_v3_walkforward.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# Tab 4
# ============================================================

with tab4:
    st.subheader("64 卦樣本外條件統計")

    hs = hex_stats(wf)

    if hs.empty:
        st.warning(
            "目前訊號門檻下沒有足夠交易樣本。"
        )
    else:
        show = hs.copy()

        show["P(↑)"] = (
            show["p_up"] * 100
        ).round(1)

        show["平均隔日%"] = (
            show["avg1"] * 100
        ).round(2)

        show["平均5日%"] = (
            show["avg5"] * 100
        ).round(2)

        show["模型命中%"] = (
            show["win_model"] * 100
        ).round(1)

        show = show[
            [
                "HexNo",
                "Hexagram",
                "samples",
                "P(↑)",
                "平均隔日%",
                "平均5日%",
                "模型命中%",
            ]
        ]

        show.columns = [
            "卦序",
            "卦名",
            "樣本",
            "上漲率%",
            "平均隔日%",
            "平均5日%",
            "模型命中%",
        ]

        st.dataframe(
            show,
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("動爻統計")

    ls = line_stats(wf)

    if ls.empty:
        st.info(
            "目前沒有足夠的動爻交易樣本。"
        )
    else:
        line_show = ls.copy()

        line_show["上漲率%"] = (
            line_show["p_up"] * 100
        ).round(1)

        line_show["平均隔日%"] = (
            line_show["avg1"] * 100
        ).round(2)

        line_show["模型命中%"] = (
            line_show["win_model"] * 100
        ).round(1)

        st.dataframe(
            line_show[
                [
                    "MovingLine",
                    "samples",
                    "上漲率%",
                    "平均隔日%",
                    "模型命中%",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

    st.info(
        "注意：卦象與動爻樣本天然會被切得很細。"
        "如果樣本不足，不應把高勝率視為可靠訊號。"
    )


# ============================================================
# Tab 5
# ============================================================

with tab5:
    st.subheader("研究資料集")

    st.dataframe(
        dataset.tail(300),
        use_container_width=True,
    )

    st.download_button(
        "⬇️ 匯出完整研究資料",
        dataset.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name=(
            f"{symbol.replace('.', '_')}"
            "_v3_dataset.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# Disclaimer
# ============================================================

st.divider()

st.warning(
    """
    ⚠️ 研究限制

    1. 易經是傳統占筮文化，本 App 不宣稱卦象本身具有已證實的股票預測能力。
    2. V3 已改成 walk-forward 的樣本外框架，但仍可能受到資料探勘、樣本選擇、
       regime change、交易成本、滑價與市場制度變化影響。
    3. 本回測的交易成本是簡化 bps 假設，不等於台股實際完整成本。
    4. V3 的條件統計仍屬簡化模型；正式研究應再加入多重比較修正、bootstrap、
       信賴區間、基準模型與不同市場期間的穩健性測試。
    5. 歷史績效不代表未來績效，也不構成任何投資建議。
    """
)

st.caption(
    "易經 × 台股量化研究 V3｜Research / Education / Entertainment Only"
)

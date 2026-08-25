import os
import time
import math
import threading
import logging

from datetime import datetime, timedelta, time as dt_time
from zoneinfo import ZoneInfo

import requests
import pandas as pd

from flask import Flask, jsonify, render_template_string


# ============================================================
# APP
# ============================================================

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


# ============================================================
# ALPACA
# ============================================================

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "").strip()
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "").strip()

TRADING_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"
DATA_FEED = os.getenv("DATA_FEED", "iex").strip().lower()

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}


# ============================================================
# SCANNER SETTINGS
# ============================================================

TIMEFRAME = "4Min"

EMA_FAST = 5
EMA_SLOW = 9
EMA_TREND = 30

PREMARKET_START = dt_time(4, 0)
PREMARKET_END = dt_time(9, 30)

MIN_PRICE = float(os.getenv("MIN_PRICE", "5"))
MIN_DOLLAR_VOLUME = float(os.getenv("MIN_DOLLAR_VOLUME", "5000000"))
MAX_LIVE_UNIVERSE = int(os.getenv("MAX_LIVE_UNIVERSE", "250"))
WATCHLIST_SIZE = int(os.getenv("WATCHLIST_SIZE", "10"))
MIN_SETUP_SCORE = float(os.getenv("MIN_SETUP_SCORE", "70"))
SCAN_SECONDS = int(os.getenv("SCAN_SECONDS", "240"))
SNAPSHOT_BATCH = int(os.getenv("SNAPSHOT_BATCH", "200"))
BAR_BATCH = int(os.getenv("BAR_BATCH", "50"))
LEVEL_LOOKBACK = int(os.getenv("LEVEL_LOOKBACK", "90"))

PRIORITY = [
    "SPY",
    "QQQ",
    "IWM",
    "AAPL",
    "NVDA",
    "TSLA",
    "AMD",
    "AMZN",
    "META",
    "MSFT",
    "GOOGL",
    "NFLX",
    "AVGO",
    "PLTR",
    "COIN",
    "MSTR",
]


# ============================================================
# STATE
# ============================================================

lock = threading.Lock()

STATE = {
    "status": "STARTING",
    "last_scan": None,
    "universe_count": 0,
    "liquid_count": 0,
    "watchlist_count": 0,
    "watchlist": [],
    "error": None,
}


# ============================================================
# REQUEST
# ============================================================

def req(method, url, params=None, timeout=45):
    response = requests.request(
        method,
        url,
        headers=HEADERS,
        params=params,
        timeout=timeout,
    )

    if not response.ok:
        raise RuntimeError(
            f"{response.status_code}: {response.text[:400]}"
        )

    return response.json() if response.text else {}


# ============================================================
# HELPERS
# ============================================================

def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


# ============================================================
# STOCK UNIVERSE
# ============================================================

def get_universe():
    assets = req(
        "GET",
        f"{TRADING_URL}/v2/assets",
        params={
            "status": "active",
            "asset_class": "us_equity",
        },
    )

    symbols = []

    for asset in assets:
        symbol = str(
            asset.get("symbol", "")
        ).upper().strip()

        if not symbol:
            continue

        if not asset.get("tradable", False):
            continue

        if "/" in symbol or "." in symbol:
            continue

        symbols.append(symbol)

    return list(
        dict.fromkeys(
            PRIORITY + sorted(set(symbols))
        )
    )


# ============================================================
# LIQUIDITY FILTER
# ============================================================

def liquid_universe(symbols):
    ranked = []

    for batch in chunks(
        symbols,
        SNAPSHOT_BATCH
    ):
        data = req(
            "GET",
            f"{DATA_URL}/v2/stocks/snapshots",
            params={
                "symbols": ",".join(batch),
                "feed": DATA_FEED,
            },
        )

        for symbol, snapshot in (data or {}).items():
            day = snapshot.get("dailyBar") or {}
            previous = snapshot.get("prevDailyBar") or {}

            price = float(
                (snapshot.get("latestTrade") or {}).get("p")
                or day.get("c")
                or 0
            )

            volume = float(
                day.get("v")
                or previous.get("v")
                or 0
            )

            dollar_volume = price * volume

            if (
                price >= MIN_PRICE
                and
                dollar_volume >= MIN_DOLLAR_VOLUME
            ):
                ranked.append(
                    (
                        symbol,
                        dollar_volume,
                        price,
                    )
                )

        time.sleep(0.05)

    ranked.sort(
        key=lambda item: item[1],
        reverse=True
    )

    keep = {
        symbol
        for symbol, _, _
        in ranked[:MAX_LIVE_UNIVERSE]
    }

    keep.update(
        symbol
        for symbol
        in PRIORITY
        if symbol in symbols
    )

    metadata = {
        symbol: {
            "dollar_volume": dollar_volume,
            "snapshot_price": price,
        }
        for symbol, dollar_volume, price
        in ranked
        if symbol in keep
    }

    return list(keep), metadata


# ============================================================
# BARS
# ============================================================

def get_batch_bars(
    symbols,
    days=3
):
    if not symbols:
        return {}

    end = datetime.now(UTC)
    start = end - timedelta(days=days)

    params = {
        "symbols": ",".join(symbols),
        "timeframe": TIMEFRAME,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "adjustment": "raw",
        "feed": DATA_FEED,
        "sort": "asc",
        "limit": 10000,
    }

    output = {
        symbol: []
        for symbol in symbols
    }

    page_token = None

    while True:
        if page_token:
            params["page_token"] = page_token
        else:
            params.pop("page_token", None)

        data = req(
            "GET",
            f"{DATA_URL}/v2/stocks/bars",
            params=params,
        )

        for symbol, bars in (
            data.get("bars") or {}
        ).items():
            output.setdefault(
                symbol,
                []
            ).extend(bars)

        page_token = data.get(
            "next_page_token"
        )

        if not page_token:
            break

    return output


def to_df(bars):
    if not bars:
        return pd.DataFrame()

    df = pd.DataFrame(
        bars
    ).rename(
        columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
    )

    required = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    if not required.issubset(df.columns):
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = (
        df
        .set_index("timestamp")
        .tz_convert(NY)
        .sort_index()
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):
    df = df.copy()

    df["ema5"] = df["close"].ewm(
        span=EMA_FAST,
        adjust=False
    ).mean()

    df["ema9"] = df["close"].ewm(
        span=EMA_SLOW,
        adjust=False
    ).mean()

    df["ema30"] = df["close"].ewm(
        span=EMA_TREND,
        adjust=False
    ).mean()

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    df["atr"] = true_range.rolling(
        14,
        min_periods=5
    ).mean()

    dates = pd.Series(
        df.index.date,
        index=df.index
    )

    typical_price = (
        df["high"]
        + df["low"]
        + df["close"]
    ) / 3

    cumulative_volume = (
        df["volume"]
        .groupby(dates)
        .cumsum()
        .replace(0, math.nan)
    )

    df["vwap"] = (
        (typical_price * df["volume"])
        .groupby(dates)
        .cumsum()
        /
        cumulative_volume
    )

    df["vol_sma20"] = (
        df["volume"]
        .rolling(
            20,
            min_periods=5
        )
        .mean()
    )

    return df


def closed_only(df):
    if df.empty:
        return df

    now = datetime.now(NY)

    return df[
        df.index
        + pd.Timedelta(minutes=4)
        <= now
    ]


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def cluster_levels(
    values,
    tolerance
):
    values = sorted(
        float(value)
        for value in values
        if pd.notna(value)
    )

    clusters = []

    for value in values:
        if (
            not clusters
            or
            abs(
                value
                - sum(clusters[-1])
                / len(clusters[-1])
            )
            > tolerance
        ):
            clusters.append([value])
        else:
            clusters[-1].append(value)

    return [
        (
            sum(cluster)
            / len(cluster),
            len(cluster),
        )
        for cluster in clusters
    ]


def levels_from_df(
    df,
    price,
    atr
):
    window = df.tail(
        LEVEL_LOOKBACK
    )

    lows = []
    highs = []

    for i in range(
        2,
        len(window) - 2
    ):
        low = float(
            window["low"].iloc[i]
        )

        high = float(
            window["high"].iloc[i]
        )

        if low <= float(
            window["low"]
            .iloc[i - 2:i + 3]
            .min()
        ):
            lows.append(low)

        if high >= float(
            window["high"]
            .iloc[i - 2:i + 3]
            .max()
        ):
            highs.append(high)

    tolerance = max(
        price * 0.0008,
        atr * 0.18,
        0.02,
    )

    supports = cluster_levels(
        lows,
        tolerance
    )

    resistances = cluster_levels(
        highs,
        tolerance
    )

    below = [
        (level, touches)
        for level, touches in supports
        if level < price
    ]

    above = [
        (level, touches)
        for level, touches in resistances
        if level > price
    ]

    support = max(
        below,
        default=(None, 0),
        key=lambda item: item[0]
    )

    resistance = min(
        above,
        default=(None, 0),
        key=lambda item: item[0]
    )

    return (
        support,
        resistance,
        tolerance
    )


def previous_day_levels(
    df,
    today
):
    previous = df[
        df.index.date < today
    ]

    if previous.empty:
        return None, None

    previous_date = (
        previous.index.date[-1]
    )

    previous_day = previous[
        previous.index.date
        == previous_date
    ]

    return (
        float(
            previous_day["high"].max()
        ),
        float(
            previous_day["low"].min()
        )
    )


# ============================================================
# ANALYZE
# ============================================================

def analyze(
    symbol,
    raw,
    metadata
):
    df = closed_only(
        add_indicators(
            to_df(raw)
        )
    )

    if len(df) < 35:
        return None

    today = datetime.now(
        NY
    ).date()

    today_df = df[
        df.index.date == today
    ]

    if len(today_df) < 3:
        return None

    row = today_df.iloc[-1]
    previous = today_df.iloc[-2]

    price = float(
        row["close"]
    )

    atr = float(
        row["atr"]
        if pd.notna(row["atr"])
        else max(
            price * 0.003,
            0.05
        )
    )

    vwap = float(
        row["vwap"]
        if pd.notna(row["vwap"])
        else price
    )

    rvol = float(
        row["volume"]
        / row["vol_sma20"]
        if (
            pd.notna(
                row["vol_sma20"]
            )
            and
            row["vol_sma20"]
        )
        else 1.0
    )

    premarket = today_df[
        (
            today_df.index.time
            >= PREMARKET_START
        )
        &
        (
            today_df.index.time
            < PREMARKET_END
        )
    ]

    pm_high = (
        float(
            premarket["high"].max()
        )
        if not premarket.empty
        else None
    )

    pm_low = (
        float(
            premarket["low"].min()
        )
        if not premarket.empty
        else None
    )

    previous_day_high, previous_day_low = (
        previous_day_levels(
            df,
            today
        )
    )

    support, resistance, tolerance = (
        levels_from_df(
            df,
            price,
            atr
        )
    )

    support_level, support_touches = support
    resistance_level, resistance_touches = resistance

    if (
        pm_low
        and pm_low < price
        and (
            support_level is None
            or pm_low > support_level
        )
    ):
        support_level = pm_low
        support_touches = 3

    if (
        previous_day_low
        and previous_day_low < price
        and (
            support_level is None
            or previous_day_low > support_level
        )
    ):
        support_level = previous_day_low
        support_touches = max(
            support_touches,
            2
        )

    if (
        pm_high
        and pm_high > price
        and (
            resistance_level is None
            or pm_high < resistance_level
        )
    ):
        resistance_level = pm_high
        resistance_touches = 3

    if (
        previous_day_high
        and previous_day_high > price
        and (
            resistance_level is None
            or previous_day_high < resistance_level
        )
    ):
        resistance_level = previous_day_high
        resistance_touches = max(
            resistance_touches,
            2
        )

    bull = (
        row["ema5"]
        >
        row["ema9"]
        >
        row["ema30"]
    )

    bear = (
        row["ema5"]
        <
        row["ema9"]
        <
        row["ema30"]
    )

    momentum = (
        price
        -
        float(
            today_df["close"].iloc[-3]
        )
    ) / max(
        atr,
        0.000001
    )

    if (
        bull
        and
        price > vwap
    ):
        direction = "CALL"

    elif (
        bear
        and
        price < vwap
    ):
        direction = "PUT"

    elif momentum > 0.35:
        direction = "CALL"

    elif momentum < -0.35:
        direction = "PUT"

    else:
        direction = "NONE"

    if direction == "NONE":
        return None

    trigger = (
        resistance_level
        if direction == "CALL"
        else support_level
    )

    if trigger is None:
        return None

    distance = abs(
        price - trigger
    )

    proximity = max(
        0.0,
        1.0
        -
        distance
        /
        max(
            atr * 1.25,
            0.05
        )
    )

    if direction == "CALL":
        above = sorted(
            [
                level
                for level, _
                in cluster_levels(
                    df.tail(
                        LEVEL_LOOKBACK
                    )["high"].tolist(),
                    tolerance
                )
                if level
                >
                max(
                    price,
                    trigger
                )
            ]
        )

        target = (
            above[0]
            if above
            else
            trigger
            +
            max(
                atr,
                price * 0.002
            )
        )

    else:
        below = sorted(
            [
                level
                for level, _
                in cluster_levels(
                    df.tail(
                        LEVEL_LOOKBACK
                    )["low"].tolist(),
                    tolerance
                )
                if level
                <
                min(
                    price,
                    trigger
                )
            ],
            reverse=True
        )

        target = (
            below[0]
            if below
            else
            trigger
            -
            max(
                atr,
                price * 0.002
            )
        )

    score = 0.0

    if (
        (
            direction == "CALL"
            and bull
        )
        or
        (
            direction == "PUT"
            and bear
        )
    ):
        score += 25

    if (
        (
            direction == "CALL"
            and price > vwap
        )
        or
        (
            direction == "PUT"
            and price < vwap
        )
    ):
        score += 15

    score += min(
        max(
            rvol - 0.8,
            0
        )
        /
        1.7,
        1
    ) * 15

    score += min(
        abs(momentum)
        /
        1.2,
        1
    ) * 10

    score += (
        proximity * 20
    )

    touches = (
        resistance_touches
        if direction == "CALL"
        else support_touches
    )

    score += min(
        touches / 3,
        1
    ) * 10

    room = (
        abs(
            target - trigger
        )
        if target
        else 0
    )

    if room >= atr * 0.6:
        score += 5

    score = round(
        min(
            score,
            100
        ),
        1
    )

    previous_close = float(
        previous["close"]
    )

    if direction == "CALL":
        crossed = (
            previous_close <= trigger
            and
            price > trigger
        )

        if crossed:
            status = "BREAK_CONFIRMED"
        elif price <= trigger + tolerance:
            status = "WAITING_FOR_BREAK"
        else:
            status = "ABOVE_LEVEL"

    else:
        crossed = (
            previous_close >= trigger
            and
            price < trigger
        )

        if crossed:
            status = "BREAK_CONFIRMED"
        elif price >= trigger - tolerance:
            status = "WAITING_FOR_BREAK"
        else:
            status = "BELOW_LEVEL"

    return {
        "symbol": symbol,
        "direction": direction,
        "score": score,
        "status": status,
        "price": round(
            price,
            4
        ),
        "trigger": round(
            trigger,
            4
        ),
        "support": (
            round(
                support_level,
                4
            )
            if support_level
            else None
        ),
        "resistance": (
            round(
                resistance_level,
                4
            )
            if resistance_level
            else None
        ),
        "target": (
            round(
                target,
                4
            )
            if target
            else None
        ),
        "ema5": round(
            float(
                row["ema5"]
            ),
            4
        ),
        "ema9": round(
            float(
                row["ema9"]
            ),
            4
        ),
        "ema30": round(
            float(
                row["ema30"]
            ),
            4
        ),
        "vwap": round(
            vwap,
            4
        ),
        "atr": round(
            atr,
            4
        ),
        "rvol": round(
            rvol,
            2
        ),
        "dollar_volume": round(
            float(
                metadata.get(
                    symbol,
                    {}
                ).get(
                    "dollar_volume",
                    0
                )
            ),
            2
        ),
        "bar_time": (
            today_df.index[-1]
            .isoformat()
        ),
        "touches": int(
            touches
        ),
    }


# ============================================================
# SCAN
# ============================================================

def run_scan():
    with lock:
        STATE.update(
            status="SCANNING",
            error=None
        )

    symbols = get_universe()

    live, metadata = liquid_universe(
        symbols
    )

    results = []

    for batch in chunks(
        live,
        BAR_BATCH
    ):
        bars = get_batch_bars(
            batch
        )

        for symbol in batch:
            try:
                item = analyze(
                    symbol,
                    bars.get(
                        symbol,
                        []
                    ),
                    metadata
                )

                if (
                    item
                    and
                    item["score"]
                    >= MIN_SETUP_SCORE
                ):
                    results.append(
                        item
                    )

            except Exception as error:
                logging.warning(
                    "%s analyze error: %s",
                    symbol,
                    error
                )

        time.sleep(0.05)

    results.sort(
        key=lambda item: (
            item["score"],
            item["dollar_volume"]
        ),
        reverse=True
    )

    watchlist = results[
        :WATCHLIST_SIZE
    ]

    with lock:
        STATE.update(
            status="READY",
            last_scan=datetime.now(
                NY
            ).isoformat(),
            universe_count=len(
                symbols
            ),
            liquid_count=len(
                live
            ),
            watchlist_count=len(
                watchlist
            ),
            watchlist=watchlist,
            error=None
        )

    logging.info(
        "SCAN READY | universe=%s | liquid=%s | watch=%s",
        len(symbols),
        len(live),
        len(watchlist)
    )


# ============================================================
# LOOP
# ============================================================

def loop():
    while True:
        try:
            run_scan()

        except Exception as error:
            logging.exception(
                "SCAN FAILED"
            )

            with lock:
                STATE.update(
                    status="ERROR",
                    error=str(
                        error
                    )[:500]
                )

        time.sleep(
            SCAN_SECONDS
        )


# ============================================================
# MOBILE WATCHLIST WEBSITE
# ============================================================

HTML = """
<!doctype html>
<html>

<head>

<meta
    name="viewport"
    content="width=device-width,initial-scale=1,viewport-fit=cover"
>

<meta
    http-equiv="refresh"
    content="20"
>

<title>
90% AI Trade Scanner
</title>

<style>

*{
    box-sizing:border-box;
}

body{
    margin:0;
    background:#080d14;
    color:#f4f7fb;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;
    min-height:100vh;
}

.container{
    max-width:950px;
    margin:auto;
    padding:18px 13px 35px;
}

.header{
    display:flex;
    justify-content:space-between;
    align-items:flex-start;
    gap:12px;
    margin-bottom:15px;
}

.title{
    font-size:27px;
    font-weight:950;
}

.subtitle{
    color:#8da0b5;
    font-size:12px;
    margin-top:5px;
    line-height:1.4;
}

.ready{
    background:#10281e;
    border:1px solid #275a40;
    color:#54df8d;
    border-radius:999px;
    padding:7px 10px;
    font-size:11px;
    font-weight:900;
}

.stats{
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:7px;
    margin-bottom:15px;
}

.stat{
    background:#111923;
    border:1px solid #213044;
    border-radius:13px;
    padding:10px;
}

.stat-label{
    color:#7d91a7;
    font-size:9px;
    text-transform:uppercase;
}

.stat-value{
    margin-top:4px;
    font-size:19px;
    font-weight:900;
}

.card{
    background:#111923;
    border:1px solid #26384e;
    border-radius:17px;
    padding:14px;
    margin:11px 0;
}

.card.call{
    border-left:4px solid #43dc81;
}

.card.put{
    border-left:4px solid #ff626d;
}

.top{
    display:flex;
    justify-content:space-between;
    align-items:flex-start;
}

.symbol{
    font-size:29px;
    font-weight:950;
}

.direction{
    margin-top:6px;
    font-size:12px;
    font-weight:900;
}

.call-text{
    color:#48df87;
}

.put-text{
    color:#ff6d77;
}

.score{
    text-align:right;
    font-size:28px;
    font-weight:950;
}

.score-label{
    color:#7c91a7;
    font-size:8px;
    text-align:right;
}

.status{
    display:inline-block;
    margin-top:11px;
    background:#192638;
    border:1px solid #29405a;
    border-radius:9px;
    padding:6px 9px;
    color:#c5d4e4;
    font-size:10px;
    font-weight:900;
}

.confirmed{
    background:#2c2613;
    border-color:#66551d;
    color:#ffd25e;
}

.levels{
    display:grid;
    grid-template-columns:repeat(2,minmax(0,1fr));
    gap:7px;
    margin-top:12px;
}

.level{
    background:#091019;
    border:1px solid #1e2c3d;
    border-radius:11px;
    padding:9px;
}

.label{
    color:#798da4;
    font-size:8px;
    text-transform:uppercase;
}

.value{
    margin-top:3px;
    font-size:16px;
    font-weight:850;
}

.entry{
    color:#ffd15c;
}

.support{
    color:#62e09a;
}

.resistance{
    color:#ff7e87;
}

.target{
    color:#6cd2ff;
}

.indicators{
    display:grid;
    grid-template-columns:repeat(4,minmax(0,1fr));
    gap:5px;
    margin-top:9px;
}

.indicator{
    background:#151f2b;
    border-radius:8px;
    padding:7px 3px;
    text-align:center;
}

.indicator-label{
    color:#74899f;
    font-size:8px;
}

.indicator-value{
    margin-top:2px;
    font-size:11px;
    font-weight:850;
}

.empty{
    background:#111923;
    border:1px solid #26384e;
    border-radius:15px;
    padding:24px 16px;
    text-align:center;
    color:#91a4b9;
    line-height:1.5;
}

.error{
    margin:10px 0;
    background:#2a1218;
    border:1px solid #68313b;
    border-radius:10px;
    padding:10px;
    color:#ff8a94;
    font-size:11px;
}

.footer{
    margin-top:15px;
    color:#637990;
    font-size:9px;
    text-align:center;
    line-height:1.6;
}

@media(min-width:700px){

    .cards{
        display:grid;
        grid-template-columns:repeat(2,minmax(0,1fr));
        gap:11px;
    }

    .card{
        margin:0;
    }
}

</style>

</head>

<body>

<div class="container">


<div class="header">

<div>

<div class="title">
90% AI Trade Scanner
</div>

<div class="subtitle">
CALL / PUT candidates with support,
resistance, entry trigger and target
</div>

</div>

<div class="ready">
{{ state.status }}
</div>

</div>


<div class="stats">

<div class="stat">

<div class="stat-label">
WATCHING
</div>

<div class="stat-value">
{{ state.watchlist_count }}
</div>

</div>


<div class="stat">

<div class="stat-label">
LIQUID
</div>

<div class="stat-value">
{{ state.liquid_count }}
</div>

</div>


<div class="stat">

<div class="stat-label">
UNIVERSE
</div>

<div class="stat-value">
{{ state.universe_count }}
</div>

</div>

</div>


{% if state.error %}

<div class="error">
{{ state.error }}
</div>

{% endif %}


{% if state.watchlist %}

<div class="cards">

{% for item in state.watchlist %}

<div class="card {{ 'call' if item.direction == 'CALL' else 'put' }}">

<div class="top">

<div>

<div class="symbol">
{{ item.symbol }}
</div>

<div class="direction {{ 'call-text' if item.direction == 'CALL' else 'put-text' }}">
{{ item.direction }}
</div>

</div>


<div>

<div class="score">
{{ item.score }}
</div>

<div class="score-label">
SETUP SCORE
</div>

</div>

</div>


<div class="status {{ 'confirmed' if item.status == 'BREAK_CONFIRMED' else '' }}">

{{ item.status.replace('_',' ') }}

</div>


<div class="levels">


<div class="level">

<div class="label">
CURRENT PRICE
</div>

<div class="value">
${{ "%.2f"|format(item.price) }}
</div>

</div>


<div class="level">

<div class="label">
ENTRY TRIGGER
</div>

<div class="value entry">
${{ "%.2f"|format(item.trigger) }}
</div>

</div>


<div class="level">

<div class="label">
SUPPORT
</div>

<div class="value support">

{% if item.support is not none %}

${{ "%.2f"|format(item.support) }}

{% else %}

—

{% endif %}

</div>

</div>


<div class="level">

<div class="label">
RESISTANCE
</div>

<div class="value resistance">

{% if item.resistance is not none %}

${{ "%.2f"|format(item.resistance) }}

{% else %}

—

{% endif %}

</div>

</div>


<div class="level">

<div class="label">
TARGET
</div>

<div class="value target">

{% if item.target is not none %}

${{ "%.2f"|format(item.target) }}

{% else %}

—

{% endif %}

</div>

</div>


<div class="level">

<div class="label">
LEVEL TOUCHES
</div>

<div class="value">
{{ item.touches }}
</div>

</div>


</div>


<div class="indicators">


<div class="indicator">

<div class="indicator-label">
RVOL
</div>

<div class="indicator-value">
{{ item.rvol }}x
</div>

</div>


<div class="indicator">

<div class="indicator-label">
EMA5
</div>

<div class="indicator-value">
{{ "%.2f"|format(item.ema5) }}
</div>

</div>


<div class="indicator">

<div class="indicator-label">
EMA9
</div>

<div class="indicator-value">
{{ "%.2f"|format(item.ema9) }}
</div>

</div>


<div class="indicator">

<div class="indicator-label">
VWAP
</div>

<div class="indicator-value">
{{ "%.2f"|format(item.vwap) }}
</div>

</div>


</div>


</div>

{% endfor %}

</div>


{% else %}

<div class="empty">

<strong>
No qualifying setups yet.
</strong>

<br><br>

The scanner will automatically add
CALL or PUT candidates when they meet
the minimum setup score.

</div>

{% endif %}


<div class="footer">

Last scan:

{% if state.last_scan %}

{{ state.last_scan }}

{% else %}

Waiting for first scan...

{% endif %}

<br>

Scanner runs every {{ scan_seconds }} seconds.

Website refreshes every 20 seconds.

</div>


</div>

</body>

</html>
"""


# ============================================================
# WEBSITE
# ============================================================

@app.get("/")
def home():
    with lock:
        snapshot = dict(
            STATE
        )

    return render_template_string(
        HTML,
        state=snapshot,
        scan_seconds=SCAN_SECONDS
    )


# ============================================================
# WATCHLIST API
# main.py READS THIS
# ============================================================

@app.get("/api/watchlist")
def api_watchlist():
    with lock:
        return jsonify(
            dict(
                STATE
            )
        )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    with lock:
        return jsonify({
            "ok": True,
            "status": STATE["status"],
            "last_scan": STATE["last_scan"],
            "watchlist_count": STATE["watchlist_count"],
            "timeframe": TIMEFRAME,
            "min_setup_score": MIN_SETUP_SCORE,
        })


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    threading.Thread(
        target=loop,
        daemon=True
    ).start()

    app.run(
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                "10000"
            )
        )
    )

else:

    threading.Thread(
        target=loop,
        daemon=True
    ).start()
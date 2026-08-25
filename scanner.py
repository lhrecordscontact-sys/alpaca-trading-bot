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

ALPACA_API_KEY = os.getenv(
    "ALPACA_API_KEY",
    ""
).strip()

ALPACA_SECRET_KEY = os.getenv(
    "ALPACA_SECRET_KEY",
    ""
).strip()

TRADING_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"

DATA_FEED = os.getenv(
    "DATA_FEED",
    "iex"
).strip().lower()

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

MIN_PRICE = float(
    os.getenv(
        "MIN_PRICE",
        "5"
    )
)

MIN_DOLLAR_VOLUME = float(
    os.getenv(
        "MIN_DOLLAR_VOLUME",
        "5000000"
    )
)

MAX_LIVE_UNIVERSE = int(
    os.getenv(
        "MAX_LIVE_UNIVERSE",
        "250"
    )
)

WATCHLIST_SIZE = int(
    os.getenv(
        "WATCHLIST_SIZE",
        "10"
    )
)

NEAR_MISS_SIZE = int(
    os.getenv(
        "NEAR_MISS_SIZE",
        "10"
    )
)

MIN_SETUP_SCORE = float(
    os.getenv(
        "MIN_SETUP_SCORE",
        "70"
    )
)

# Recalculate the 250 active stocks every 30 minutes.
LIQUIDITY_REFRESH_SECONDS = int(
    os.getenv(
        "LIQUIDITY_REFRESH_SECONDS",
        "1800"
    )
)

# Rebuild the full Alpaca stock universe every 6 hours.
UNIVERSE_REFRESH_SECONDS = int(
    os.getenv(
        "UNIVERSE_REFRESH_SECONDS",
        "21600"
    )
)

# Analyze active stocks every 4 minutes.
SCAN_SECONDS = int(
    os.getenv(
        "SCAN_SECONDS",
        "240"
    )
)

SNAPSHOT_BATCH = int(
    os.getenv(
        "SNAPSHOT_BATCH",
        "200"
    )
)

BAR_BATCH = int(
    os.getenv(
        "BAR_BATCH",
        "50"
    )
)

LEVEL_LOOKBACK = int(
    os.getenv(
        "LEVEL_LOOKBACK",
        "90"
    )
)


# ============================================================
# PRIORITY SYMBOLS
# ============================================================

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
# CACHE
# ============================================================

CACHE = {
    "universe": [],
    "universe_updated": 0,

    "live": [],
    "metadata": {},
    "liquidity_updated": 0,
}


# ============================================================
# STATE
# ============================================================

lock = threading.Lock()

STATE = {
    "status": "STARTING",

    "last_scan": None,
    "scan_duration_seconds": None,

    "universe_count": 0,
    "liquid_count": 0,

    "universe_cache_age": None,
    "liquidity_cache_age": None,

    "watchlist_count": 0,
    "watchlist": [],

    "near_miss_count": 0,
    "near_misses": [],

    "qualification_threshold": MIN_SETUP_SCORE,

    "scan_interval_seconds": SCAN_SECONDS,

    "error": None,
}


# ============================================================
# REQUEST
# ============================================================

def req(
    method,
    url,
    params=None,
    timeout=45
):

    response = requests.request(
        method,
        url,
        headers=HEADERS,
        params=params,
        timeout=timeout,
    )

    if not response.ok:

        raise RuntimeError(
            f"{response.status_code}: "
            f"{response.text[:400]}"
        )

    if not response.text:
        return {}

    return response.json()


# ============================================================
# CHUNKS
# ============================================================

def chunks(
    items,
    size
):

    for i in range(
        0,
        len(items),
        size
    ):

        yield items[
            i:i + size
        ]


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
            asset.get(
                "symbol",
                ""
            )
        ).upper().strip()

        if not symbol:
            continue

        if not asset.get(
            "tradable",
            False
        ):
            continue

        if "/" in symbol:
            continue

        if "." in symbol:
            continue

        symbols.append(
            symbol
        )

    return list(
        dict.fromkeys(
            PRIORITY
            +
            sorted(
                set(symbols)
            )
        )
    )


# ============================================================
# CACHED UNIVERSE
# ============================================================

def cached_universe(
    force=False
):

    now = time.time()

    age = (
        now
        -
        CACHE[
            "universe_updated"
        ]
    )

    needs_refresh = (
        force
        or
        not CACHE["universe"]
        or
        age >= UNIVERSE_REFRESH_SECONDS
    )

    if needs_refresh:

        logging.info(
            "UNIVERSE REFRESH START"
        )

        symbols = get_universe()

        CACHE[
            "universe"
        ] = symbols

        CACHE[
            "universe_updated"
        ] = now

        logging.info(
            "UNIVERSE REFRESH COMPLETE | %s symbols",
            len(symbols)
        )

    return CACHE[
        "universe"
    ]


# ============================================================
# LIQUIDITY FILTER
# ============================================================

def liquid_universe(
    symbols
):

    ranked = []

    total_batches = math.ceil(
        len(symbols)
        /
        SNAPSHOT_BATCH
    )

    for number, batch in enumerate(
        chunks(
            symbols,
            SNAPSHOT_BATCH
        ),
        start=1
    ):

        logging.info(
            "LIQUIDITY %s/%s",
            number,
            total_batches
        )

        try:

            data = req(
                "GET",
                f"{DATA_URL}/v2/stocks/snapshots",
                params={
                    "symbols": ",".join(
                        batch
                    ),
                    "feed": DATA_FEED,
                },
            )

        except Exception as error:

            logging.warning(
                "Snapshot batch error: %s",
                error
            )

            continue

        for symbol, snapshot in (
            data or {}
        ).items():

            day = (
                snapshot.get(
                    "dailyBar"
                )
                or
                {}
            )

            previous = (
                snapshot.get(
                    "prevDailyBar"
                )
                or
                {}
            )

            price = float(
                (
                    snapshot.get(
                        "latestTrade"
                    )
                    or
                    {}
                ).get(
                    "p"
                )
                or
                day.get(
                    "c"
                )
                or
                previous.get(
                    "c"
                )
                or
                0
            )

            volume = float(
                day.get(
                    "v"
                )
                or
                previous.get(
                    "v"
                )
                or
                0
            )

            dollar_volume = (
                price
                *
                volume
            )

            if (
                price >= MIN_PRICE
                and
                dollar_volume
                >=
                MIN_DOLLAR_VOLUME
            ):

                ranked.append(
                    (
                        symbol,
                        dollar_volume,
                        price,
                    )
                )

    ranked.sort(
        key=lambda item:
        item[1],
        reverse=True
    )

    keep = {
        symbol
        for symbol, _, _
        in ranked[
            :MAX_LIVE_UNIVERSE
        ]
    }

    keep.update(
        symbol
        for symbol
        in PRIORITY
        if symbol in symbols
    )

    metadata = {
        symbol: {
            "dollar_volume":
                dollar_volume,

            "snapshot_price":
                price,
        }
        for symbol, dollar_volume, price
        in ranked
        if symbol in keep
    }

    # Preserve liquidity ranking.
    live = [
        symbol
        for symbol, _, _
        in ranked
        if symbol in keep
    ]

    for symbol in PRIORITY:

        if (
            symbol in keep
            and
            symbol not in live
        ):
            live.append(
                symbol
            )

    return (
        live,
        metadata
    )


# ============================================================
# CACHED LIQUIDITY POOL
# ============================================================

def cached_liquidity(
    symbols,
    force=False
):

    now = time.time()

    age = (
        now
        -
        CACHE[
            "liquidity_updated"
        ]
    )

    needs_refresh = (
        force
        or
        not CACHE["live"]
        or
        age >= LIQUIDITY_REFRESH_SECONDS
    )

    if needs_refresh:

        logging.info(
            "LIQUIDITY REFRESH START | universe=%s",
            len(symbols)
        )

        live, metadata = (
            liquid_universe(
                symbols
            )
        )

        if live:

            CACHE[
                "live"
            ] = live

            CACHE[
                "metadata"
            ] = metadata

            CACHE[
                "liquidity_updated"
            ] = now

            logging.info(
                "LIQUIDITY REFRESH COMPLETE | active=%s",
                len(live)
            )

        else:

            logging.warning(
                "Liquidity refresh returned no symbols. "
                "Keeping previous cache."
            )

    return (
        CACHE[
            "live"
        ],
        CACHE[
            "metadata"
        ]
    )


# ============================================================
# BATCH BARS
# ============================================================

def get_batch_bars(
    symbols,
    days=3
):

    if not symbols:
        return {}

    end = datetime.now(
        UTC
    )

    start = (
        end
        -
        timedelta(
            days=days
        )
    )

    params = {
        "symbols": ",".join(
            symbols
        ),

        "timeframe":
            TIMEFRAME,

        "start":
            start.isoformat(),

        "end":
            end.isoformat(),

        "adjustment":
            "raw",

        "feed":
            DATA_FEED,

        "sort":
            "asc",

        "limit":
            10000,
    }

    output = {
        symbol: []
        for symbol
        in symbols
    }

    page_token = None

    while True:

        if page_token:

            params[
                "page_token"
            ] = page_token

        else:

            params.pop(
                "page_token",
                None
            )

        data = req(
            "GET",
            f"{DATA_URL}/v2/stocks/bars",
            params=params,
        )

        for symbol, bars in (
            data.get(
                "bars"
            )
            or
            {}
        ).items():

            output.setdefault(
                symbol,
                []
            ).extend(
                bars
            )

        page_token = (
            data.get(
                "next_page_token"
            )
        )

        if not page_token:
            break

    return output


# ============================================================
# DATAFRAME
# ============================================================

def to_df(
    bars
):

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

    if not required.issubset(
        df.columns
    ):
        return pd.DataFrame()

    df[
        "timestamp"
    ] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = (
        df
        .set_index(
            "timestamp"
        )
        .tz_convert(
            NY
        )
        .sort_index()
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:

        df[
            column
        ] = pd.to_numeric(
            df[column],
            errors="coerce",
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

def add_indicators(
    df
):

    df = df.copy()

    df[
        "ema5"
    ] = df[
        "close"
    ].ewm(
        span=EMA_FAST,
        adjust=False
    ).mean()

    df[
        "ema9"
    ] = df[
        "close"
    ].ewm(
        span=EMA_SLOW,
        adjust=False
    ).mean()

    df[
        "ema30"
    ] = df[
        "close"
    ].ewm(
        span=EMA_TREND,
        adjust=False
    ).mean()

    previous_close = (
        df[
            "close"
        ].shift(
            1
        )
    )

    true_range = pd.concat(
        [
            (
                df["high"]
                -
                df["low"]
            ).abs(),

            (
                df["high"]
                -
                previous_close
            ).abs(),

            (
                df["low"]
                -
                previous_close
            ).abs(),
        ],
        axis=1,
    ).max(
        axis=1
    )

    df[
        "atr"
    ] = true_range.rolling(
        14,
        min_periods=5
    ).mean()

    dates = pd.Series(
        df.index.date,
        index=df.index
    )

    typical_price = (
        df["high"]
        +
        df["low"]
        +
        df["close"]
    ) / 3

    cumulative_volume = (
        df["volume"]
        .groupby(
            dates
        )
        .cumsum()
        .replace(
            0,
            math.nan
        )
    )

    df[
        "vwap"
    ] = (
        (
            typical_price
            *
            df["volume"]
        )
        .groupby(
            dates
        )
        .cumsum()
        /
        cumulative_volume
    )

    df[
        "vol_sma20"
    ] = (
        df["volume"]
        .rolling(
            20,
            min_periods=5
        )
        .mean()
    )

    return df


# ============================================================
# CLOSED 4-MIN CANDLES ONLY
# ============================================================

def closed_only(
    df
):

    if df.empty:
        return df

    now = datetime.now(
        NY
    )

    return df[
        df.index
        +
        pd.Timedelta(
            minutes=4
        )
        <=
        now
    ]


# ============================================================
# LEVEL CLUSTERS
# ============================================================

def cluster_levels(
    values,
    tolerance
):

    values = sorted(
        float(value)
        for value
        in values
        if pd.notna(
            value
        )
    )

    clusters = []

    for value in values:

        if not clusters:

            clusters.append(
                [value]
            )

            continue

        current_average = (
            sum(
                clusters[-1]
            )
            /
            len(
                clusters[-1]
            )
        )

        if abs(
            value
            -
            current_average
        ) > tolerance:

            clusters.append(
                [value]
            )

        else:

            clusters[-1].append(
                value
            )

    return [
        (
            sum(cluster)
            /
            len(cluster),

            len(cluster),
        )
        for cluster
        in clusters
    ]


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

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
            window[
                "low"
            ].iloc[i]
        )

        high = float(
            window[
                "high"
            ].iloc[i]
        )

        nearby_lows = (
            window[
                "low"
            ].iloc[
                i - 2:
                i + 3
            ]
        )

        nearby_highs = (
            window[
                "high"
            ].iloc[
                i - 2:
                i + 3
            ]
        )

        if low <= float(
            nearby_lows.min()
        ):

            lows.append(
                low
            )

        if high >= float(
            nearby_highs.max()
        ):

            highs.append(
                high
            )

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
        (
            level,
            touches
        )
        for level, touches
        in supports
        if level < price
    ]

    above = [
        (
            level,
            touches
        )
        for level, touches
        in resistances
        if level > price
    ]

    support = max(
        below,
        default=(
            None,
            0
        ),
        key=lambda item:
        item[0]
    )

    resistance = min(
        above,
        default=(
            None,
            0
        ),
        key=lambda item:
        item[0]
    )

    return (
        support,
        resistance,
        tolerance
    )


# ============================================================
# PREVIOUS DAY LEVELS
# ============================================================

def previous_day_levels(
    df,
    today
):

    previous = df[
        df.index.date
        <
        today
    ]

    if previous.empty:

        return (
            None,
            None
        )

    previous_date = (
        previous.index.date[
            -1
        ]
    )

    previous_day = previous[
        previous.index.date
        ==
        previous_date
    ]

    return (
        float(
            previous_day[
                "high"
            ].max()
        ),

        float(
            previous_day[
                "low"
            ].min()
        )
    )


# ============================================================
# ANALYZE STOCK
# ============================================================

def analyze(
    symbol,
    raw,
    metadata
):

    df = closed_only(
        add_indicators(
            to_df(
                raw
            )
        )
    )

    if len(df) < 35:
        return None

    today = datetime.now(
        NY
    ).date()

    today_df = df[
        df.index.date
        ==
        today
    ]

    if len(
        today_df
    ) < 3:
        return None

    row = today_df.iloc[
        -1
    ]

    previous = today_df.iloc[
        -2
    ]

    price = float(
        row[
            "close"
        ]
    )

    atr = float(
        row["atr"]
        if pd.notna(
            row["atr"]
        )
        else
        max(
            price * 0.003,
            0.05
        )
    )

    vwap = float(
        row["vwap"]
        if pd.notna(
            row["vwap"]
        )
        else
        price
    )

    if (
        pd.notna(
            row[
                "vol_sma20"
            ]
        )
        and
        row[
            "vol_sma20"
        ]
    ):

        rvol = float(
            row[
                "volume"
            ]
            /
            row[
                "vol_sma20"
            ]
        )

    else:

        rvol = 1.0


    # ========================================================
    # PREMARKET
    # ========================================================

    premarket = today_df[
        (
            today_df.index.time
            >=
            PREMARKET_START
        )
        &
        (
            today_df.index.time
            <
            PREMARKET_END
        )
    ]

    pm_high = (
        float(
            premarket[
                "high"
            ].max()
        )
        if not premarket.empty
        else
        None
    )

    pm_low = (
        float(
            premarket[
                "low"
            ].min()
        )
        if not premarket.empty
        else
        None
    )


    # ========================================================
    # PREVIOUS DAY
    # ========================================================

    (
        previous_day_high,
        previous_day_low
    ) = previous_day_levels(
        df,
        today
    )


    # ========================================================
    # SUPPORT / RESISTANCE
    # ========================================================

    (
        support,
        resistance,
        tolerance
    ) = levels_from_df(
        df,
        price,
        atr
    )

    (
        support_level,
        support_touches
    ) = support

    (
        resistance_level,
        resistance_touches
    ) = resistance


    # Premarket low as support.

    if (
        pm_low is not None
        and
        pm_low < price
        and
        (
            support_level is None
            or
            pm_low > support_level
        )
    ):

        support_level = pm_low
        support_touches = 3


    # Previous day low.

    if (
        previous_day_low
        is not None
        and
        previous_day_low
        <
        price
        and
        (
            support_level is None
            or
            previous_day_low
            >
            support_level
        )
    ):

        support_level = (
            previous_day_low
        )

        support_touches = max(
            support_touches,
            2
        )


    # Premarket high as resistance.

    if (
        pm_high is not None
        and
        pm_high > price
        and
        (
            resistance_level is None
            or
            pm_high
            <
            resistance_level
        )
    ):

        resistance_level = (
            pm_high
        )

        resistance_touches = 3


    # Previous day high.

    if (
        previous_day_high
        is not None
        and
        previous_day_high
        >
        price
        and
        (
            resistance_level is None
            or
            previous_day_high
            <
            resistance_level
        )
    ):

        resistance_level = (
            previous_day_high
        )

        resistance_touches = max(
            resistance_touches,
            2
        )


    # ========================================================
    # TREND
    # ========================================================

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
            today_df[
                "close"
            ].iloc[-3]
        )
    ) / max(
        atr,
        0.000001
    )


    # ========================================================
    # DIRECTION
    # ========================================================

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

        return None


    # ========================================================
    # TRIGGER
    # ========================================================

    trigger = (
        resistance_level
        if direction == "CALL"
        else
        support_level
    )

    if trigger is None:
        return None


    # ========================================================
    # PROXIMITY
    # ========================================================

    distance = abs(
        price
        -
        trigger
    )

    proximity = max(
        0.0,
        1.0
        -
        (
            distance
            /
            max(
                atr * 1.25,
                0.05
            )
        )
    )


    # ========================================================
    # TARGET
    # ========================================================

    if direction == "CALL":

        possible_targets = sorted(
            [
                level
                for level, _
                in cluster_levels(
                    df.tail(
                        LEVEL_LOOKBACK
                    )[
                        "high"
                    ].tolist(),

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

        if possible_targets:

            target = (
                possible_targets[0]
            )

        else:

            target = (
                trigger
                +
                max(
                    atr,
                    price * 0.002
                )
            )

    else:

        possible_targets = sorted(
            [
                level
                for level, _
                in cluster_levels(
                    df.tail(
                        LEVEL_LOOKBACK
                    )[
                        "low"
                    ].tolist(),

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

        if possible_targets:

            target = (
                possible_targets[0]
            )

        else:

            target = (
                trigger
                -
                max(
                    atr,
                    price * 0.002
                )
            )


    # ========================================================
    # SCORE
    # ========================================================

    score = 0.0


    # EMA trend = 25 points.

    if (
        (
            direction == "CALL"
            and
            bull
        )
        or
        (
            direction == "PUT"
            and
            bear
        )
    ):

        score += 25


    # VWAP = 15 points.

    if (
        (
            direction == "CALL"
            and
            price > vwap
        )
        or
        (
            direction == "PUT"
            and
            price < vwap
        )
    ):

        score += 15


    # RVOL = 15 points.

    score += min(
        max(
            rvol - 0.8,
            0
        )
        /
        1.7,
        1
    ) * 15


    # Momentum = 10 points.

    score += min(
        abs(
            momentum
        )
        /
        1.2,
        1
    ) * 10


    # Level proximity = 20.

    score += (
        proximity
        *
        20
    )


    # Level strength = 10.

    touches = (
        resistance_touches
        if direction == "CALL"
        else
        support_touches
    )

    score += min(
        touches
        /
        3,
        1
    ) * 10


    # Room to target = 5.

    room = abs(
        target
        -
        trigger
    )

    if room >= (
        atr * 0.6
    ):

        score += 5


    score = round(
        min(
            score,
            100
        ),
        1
    )


    # ========================================================
    # BREAK STATUS
    # ========================================================

    previous_close = float(
        previous[
            "close"
        ]
    )

    if direction == "CALL":

        crossed = (
            previous_close
            <=
            trigger
            and
            price
            >
            trigger
        )

        if crossed:

            status = (
                "BREAK_CONFIRMED"
            )

        elif price <= (
            trigger
            +
            tolerance
        ):

            status = (
                "WAITING_FOR_BREAK"
            )

        else:

            status = (
                "ABOVE_LEVEL"
            )

    else:

        crossed = (
            previous_close
            >=
            trigger
            and
            price
            <
            trigger
        )

        if crossed:

            status = (
                "BREAK_CONFIRMED"
            )

        elif price >= (
            trigger
            -
            tolerance
        ):

            status = (
                "WAITING_FOR_BREAK"
            )

        else:

            status = (
                "BELOW_LEVEL"
            )


    # ========================================================
    # RETURN
    # ========================================================

    return {
        "symbol":
            symbol,

        "direction":
            direction,

        "score":
            score,

        "status":
            status,

        "price":
            round(
                price,
                4
            ),

        "trigger":
            round(
                trigger,
                4
            ),

        "support":
            (
                round(
                    support_level,
                    4
                )
                if
                support_level
                is not None
                else
                None
            ),

        "resistance":
            (
                round(
                    resistance_level,
                    4
                )
                if
                resistance_level
                is not None
                else
                None
            ),

        "target":
            round(
                target,
                4
            ),

        "ema5":
            round(
                float(
                    row["ema5"]
                ),
                4
            ),

        "ema9":
            round(
                float(
                    row["ema9"]
                ),
                4
            ),

        "ema30":
            round(
                float(
                    row["ema30"]
                ),
                4
            ),

        "vwap":
            round(
                vwap,
                4
            ),

        "atr":
            round(
                atr,
                4
            ),

        "rvol":
            round(
                rvol,
                2
            ),

        "dollar_volume":
            round(
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

        "bar_time":
            today_df.index[
                -1
            ].isoformat(),

        "touches":
            int(
                touches
            ),
    }


# ============================================================
# RUN FAST SCAN
# ============================================================

def run_scan():

    started = time.time()

    with lock:

        STATE.update(
            status="SCANNING",
            error=None
        )

    # ========================================================
    # CACHED FULL MARKET UNIVERSE
    # ========================================================

    symbols = cached_universe()

    # ========================================================
    # CACHED 250-STOCK LIQUIDITY POOL
    # ========================================================

    live, metadata = (
        cached_liquidity(
            symbols
        )
    )

    if not live:

        raise RuntimeError(
            "No liquid symbols available"
        )

    logging.info(
        "FAST SCAN START | active=%s",
        len(live)
    )

    all_results = []

    total_batches = math.ceil(
        len(live)
        /
        BAR_BATCH
    )

    # ========================================================
    # ONLY THESE ~250 STOCKS ARE CHECKED EVERY 4 MINUTES
    # ========================================================

    for number, batch in enumerate(
        chunks(
            live,
            BAR_BATCH
        ),
        start=1
    ):

        logging.info(
            "BARS %s/%s | %s stocks",
            number,
            total_batches,
            len(batch)
        )

        try:

            bars = get_batch_bars(
                batch
            )

        except Exception as error:

            logging.warning(
                "Bars batch %s failed: %s",
                number,
                error
            )

            continue

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

                if item:

                    all_results.append(
                        item
                    )

            except Exception as error:

                logging.warning(
                    "%s analyze error: %s",
                    symbol,
                    error
                )


    # ========================================================
    # SORT BEST FIRST
    # ========================================================

    all_results.sort(
        key=lambda item: (
            item[
                "score"
            ],
            item[
                "dollar_volume"
            ]
        ),
        reverse=True
    )


    # ========================================================
    # QUALIFIED
    # ========================================================

    qualified = [
        item
        for item
        in all_results
        if item[
            "score"
        ]
        >=
        MIN_SETUP_SCORE
    ]

    watchlist = qualified[
        :WATCHLIST_SIZE
    ]

    scan_time = datetime.now(
        NY
    )

    scan_id = scan_time.strftime(
        "%Y%m%d-%H%M%S"
    )

    for item in watchlist:

        item[
            "qualification"
        ] = "QUALIFIED"

        item[
            "scan_id"
        ] = scan_id

        item[
            "scanned_at"
        ] = scan_time.isoformat()


    # ========================================================
    # NEAR MISSES
    # ========================================================

    near_misses = [
        item
        for item
        in all_results
        if item[
            "score"
        ]
        <
        MIN_SETUP_SCORE
    ][
        :NEAR_MISS_SIZE
    ]

    for item in near_misses:

        item[
            "qualification"
        ] = "WATCH_ONLY"


    # ========================================================
    # TIMING
    # ========================================================

    completed = time.time()

    duration = round(
        completed
        -
        started,
        2
    )

    universe_age = round(
        completed
        -
        CACHE[
            "universe_updated"
        ],
        1
    )

    liquidity_age = round(
        completed
        -
        CACHE[
            "liquidity_updated"
        ],
        1
    )


    # ========================================================
    # SAVE STATE
    # ========================================================

    with lock:

        STATE.update(
            status="READY",

            last_scan=(
                scan_time.isoformat()
            ),

            scan_duration_seconds=(
                duration
            ),

            universe_count=len(
                symbols
            ),

            liquid_count=len(
                live
            ),

            universe_cache_age=(
                universe_age
            ),

            liquidity_cache_age=(
                liquidity_age
            ),

            watchlist_count=len(
                watchlist
            ),

            watchlist=watchlist,

            near_miss_count=len(
                near_misses
            ),

            near_misses=near_misses,

            error=None,
        )

    logging.info(
        "SCAN READY | active=%s | setups=%s | qualified=%s | duration=%.2fs",
        len(live),
        len(all_results),
        len(watchlist),
        duration,
    )


# ============================================================
# LOOP
# ============================================================

def loop():

    while True:

        cycle_started = time.time()

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

        elapsed = (
            time.time()
            -
            cycle_started
        )

        # Important:
        # SCAN_SECONDS is now measured from the START
        # of the last scan instead of adding 240 seconds
        # after the scan finishes.

        sleep_seconds = max(
            5,
            SCAN_SECONDS
            -
            elapsed
        )

        logging.info(
            "NEXT SCAN IN %.1f seconds",
            sleep_seconds
        )

        time.sleep(
            sleep_seconds
        )


# ============================================================
# WEBSITE
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

<meta
name="theme-color"
content="#080d14"
>

<title>
AI Trade Setup Scanner
</title>

<style>

*{
    box-sizing:border-box;
}

body{
    margin:0;
    background:#080d14;
    color:#f4f7fb;
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        Arial,
        sans-serif;
}

.container{
    max-width:900px;
    margin:auto;
    padding:22px 16px 60px;
}

h1{
    font-size:30px;
    margin-bottom:5px;
}

.subtitle{
    color:#8fa2b8;
    line-height:1.5;
    margin-bottom:22px;
}

.badge{
    display:inline-block;
    background:#10281e;
    color:#54df8d;
    border:1px solid #275a40;
    border-radius:100px;
    padding:7px 12px;
    font-weight:800;
}

.stats{
    display:grid;
    grid-template-columns:repeat(2,1fr);
    gap:10px;
    margin:20px 0;
}

.stat{
    background:#111923;
    border:1px solid #26384e;
    border-radius:16px;
    padding:14px;
}

.label{
    color:#8195ab;
    font-size:12px;
}

.value{
    font-size:26px;
    font-weight:900;
    margin-top:5px;
}

h2{
    margin-top:28px;
    margin-bottom:4px;
}

.card{
    background:#111923;
    border:1px solid #26384e;
    border-radius:18px;
    padding:17px;
    margin:12px 0;
}

.symbol{
    font-size:27px;
    font-weight:950;
}

.call{
    color:#49df88;
}

.put{
    color:#ff6d77;
}

.score{
    font-size:23px;
    font-weight:900;
}

.line{
    margin-top:7px;
    line-height:1.5;
}

.muted{
    color:#90a3b8;
}

.empty{
    padding:35px 18px;
    text-align:center;
    background:#111923;
    border:1px solid #26384e;
    border-radius:18px;
    color:#94a8bf;
}

.footer{
    margin-top:35px;
    text-align:center;
    color:#72869d;
    font-size:12px;
    line-height:1.7;
}

</style>

</head>

<body>

<div class="container">

<h1>
AI Trade Setup Scanner
</h1>

<div class="subtitle">
CALL / PUT candidates ranked by setup quality.
The active liquidity pool is cached so the scanner can
recheck setups faster.
</div>

<div class="badge">
{{ state.status }}
</div>


<div class="stats">

<div class="stat">
<div class="label">
QUALIFIED
</div>
<div class="value">
{{ state.watchlist_count }}
</div>
</div>

<div class="stat">
<div class="label">
NEAR MISSES
</div>
<div class="value">
{{ state.near_miss_count }}
</div>
</div>

<div class="stat">
<div class="label">
LIQUID
</div>
<div class="value">
{{ state.liquid_count }}
</div>
</div>

<div class="stat">
<div class="label">
UNIVERSE
</div>
<div class="value">
{{ state.universe_count }}
</div>
</div>

</div>


<h2>
Qualified Setups
</h2>

<div class="subtitle">
Score {{ state.qualification_threshold }} or higher.
These are the stocks exposed through the trading-bot watchlist API.
</div>


{% if state.watchlist %}

{% for x in state.watchlist %}

<div class="card">

<div>
<span class="symbol">
{{ x.symbol }}
</span>

<span class="symbol {{ 'call' if x.direction == 'CALL' else 'put' }}">
{{ x.direction }}
</span>

<span class="score">
{{ x.score }}/100
</span>
</div>

<div class="line">
{{ x.status }}
· price {{ x.price }}
· trigger <b>{{ x.trigger }}</b>
· target {{ x.target }}
</div>

<div class="line muted">
S {{ x.support }}
· R {{ x.resistance }}
· VWAP {{ x.vwap }}
· RVOL {{ x.rvol }}
· touches {{ x.touches }}
</div>

</div>

{% endfor %}

{% else %}

<div class="empty">
No qualified setups right now.
</div>

{% endif %}


<h2>
Top Near Misses
</h2>

<div class="subtitle">
WATCH ONLY — below qualification threshold and not sent to the trading bot.
</div>


{% if state.near_misses %}

{% for x in state.near_misses %}

<div class="card">

<div>
<span class="symbol">
{{ x.symbol }}
</span>

<span class="symbol {{ 'call' if x.direction == 'CALL' else 'put' }}">
{{ x.direction }}
</span>

<span class="score">
{{ x.score }}/100
</span>
</div>

<div class="line">
{{ x.status }}
· price {{ x.price }}
· trigger {{ x.trigger }}
</div>

</div>

{% endfor %}

{% else %}

<div class="empty">
No near-miss candidates available.
</div>

{% endif %}


<div class="footer">

Last scan:
{{ state.last_scan }}

<br>

Last scan duration:
{{ state.scan_duration_seconds }} seconds

<br>

Active stocks:
{{ state.liquid_count }}

<br>

Full market refresh:
every {{ universe_minutes }} minutes

<br>

Liquidity pool refresh:
every {{ liquidity_minutes }} minutes

<br>

Active setup scan:
every {{ scan_minutes }} minutes

<br>

Website refresh:
every 20 seconds

</div>

</div>

</body>
</html>
"""


# ============================================================
# HOME
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

        scan_minutes=round(
            SCAN_SECONDS
            /
            60,
            1
        ),

        liquidity_minutes=round(
            LIQUIDITY_REFRESH_SECONDS
            /
            60,
            1
        ),

        universe_minutes=round(
            UNIVERSE_REFRESH_SECONDS
            /
            60,
            1
        ),
    )


# ============================================================
# WATCHLIST API FOR AUTO TRADER
# ============================================================

@app.get("/api/watchlist")
def api_watchlist():

    with lock:

        return jsonify({
            "status":
                STATE[
                    "status"
                ],

            "last_scan":
                STATE[
                    "last_scan"
                ],

            "qualification_threshold":
                MIN_SETUP_SCORE,

            "watchlist_count":
                STATE[
                    "watchlist_count"
                ],

            "watchlist":
                STATE[
                    "watchlist"
                ],
        })


# ============================================================
# FULL SCANNER API
# ============================================================

@app.get("/api/state")
def api_state():

    with lock:

        return jsonify(
            STATE
        )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    with lock:

        return jsonify({
            "ok":
                STATE[
                    "status"
                ]
                !=
                "ERROR",

            "status":
                STATE[
                    "status"
                ],

            "last_scan":
                STATE[
                    "last_scan"
                ],

            "scan_duration_seconds":
                STATE[
                    "scan_duration_seconds"
                ],

            "universe_count":
                STATE[
                    "universe_count"
                ],

            "liquid_count":
                STATE[
                    "liquid_count"
                ],

            "qualified":
                STATE[
                    "watchlist_count"
                ],
        })


# ============================================================
# MANUAL CACHE REFRESH
# ============================================================

@app.get("/api/refresh-liquidity")
def refresh_liquidity():

    CACHE[
        "liquidity_updated"
    ] = 0

    return jsonify({
        "ok": True,
        "message":
            "Liquidity refresh queued for next scan."
    })


@app.get("/api/refresh-universe")
def refresh_universe():

    CACHE[
        "universe_updated"
    ] = 0

    CACHE[
        "liquidity_updated"
    ] = 0

    return jsonify({
        "ok": True,
        "message":
            "Full universe refresh queued for next scan."
    })


# ============================================================
# START
# ============================================================

def start_scanner():

    threading.Thread(
        target=loop,
        daemon=True,
    ).start()


if __name__ == "__main__":

    start_scanner()

    app.run(
        host="0.0.0.0",

        port=int(
            os.getenv(
                "PORT",
                "10000"
            )
        ),
    )

else:

    start_scanner()
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

LIQUIDITY_REFRESH_SECONDS = int(
    os.getenv(
        "LIQUIDITY_REFRESH_SECONDS",
        "1800"
    )
)

UNIVERSE_REFRESH_SECONDS = int(
    os.getenv(
        "UNIVERSE_REFRESH_SECONDS",
        "21600"
    )
)

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
# CACHED LIQUIDITY
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
        "symbols":
            ",".join(
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
        df[
            "timestamp"
        ],
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
            df[
                column
            ],
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
                df[
                    "high"
                ]
                -
                df[
                    "low"
                ]
            ).abs(),

            (
                df[
                    "high"
                ]
                -
                previous_close
            ).abs(),

            (
                df[
                    "low"
                ]
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
        df[
            "high"
        ]
        +
        df[
            "low"
        ]
        +
        df[
            "close"
        ]
    ) / 3

    cumulative_volume = (
        df[
            "volume"
        ]
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
            df[
                "volume"
            ]
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
        df[
            "volume"
        ]
        .rolling(
            20,
            min_periods=5
        )
        .mean()
    )

    return df


# ============================================================
# CLOSED 4-MINUTE CANDLES
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
                clusters[
                    -1
                ]
            )
            /
            len(
                clusters[
                    -1
                ]
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

            clusters[
                -1
            ].append(
                value
            )

    return [
        (
            sum(
                cluster
            )
            /
            len(
                cluster
            ),

            len(
                cluster
            ),
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
        len(
            window
        ) - 2
    ):

        low = float(
            window[
                "low"
            ].iloc[
                i
            ]
        )

        high = float(
            window[
                "high"
            ].iloc[
                i
            ]
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
        item[
            0
        ]
    )

    resistance = min(
        above,
        default=(
            None,
            0
        ),
        key=lambda item:
        item[
            0
        ]
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
# ANALYZE
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

    if len(
        df
    ) < 35:

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
        row[
            "atr"
        ]
        if pd.notna(
            row[
                "atr"
            ]
        )
        else
        max(
            price * 0.003,
            0.05
        )
    )

    vwap = float(
        row[
            "vwap"
        ]
        if pd.notna(
            row[
                "vwap"
            ]
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

        support_level = (
            pm_low
        )

        support_touches = 3


    if (
        previous_day_low
        is not None
        and
        previous_day_low < price
        and
        (
            support_level is None
            or
            previous_day_low > support_level
        )
    ):

        support_level = (
            previous_day_low
        )

        support_touches = max(
            support_touches,
            2
        )


    if (
        pm_high is not None
        and
        pm_high > price
        and
        (
            resistance_level is None
            or
            pm_high < resistance_level
        )
    ):

        resistance_level = (
            pm_high
        )

        resistance_touches = 3


    if (
        previous_day_high
        is not None
        and
        previous_day_high > price
        and
        (
            resistance_level is None
            or
            previous_day_high < resistance_level
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
        row[
            "ema5"
        ]
        >
        row[
            "ema9"
        ]
        >
        row[
            "ema30"
        ]
    )

    bear = (
        row[
            "ema5"
        ]
        <
        row[
            "ema9"
        ]
        <
        row[
            "ema30"
        ]
    )

    momentum = (
        price
        -
        float(
            today_df[
                "close"
            ].iloc[
                -3
            ]
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
                possible_targets[
                    0
                ]
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
                possible_targets[
                    0
                ]
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
        abs(
            momentum
        )
        /
        1.2,
        1
    ) * 10


    score += (
        proximity
        *
        20
    )


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
    # STATUS
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
                if support_level is not None
                else None
            ),

        "resistance":
            (
                round(
                    resistance_level,
                    4
                )
                if resistance_level is not None
                else None
            ),

        "target":
            round(
                target,
                4
            ),

        "ema5":
            round(
                float(
                    row[
                        "ema5"
                    ]
                ),
                4
            ),

        "ema9":
            round(
                float(
                    row[
                        "ema9"
                    ]
                ),
                4
            ),

        "ema30":
            round(
                float(
                    row[
                        "ema30"
                    ]
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
# RUN SCAN
# ============================================================

def run_scan():

    started = time.time()

    with lock:

        STATE.update(
            status="SCANNING",
            error=None
        )

    symbols = cached_universe()

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
        len(
            live
        )
    )

    all_results = []

    total_batches = math.ceil(
        len(
            live
        )
        /
        BAR_BATCH
    )

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
            len(
                batch
            )
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
    # SORT
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

    for rank, item in enumerate(
        watchlist,
        start=1
    ):

        item[
            "rank"
        ] = rank

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

    for rank, item in enumerate(
        near_misses,
        start=1
    ):

        item[
            "rank"
        ] = rank

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
        len(
            live
        ),
        len(
            all_results
        ),
        len(
            watchlist
        ),
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
# MOCKUP-STYLE WEBSITE
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
content="#070b11"
>

<title>
90% Scanner Watchlist
</title>


<style>

*{
    box-sizing:border-box;
}

body{
    margin:0;
    background:
        radial-gradient(
            circle at top,
            #111c2a 0%,
            #080d14 36%,
            #05080d 100%
        );
    color:#f5f7fa;
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        Arial,
        sans-serif;
    min-height:100vh;
}

.container{
    max-width:1100px;
    margin:auto;
    padding:
        18px
        13px
        50px;
}

.header{
    display:flex;
    justify-content:space-between;
    align-items:flex-start;
    gap:10px;
}

.title{
    font-size:28px;
    font-weight:950;
    letter-spacing:-1px;
}

.subtitle{
    margin-top:5px;
    color:#8398af;
    font-size:12px;
    line-height:1.45;
}

.ready{
    flex:0 0 auto;
    background:#10271d;
    color:#55df8e;
    border:1px solid #285c42;
    border-radius:999px;
    padding:7px 11px;
    font-size:11px;
    font-weight:900;
}

.scanning{
    color:#ffd96a;
    background:#332a0f;
    border-color:#705d1d;
}

.error{
    color:#ff8189;
    background:#351418;
    border-color:#74313a;
}

.stats{
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:7px;
    margin:18px 0;
}

.stat{
    background:#101822;
    border:1px solid #213147;
    border-radius:14px;
    padding:11px;
    min-width:0;
}

.stat-label{
    color:#8296ad;
    font-size:9px;
    font-weight:800;
    letter-spacing:.5px;
}

.stat-value{
    margin-top:5px;
    font-size:21px;
    font-weight:950;
}

.section{
    margin-top:24px;
}

.section-title{
    font-size:21px;
    font-weight:950;
    margin-bottom:4px;
}

.section-sub{
    color:#8195ab;
    font-size:11px;
    line-height:1.4;
    margin-bottom:12px;
}

.table-wrap{
    width:100%;
    overflow-x:auto;
    border:
        1px solid
        #23354b;
    border-radius:16px;
    background:#0c121b;
}

table{
    width:100%;
    min-width:950px;
    border-collapse:collapse;
}

th{
    text-align:left;
    padding:11px 10px;
    color:#7f94ab;
    font-size:9px;
    letter-spacing:.55px;
    background:#101923;
    border-bottom:1px solid #26384e;
    white-space:nowrap;
}

td{
    padding:13px 10px;
    font-size:12px;
    border-bottom:1px solid #1b2939;
    white-space:nowrap;
    vertical-align:middle;
}

tr:last-child td{
    border-bottom:none;
}

.rank{
    color:#7e91a6;
    font-weight:800;
}

.symbol{
    font-size:16px;
    font-weight:950;
}

.badge{
    display:inline-block;
    border-radius:999px;
    padding:5px 8px;
    font-size:10px;
    font-weight:950;
}

.call{
    color:#55e791;
    background:#123120;
    border:1px solid #296042;
}

.put{
    color:#ff747d;
    background:#35171b;
    border:1px solid #753039;
}

.score{
    font-size:15px;
    font-weight:950;
}

.score-high{
    color:#55e791;
}

.score-mid{
    color:#f4d76a;
}

.status{
    font-size:10px;
    font-weight:900;
    border-radius:999px;
    padding:5px 8px;
    display:inline-block;
}

.waiting{
    color:#f4d76a;
    background:#332a0f;
    border:1px solid #6c5a1c;
}

.confirmed{
    color:#54e38e;
    background:#123120;
    border:1px solid #296042;
}

.level{
    color:#e7edf4;
    font-weight:800;
}

.muted{
    color:#8498af;
}

.cards{
    display:none;
}

.card{
    background:
        linear-gradient(
            180deg,
            #111b27,
            #0b121a
        );
    border:1px solid #26384e;
    border-radius:17px;
    padding:14px;
    margin-bottom:11px;
}

.card.call-card{
    border-left:4px solid #4dde88;
}

.card.put-card{
    border-left:4px solid #ff6873;
}

.card-top{
    display:flex;
    justify-content:space-between;
    gap:10px;
    align-items:flex-start;
}

.card-symbol{
    font-size:25px;
    font-weight:950;
}

.card-score{
    font-size:21px;
    font-weight:950;
}

.info-grid{
    display:grid;
    grid-template-columns:repeat(2,1fr);
    gap:9px;
    margin-top:13px;
}

.info{
    background:#0b121a;
    border:1px solid #1f2e40;
    border-radius:10px;
    padding:9px;
}

.info-label{
    color:#74899f;
    font-size:8px;
    font-weight:800;
    letter-spacing:.5px;
}

.info-value{
    margin-top:4px;
    font-size:13px;
    font-weight:850;
}

.empty{
    background:#101822;
    border:1px solid #26384e;
    border-radius:17px;
    padding:35px 17px;
    text-align:center;
    color:#8fa3b9;
}

.legend{
    margin-top:25px;
    background:#0d141d;
    border:1px solid #23354a;
    border-radius:15px;
    padding:13px;
    color:#879bb1;
    font-size:11px;
    line-height:1.6;
}

.footer{
    margin-top:30px;
    text-align:center;
    color:#6e8399;
    font-size:11px;
    line-height:1.7;
}

@media(max-width:700px){

    .container{
        padding-left:11px;
        padding-right:11px;
    }

    .title{
        font-size:25px;
    }

    .stats{
        grid-template-columns:repeat(4,1fr);
        gap:5px;
    }

    .stat{
        padding:9px 6px;
    }

    .stat-value{
        font-size:18px;
    }

    .stat-label{
        font-size:7px;
    }

    .table-wrap{
        display:none;
    }

    .cards{
        display:block;
    }
}

</style>

</head>


<body>

<div class="container">


<div class="header">

<div>

<div class="title">
90% Scanner Watchlist
</div>

<div class="subtitle">
4-minute CALL / PUT setup scanner • ranked by setup quality
<br>
Qualified setups are sent to the Alpaca confirmation bot.
</div>

</div>


<div
class="
ready
{% if state.status == 'SCANNING' %}
scanning
{% elif state.status == 'ERROR' %}
error
{% endif %}
"
>

{{ state.status }}

</div>

</div>


<div class="stats">

<div class="stat">

<div class="stat-label">
QUALIFIED
</div>

<div class="stat-value">
{{ state.watchlist_count }}
</div>

</div>


<div class="stat">

<div class="stat-label">
NEAR MISS
</div>

<div class="stat-value">
{{ state.near_miss_count }}
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


<div class="section">

<div class="section-title">
Qualified Setups
</div>

<div class="section-sub">
Score {{ state.qualification_threshold }}+ • Highest score first • Bot still waits for confirmation before entry.
</div>


{% if state.watchlist %}


<div class="table-wrap">

<table>

<thead>

<tr>

<th>#</th>
<th>SYMBOL</th>
<th>DIRECTION</th>
<th>SCORE</th>
<th>PRICE</th>
<th>STATUS</th>
<th>TRIGGER</th>
<th>TARGET</th>
<th>SUPPORT</th>
<th>RESISTANCE</th>
<th>VWAP</th>
<th>RVOL</th>

</tr>

</thead>


<tbody>

{% for x in state.watchlist %}

<tr>

<td class="rank">
{{ x.rank }}
</td>

<td class="symbol">
{{ x.symbol }}
</td>

<td>

<span
class="
badge
{{ 'call' if x.direction == 'CALL' else 'put' }}
"
>

{{ x.direction }}

</span>

</td>

<td
class="
score
{{ 'score-high' if x.score >= 90 else 'score-mid' }}
"
>
{{ x.score }}
</td>

<td>
${{ x.price }}
</td>

<td>

<span
class="
status
{{ 'confirmed' if x.status == 'BREAK_CONFIRMED' else 'waiting' }}
"
>

{{ x.status }}

</span>

</td>

<td class="level">
{{ x.trigger }}
</td>

<td class="level">
{{ x.target }}
</td>

<td>
{{ x.support }}
</td>

<td>
{{ x.resistance }}
</td>

<td>
{{ x.vwap }}
</td>

<td>
{{ x.rvol }}
</td>

</tr>

{% endfor %}

</tbody>

</table>

</div>


<div class="cards">

{% for x in state.watchlist %}

<div
class="
card
{{ 'call-card' if x.direction == 'CALL' else 'put-card' }}
"
>

<div class="card-top">

<div>

<div class="card-symbol">
#{{ x.rank }} {{ x.symbol }}
</div>

<div style="margin-top:7px;">

<span
class="
badge
{{ 'call' if x.direction == 'CALL' else 'put' }}
"
>

{{ x.direction }}

</span>

<span
class="
status
{{ 'confirmed' if x.status == 'BREAK_CONFIRMED' else 'waiting' }}
"
style="margin-left:5px;"
>

{{ x.status }}

</span>

</div>

</div>


<div
class="
card-score
{{ 'score-high' if x.score >= 90 else 'score-mid' }}
"
>

{{ x.score }}

</div>

</div>


<div class="info-grid">

<div class="info">
<div class="info-label">PRICE</div>
<div class="info-value">${{ x.price }}</div>
</div>

<div class="info">
<div class="info-label">TRIGGER</div>
<div class="info-value">{{ x.trigger }}</div>
</div>

<div class="info">
<div class="info-label">TARGET</div>
<div class="info-value">{{ x.target }}</div>
</div>

<div class="info">
<div class="info-label">RVOL</div>
<div class="info-value">{{ x.rvol }}</div>
</div>

<div class="info">
<div class="info-label">SUPPORT</div>
<div class="info-value">{{ x.support }}</div>
</div>

<div class="info">
<div class="info-label">RESISTANCE</div>
<div class="info-value">{{ x.resistance }}</div>
</div>

<div class="info">
<div class="info-label">VWAP</div>
<div class="info-value">{{ x.vwap }}</div>
</div>

<div class="info">
<div class="info-label">TOUCHES</div>
<div class="info-value">{{ x.touches }}</div>
</div>

</div>

</div>

{% endfor %}

</div>


{% else %}

<div class="empty">

No qualified setups right now.

<br><br>

The scanner will populate this section automatically when stocks score
{{ state.qualification_threshold }} or higher.

</div>

{% endif %}

</div>



<div class="section">

<div class="section-title">
Top Near Misses
</div>

<div class="section-sub">
Below {{ state.qualification_threshold }} • Watch only • Not sent to the auto-trader.
</div>


{% if state.near_misses %}

<div class="table-wrap">

<table>

<thead>

<tr>

<th>#</th>
<th>SYMBOL</th>
<th>DIRECTION</th>
<th>SCORE</th>
<th>PRICE</th>
<th>STATUS</th>
<th>TRIGGER</th>
<th>SUPPORT</th>
<th>RESISTANCE</th>
<th>RVOL</th>

</tr>

</thead>

<tbody>

{% for x in state.near_misses %}

<tr>

<td class="rank">
{{ x.rank }}
</td>

<td class="symbol">
{{ x.symbol }}
</td>

<td>

<span
class="
badge
{{ 'call' if x.direction == 'CALL' else 'put' }}
"
>

{{ x.direction }}

</span>

</td>

<td class="score score-mid">
{{ x.score }}
</td>

<td>
${{ x.price }}
</td>

<td>

<span class="status waiting">
{{ x.status }}
</span>

</td>

<td>
{{ x.trigger }}
</td>

<td>
{{ x.support }}
</td>

<td>
{{ x.resistance }}
</td>

<td>
{{ x.rvol }}
</td>

</tr>

{% endfor %}

</tbody>

</table>

</div>


<div class="cards">

{% for x in state.near_misses %}

<div
class="
card
{{ 'call-card' if x.direction == 'CALL' else 'put-card' }}
"
>

<div class="card-top">

<div>

<div class="card-symbol">
{{ x.symbol }}
</div>

<div style="margin-top:7px;">

<span
class="
badge
{{ 'call' if x.direction == 'CALL' else 'put' }}
"
>

{{ x.direction }}

</span>

</div>

</div>

<div class="card-score score-mid">
{{ x.score }}
</div>

</div>


<div class="info-grid">

<div class="info">
<div class="info-label">PRICE</div>
<div class="info-value">${{ x.price }}</div>
</div>

<div class="info">
<div class="info-label">TRIGGER</div>
<div class="info-value">{{ x.trigger }}</div>
</div>

<div class="info">
<div class="info-label">SUPPORT</div>
<div class="info-value">{{ x.support }}</div>
</div>

<div class="info">
<div class="info-label">RESISTANCE</div>
<div class="info-value">{{ x.resistance }}</div>
</div>

</div>

</div>

{% endfor %}

</div>


{% else %}

<div class="empty">
No near-miss setups right now.
</div>

{% endif %}

</div>



<div class="legend">

<b>S</b> = Support &nbsp; • &nbsp;
<b>R</b> = Resistance &nbsp; • &nbsp;
<b>RVOL</b> = Relative Volume

<br>

<b>WAITING_FOR_BREAK</b> means the stock is qualified but has not completed the required level break yet.

<br>

<b>BREAK_CONFIRMED</b> means the scanner detected the completed 4-minute break. The Alpaca bot still performs its separate confirmation and AI review before entering.

</div>


<div class="footer">

Last scan:
{{ state.last_scan }}

<br>

Scan duration:
{{ state.scan_duration_seconds }} seconds

<br>

Active setup scan:
every {{ scan_minutes }} minutes

<br>

Liquidity pool refresh:
every {{ liquidity_minutes }} minutes

<br>

Full universe refresh:
every {{ universe_minutes }} minutes

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
# WATCHLIST API FOR ALPACA BOT
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
# FULL STATE API
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
# MANUAL REFRESH
# ============================================================

@app.get("/api/refresh-liquidity")
def refresh_liquidity():

    CACHE[
        "liquidity_updated"
    ] = 0

    return jsonify({
        "ok":
            True,

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
        "ok":
            True,

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
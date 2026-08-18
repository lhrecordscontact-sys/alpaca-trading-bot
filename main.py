import os
import math
import time
import threading
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests
from flask import Flask, jsonify, request


# ============================================================
# ALPACA 0DTE PAPER-TRADING BOT
# ============================================================
# PAPER TRADING ONLY
#
# Strategy:
# - Scans stocks with 0DTE options available today
# - Uses 4-minute candles
# - EMA 5 / EMA 9 / EMA 30 / VWAP
# - CALL on bullish confirmation
# - PUT on bearish confirmation
# - Take 50% off at +30%
# - Remaining contracts become runners
# - Runner trails 15% from highest option price after TP
# - 20% hard option-premium stop before TP
# - Strategy invalidation exit through EMA 9
# ============================================================

app = Flask(__name__)

ET = ZoneInfo("America/New_York")

ALPACA_API_KEY = os.environ.get(
    "ALPACA_API_KEY",
    "",
).strip()

ALPACA_SECRET_KEY = os.environ.get(
    "ALPACA_SECRET_KEY",
    "",
).strip()

# ============================================================
# ALPACA URLS
# ============================================================

# HARD-WIRED TO PAPER TRADING
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    "Content-Type": "application/json",
}


# ============================================================
# BOT SETTINGS
# ============================================================

AUTO_TRADE = (
    os.environ.get(
        "AUTO_TRADE",
        "true",
    )
    .strip()
    .lower()
    == "true"
)

RUN_BOT_LOOP = (
    os.environ.get(
        "RUN_BOT_LOOP",
        "true",
    )
    .strip()
    .lower()
    == "true"
)

BOT_LOOP_SECONDS = 20
UNIVERSE_REFRESH_SECONDS = 600
FULL_SCAN_REFRESH_SECONDS = 60


# ============================================================
# SCANNER SETTINGS
# ============================================================

MIN_STOCK_PRICE = 5.00
MIN_DAILY_VOLUME = 1_000_000
MAX_SIGNAL_CANDIDATES = 40


# ============================================================
# STRATEGY SETTINGS
# ============================================================

BAR_TIMEFRAME = "4Min"
BAR_LOOKBACK_MINUTES = 420
MIN_BARS_REQUIRED = 35


# ============================================================
# ACCOUNT RISK
# ============================================================

RISK_PER_TRADE_PERCENT = 0.015
MAX_POSITION_VALUE_PERCENT = 0.15
MAX_DAILY_LOSS_PERCENT = 0.03
MAX_OPEN_POSITIONS = 3
MAX_CONTRACTS_PER_TRADE = 10
MIN_CONTRACTS_FOR_RUNNER = 2


# ============================================================
# TRADE MANAGEMENT
# ============================================================

HARD_STOP_PERCENT = 0.20

TAKE_PROFIT_PERCENT = 0.30
TAKE_PROFIT_FRACTION = 0.50

RUNNER_TRAIL_PERCENT = 0.15


# ============================================================
# OPTION SELECTION
# ============================================================

STRIKE_SEARCH_PERCENT = 0.06
MAX_OPTION_SPREAD_PERCENT = 0.25
MIN_OPTION_MID_PRICE = 0.10


# ============================================================
# TIME SETTINGS
# ============================================================

LAST_ENTRY_HOUR_ET = 14
LAST_ENTRY_MINUTE_ET = 45

FORCE_EXIT_HOUR_ET = 15
FORCE_EXIT_MINUTE_ET = 15

ENTRY_COOLDOWN_MINUTES = 30


# ============================================================
# STATE
# ============================================================

STATE_LOCK = threading.Lock()
BOT_THREAD_LOCK = threading.Lock()

BOT_THREAD_STARTED = False

TRADE_STATE = {}
LAST_ENTRY_TIME = {}

UNIVERSE_CACHE = {
    "symbols": [],
    "updated_at": 0.0,
}

SCAN_CACHE = {
    "candidates": [],
    "updated_at": 0.0,
}

LAST_LOOP_ERROR = None
LAST_LOOP_TIME = None


# ============================================================
# BASIC HELPERS
# ============================================================

def now_et():
    return datetime.now(ET)


def today_et():
    return now_et().date()


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default

        return float(value)

    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        if value is None:
            return default

        return int(float(value))

    except (TypeError, ValueError):
        return default


def require_keys():
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise RuntimeError(
            "Missing ALPACA_API_KEY or ALPACA_SECRET_KEY"
        )


def chunk_list(items, size):
    for i in range(
        0,
        len(items),
        size,
    ):
        yield items[i:i + size]


def parse_timestamp(value):
    if not value:
        return None

    text = str(value).replace(
        "Z",
        "+00:00",
    )

    try:
        return datetime.fromisoformat(text)

    except ValueError:
        return None


def entry_window_open():
    current = now_et()

    if current.weekday() >= 5:
        return False

    market_open = (
        9 * 60
        + 30
    )

    entry_cutoff = (
        LAST_ENTRY_HOUR_ET * 60
        + LAST_ENTRY_MINUTE_ET
    )

    current_minutes = (
        current.hour * 60
        + current.minute
    )

    return (
        market_open
        <= current_minutes
        < entry_cutoff
    )


def force_exit_time():
    current = now_et()

    cutoff = (
        FORCE_EXIT_HOUR_ET * 60
        + FORCE_EXIT_MINUTE_ET
    )

    current_minutes = (
        current.hour * 60
        + current.minute
    )

    return current_minutes >= cutoff


# ============================================================
# HTTP HELPERS
# ============================================================

def alpaca_request(
    method,
    url,
    params=None,
    payload=None,
    timeout=30,
):
    require_keys()

    response = requests.request(
        method,
        url,
        headers=HEADERS,
        params=params,
        json=payload,
        timeout=timeout,
    )

    try:
        data = response.json()

    except Exception:
        data = {
            "raw": response.text
        }

    if response.status_code >= 400:
        raise RuntimeError(
            f"Alpaca HTTP "
            f"{response.status_code}: "
            f"{data}"
        )

    return data


def trading_get(
    path,
    params=None,
):
    return alpaca_request(
        "GET",
        f"{ALPACA_BASE_URL}{path}",
        params=params,
    )


def trading_post(
    path,
    payload,
):
    return alpaca_request(
        "POST",
        f"{ALPACA_BASE_URL}{path}",
        payload=payload,
    )


def market_get(
    path,
    params=None,
):
    return alpaca_request(
        "GET",
        f"{DATA_BASE_URL}{path}",
        params=params,
    )


# ============================================================
# ACCOUNT
# ============================================================

def get_account():
    return trading_get(
        "/v2/account"
    )


def get_positions():
    data = trading_get(
        "/v2/positions"
    )

    if isinstance(
        data,
        list,
    ):
        return data

    return []


def get_option_positions():
    positions = []

    for position in get_positions():

        if (
            position.get("asset_class")
            == "us_option"
        ):
            positions.append(
                position
            )

    return positions


def account_equity():
    account = get_account()

    return safe_float(
        account.get("equity"),
        0.0,
    )


def account_buying_power():
    account = get_account()

    return safe_float(
        account.get(
            "options_buying_power"
        )
        or account.get(
            "buying_power"
        ),
        0.0,
    )


def daily_pnl():
    account = get_account()

    equity = safe_float(
        account.get("equity"),
        0.0,
    )

    last_equity = safe_float(
        account.get("last_equity"),
        0.0,
    )

    return (
        equity
        - last_equity
    )


def daily_loss_limit_reached():
    account = get_account()

    equity = safe_float(
        account.get("equity"),
        0.0,
    )

    last_equity = safe_float(
        account.get("last_equity"),
        0.0,
    )

    if last_equity <= 0:
        return False

    maximum_loss = (
        last_equity
        * MAX_DAILY_LOSS_PERCENT
    )

    daily_loss = (
        equity
        - last_equity
    )

    return (
        daily_loss
        <= -maximum_loss
    )


# ============================================================
# ORDER EXECUTION
# ============================================================

def submit_market_order(
    symbol,
    qty,
    side,
):
    payload = {
        "symbol": symbol,
        "qty": str(
            int(qty)
        ),
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }

    return trading_post(
        "/v2/orders",
        payload,
    )


def wait_for_order_fill(
    order_id,
    timeout_seconds=20,
):
    deadline = (
        time.time()
        + timeout_seconds
    )

    while time.time() < deadline:

        order = trading_get(
            f"/v2/orders/{order_id}"
        )

        status = str(
            order.get(
                "status",
                "",
            )
        ).lower()

        if status == "filled":
            return order

        if status in {
            "canceled",
            "expired",
            "rejected",
            "suspended",
        }:
            return order

        time.sleep(1)

    return trading_get(
        f"/v2/orders/{order_id}"
    )


def close_option_position(
    symbol,
    qty,
):
    if qty <= 0:
        return None

    return submit_market_order(
        symbol,
        qty,
        "sell",
    )


# ============================================================
# 0DTE UNIVERSE
# ============================================================

def get_0dte_contracts_all():
    expiration = (
        today_et().isoformat()
    )

    contracts = []

    page_token = None

    while True:

        params = {
            "status": "active",
            "expiration_date": expiration,
            "limit": 10000,
        }

        if page_token:
            params[
                "page_token"
            ] = page_token

        data = trading_get(
            "/v2/options/contracts",
            params=params,
        )

        page = (
            data.get(
                "option_contracts"
            )
            or data.get(
                "contracts"
            )
            or []
        )

        if isinstance(
            page,
            list,
        ):
            contracts.extend(
                page
            )

        page_token = data.get(
            "next_page_token"
        )

        if not page_token:
            break

    return contracts


def refresh_0dte_universe(
    force=False,
):
    now_timestamp = time.time()

    with STATE_LOCK:

        cached = list(
            UNIVERSE_CACHE[
                "symbols"
            ]
        )

        updated_at = (
            UNIVERSE_CACHE[
                "updated_at"
            ]
        )

    if (
        not force
        and cached
        and (
            now_timestamp
            - updated_at
        )
        < UNIVERSE_REFRESH_SECONDS
    ):
        return cached

    contracts = (
        get_0dte_contracts_all()
    )

    symbols = set()

    for contract in contracts:

        underlying = str(
            contract.get(
                "underlying_symbol"
            )
            or ""
        ).upper().strip()

        if underlying:
            symbols.add(
                underlying
            )

    symbols = sorted(
        symbols
    )

    with STATE_LOCK:

        UNIVERSE_CACHE[
            "symbols"
        ] = symbols

        UNIVERSE_CACHE[
            "updated_at"
        ] = now_timestamp

    print(
        f"[UNIVERSE] "
        f"{len(symbols)} "
        f"stocks have 0DTE "
        f"contracts today.",
        flush=True,
    )

    return symbols


# ============================================================
# STOCK SNAPSHOTS
# ============================================================

def get_stock_snapshots(
    symbols,
):
    results = {}

    for group in chunk_list(
        symbols,
        100,
    ):

        data = market_get(
            "/v2/stocks/snapshots",
            params={
                "symbols": ",".join(
                    group
                ),
                "feed": "iex",
            },
        )

        snapshots = (
            data.get(
                "snapshots"
            )
            if isinstance(
                data,
                dict,
            )
            else None
        )

        if isinstance(
            snapshots,
            dict,
        ):
            results.update(
                snapshots
            )

        elif isinstance(
            data,
            dict,
        ):
            results.update(
                data
            )

    return results


def score_snapshot(
    symbol,
    snapshot,
):
    latest_trade = (
        snapshot.get(
            "latestTrade"
        )
        or snapshot.get(
            "latest_trade"
        )
        or {}
    )

    minute_bar = (
        snapshot.get(
            "minuteBar"
        )
        or snapshot.get(
            "minute_bar"
        )
        or {}
    )

    daily_bar = (
        snapshot.get(
            "dailyBar"
        )
        or snapshot.get(
            "daily_bar"
        )
        or {}
    )

    previous_bar = (
        snapshot.get(
            "prevDailyBar"
        )
        or snapshot.get(
            "prev_daily_bar"
        )
        or {}
    )

    price = safe_float(
        latest_trade.get("p")
        or minute_bar.get("c")
        or daily_bar.get("c"),
        0.0,
    )

    volume = safe_float(
        daily_bar.get("v"),
        0.0,
    )

    previous_volume = safe_float(
        previous_bar.get("v"),
        0.0,
    )

    previous_close = safe_float(
        previous_bar.get("c"),
        0.0,
    )

    day_open = safe_float(
        daily_bar.get("o"),
        0.0,
    )

    day_high = safe_float(
        daily_bar.get("h"),
        0.0,
    )

    day_low = safe_float(
        daily_bar.get("l"),
        0.0,
    )

    if (
        price
        < MIN_STOCK_PRICE
    ):
        return None

    if (
        volume
        < MIN_DAILY_VOLUME
    ):
        return None

    percent_change = 0.0

    if previous_close > 0:

        percent_change = (
            (
                price
                - previous_close
            )
            / previous_close
        ) * 100.0

    relative_volume = 0.0

    if previous_volume > 0:

        relative_volume = (
            volume
            / previous_volume
        )

    range_position = 0.5

    if day_high > day_low:

        range_position = (
            price
            - day_low
        ) / (
            day_high
            - day_low
        )

    score = 0.0

    score += min(
        30.0,
        math.log10(
            max(
                volume,
                1,
            )
        )
        * 4.0,
    )

    score += min(
        30.0,
        abs(
            percent_change
        )
        * 7.5,
    )

    score += min(
        20.0,
        relative_volume
        * 20.0,
    )

    edge_strength = (
        abs(
            range_position
            - 0.5
        )
        * 2.0
    )

    score += min(
        20.0,
        edge_strength
        * 20.0,
    )

    direction = "neutral"

    if day_open > 0:

        if price > day_open:
            direction = "bullish"

        elif price < day_open:
            direction = "bearish"

    return {
        "symbol": symbol,
        "price": round(
            price,
            4,
        ),
        "volume": int(
            volume
        ),
        "relative_volume": round(
            relative_volume,
            3,
        ),
        "percent_change": round(
            percent_change,
            3,
        ),
        "range_position": round(
            range_position,
            3,
        ),
        "direction": direction,
        "score": round(
            score,
            2,
        ),
    }


def refresh_scanner(
    force=False,
):
    now_timestamp = time.time()

    with STATE_LOCK:

        cached = list(
            SCAN_CACHE[
                "candidates"
            ]
        )

        updated_at = (
            SCAN_CACHE[
                "updated_at"
            ]
        )

    if (
        not force
        and cached
        and (
            now_timestamp
            - updated_at
        )
        < FULL_SCAN_REFRESH_SECONDS
    ):
        return cached

    universe = (
        refresh_0dte_universe()
    )

    if not universe:
        return []

    snapshots = (
        get_stock_snapshots(
            universe
        )
    )

    candidates = []

    for symbol in universe:

        result = score_snapshot(
            symbol,
            snapshots.get(
                symbol
            )
            or {},
        )

        if result:

            candidates.append(
                result
            )

    candidates.sort(
        key=lambda item: item[
            "score"
        ],
        reverse=True,
    )

    candidates = (
        candidates[
            :MAX_SIGNAL_CANDIDATES
        ]
    )

    with STATE_LOCK:

        SCAN_CACHE[
            "candidates"
        ] = candidates

        SCAN_CACHE[
            "updated_at"
        ] = now_timestamp

    if candidates:

        print(
            "[SCAN] "
            + ", ".join(
                f'{item["symbol"]}:'
                f'{item["score"]:.1f}'
                for item
                in candidates[:10]
            ),
            flush=True,
        )

    return candidates


# ============================================================
# 4-MINUTE STOCK BARS
# ============================================================

def get_stock_bars(
    symbol,
):
    end_time = (
        datetime.now(
            timezone.utc
        )
    )

    start_time = (
        end_time
        - timedelta(
            minutes=(
                BAR_LOOKBACK_MINUTES
            )
        )
    )

    data = market_get(
        f"/v2/stocks/"
        f"{symbol}/bars",
        params={
            "timeframe": (
                BAR_TIMEFRAME
            ),
            "start": (
                start_time.isoformat()
            ),
            "end": (
                end_time.isoformat()
            ),
            "limit": 200,
            "adjustment": "raw",
            "feed": "iex",
        },
    )

    bars = (
        data.get("bars")
        or []
    )

    cleaned = []

    for bar in bars:

        timestamp = (
            parse_timestamp(
                bar.get("t")
            )
        )

        if timestamp is None:
            continue

        # Ignore a candle that
        # may still be forming.
        if (
            timestamp
            + timedelta(
                minutes=4
            )
            > end_time
        ):
            continue

        cleaned.append(
            {
                "t": timestamp,
                "o": safe_float(
                    bar.get("o"),
                    0.0,
                ),
                "h": safe_float(
                    bar.get("h"),
                    0.0,
                ),
                "l": safe_float(
                    bar.get("l"),
                    0.0,
                ),
                "c": safe_float(
                    bar.get("c"),
                    0.0,
                ),
                "v": safe_float(
                    bar.get("v"),
                    0.0,
                ),
            }
        )

    return cleaned


# ============================================================
# INDICATORS
# ============================================================

def ema(
    values,
    length,
):
    if not values:
        return []

    alpha = (
        2.0
        / (
            length
            + 1.0
        )
    )

    output = [
        values[0]
    ]

    for value in values[1:]:

        output.append(
            (
                value
                * alpha
            )
            + (
                output[-1]
                * (
                    1.0
                    - alpha
                )
            )
        )

    return output


def session_vwap(
    bars,
):
    output = []

    cumulative_pv = 0.0
    cumulative_volume = 0.0

    active_date = None

    for bar in bars:

        bar_date = (
            bar["t"]
            .astimezone(ET)
            .date()
        )

        if (
            bar_date
            != active_date
        ):
            active_date = (
                bar_date
            )

            cumulative_pv = 0.0
            cumulative_volume = 0.0

        typical_price = (
            (
                bar["h"]
                + bar["l"]
                + bar["c"]
            )
            / 3.0
        )

        volume = max(
            bar["v"],
            0.0,
        )

        cumulative_pv += (
            typical_price
            * volume
        )

        cumulative_volume += (
            volume
        )

        if cumulative_volume > 0:

            output.append(
                cumulative_pv
                / cumulative_volume
            )

        else:

            output.append(
                bar["c"]
            )

    return output


def indicator_snapshot(
    bars,
):
    if (
        len(bars)
        < MIN_BARS_REQUIRED
    ):
        return None

    closes = [
        bar["c"]
        for bar in bars
    ]

    ema5 = ema(
        closes,
        5,
    )

    ema9 = ema(
        closes,
        9,
    )

    ema30 = ema(
        closes,
        30,
    )

    vwap = session_vwap(
        bars
    )

    rows = []

    for i, bar in enumerate(
        bars
    ):

        rows.append(
            {
                **bar,
                "ema5": ema5[i],
                "ema9": ema9[i],
                "ema30": ema30[i],
                "vwap": vwap[i],
            }
        )

    return rows


# ============================================================
# SIGNAL DETECTION
# ============================================================

def detect_signal(
    symbol,
):
    bars = get_stock_bars(
        symbol
    )

    rows = indicator_snapshot(
        bars
    )

    if (
        not rows
        or len(rows) < 4
    ):
        return None

    current = rows[-1]
    previous = rows[-2]
    prior = rows[-3]

    bull_stack = (
        current["ema5"]
        > current["ema9"]
        > current["ema30"]
        and current["c"]
        > current["vwap"]
        and current["c"]
        > current["ema30"]
    )

    bear_stack = (
        current["ema5"]
        < current["ema9"]
        < current["ema30"]
        and current["c"]
        < current["vwap"]
        and current["c"]
        < current["ema30"]
    )

    recent_bull_transition = (
        prior["ema5"]
        <= prior["ema9"]
        or prior["c"]
        <= prior["vwap"]
        or prior["c"]
        <= prior["ema30"]
        or previous["ema5"]
        <= previous["ema9"]
        or previous["c"]
        <= previous["vwap"]
        or previous["c"]
        <= previous["ema30"]
    )

    recent_bear_transition = (
        prior["ema5"]
        >= prior["ema9"]
        or prior["c"]
        >= prior["vwap"]
        or prior["c"]
        >= prior["ema30"]
        or previous["ema5"]
        >= previous["ema9"]
        or previous["c"]
        >= previous["vwap"]
        or previous["c"]
        >= previous["ema30"]
    )

    bull_retest_level = max(
        previous["ema9"],
        previous["vwap"],
        previous["ema30"],
    )

    bear_retest_level = min(
        previous["ema9"],
        previous["vwap"],
        previous["ema30"],
    )

    bullish_retest = (
        previous["l"]
        <= (
            bull_retest_level
            * 1.003
        )
        and previous["c"]
        >= previous["ema9"]
    )

    bearish_retest = (
        previous["h"]
        >= (
            bear_retest_level
            * 0.997
        )
        and previous["c"]
        <= previous["ema9"]
    )

    bull_confirmation = (
        current["c"]
        > current["o"]
        and current["c"]
        > previous["h"]
        and current["c"]
        > current["ema5"]
    )

    bear_confirmation = (
        current["c"]
        < current["o"]
        and current["c"]
        < previous["l"]
        and current["c"]
        < current["ema5"]
    )

    call_signal = (
        bull_stack
        and bull_confirmation
        and (
            bullish_retest
            or recent_bull_transition
        )
    )

    put_signal = (
        bear_stack
        and bear_confirmation
        and (
            bearish_retest
            or recent_bear_transition
        )
    )

    if call_signal:
        side = "call"

    elif put_signal:
        side = "put"

    else:
        return None

    return {
        "symbol": symbol,
        "side": side,
        "bar_time": (
            current["t"]
            .isoformat()
        ),
        "underlying_price": (
            current["c"]
        ),
        "ema5": (
            current["ema5"]
        ),
        "ema9": (
            current["ema9"]
        ),
        "ema30": (
            current["ema30"]
        ),
        "vwap": (
            current["vwap"]
        ),
    }


def technical_exit_trigger(
    underlying,
    trade_side,
):
    bars = get_stock_bars(
        underlying
    )

    rows = indicator_snapshot(
        bars
    )

    if not rows:
        return False

    current = rows[-1]

    if trade_side == "call":

        return (
            current["c"]
            < current["ema9"]
        )

    if trade_side == "put":

        return (
            current["c"]
            > current["ema9"]
        )

    return False


# ============================================================
# OPTION CHAIN
# ============================================================

def get_0dte_chain(
    underlying,
    option_type,
    stock_price,
):
    low_strike = (
        stock_price
        * (
            1.0
            - STRIKE_SEARCH_PERCENT
        )
    )

    high_strike = (
        stock_price
        * (
            1.0
            + STRIKE_SEARCH_PERCENT
        )
    )

    data = market_get(
        f"/v1beta1/options/"
        f"snapshots/{underlying}",
        params={
            "expiration_date": (
                today_et()
                .isoformat()
            ),
            "type": option_type,
            "strike_price_gte": round(
                low_strike,
                2,
            ),
            "strike_price_lte": round(
                high_strike,
                2,
            ),
            "limit": 1000,
            "feed": "indicative",
        },
    )

    snapshots = (
        data.get(
            "snapshots"
        )
        or {}
    )

    if isinstance(
        snapshots,
        dict,
    ):
        return snapshots

    return {}


def parse_occ_strike(
    option_symbol,
):
    if (
        not option_symbol
        or len(option_symbol) < 8
    ):
        return None

    raw = (
        option_symbol[-8:]
    )

    if not raw.isdigit():
        return None

    return (
        int(raw)
        / 1000.0
    )


def option_quote_values(
    snapshot,
):
    quote = (
        snapshot.get(
            "latestQuote"
        )
        or snapshot.get(
            "latest_quote"
        )
        or {}
    )

    bid = safe_float(
        quote.get("bp")
        or quote.get(
            "bid_price"
        ),
        0.0,
    )

    ask = safe_float(
        quote.get("ap")
        or quote.get(
            "ask_price"
        ),
        0.0,
    )

    if (
        bid > 0
        and ask > 0
    ):

        mid = (
            bid + ask
        ) / 2.0

    else:

        trade = (
            snapshot.get(
                "latestTrade"
            )
            or snapshot.get(
                "latest_trade"
            )
            or {}
        )

        mid = safe_float(
            trade.get("p")
            or trade.get(
                "price"
            ),
            0.0,
        )

    spread_percent = None

    if (
        mid > 0
        and ask > 0
        and bid > 0
    ):

        spread_percent = (
            ask - bid
        ) / mid

    return (
        bid,
        ask,
        mid,
        spread_percent,
    )


def choose_option_contract(
    underlying,
    option_type,
    stock_price,
):
    chain = get_0dte_chain(
        underlying,
        option_type,
        stock_price,
    )

    choices = []

    for (
        option_symbol,
        snapshot,
    ) in chain.items():

        strike = parse_occ_strike(
            option_symbol
        )

        if strike is None:
            continue

        (
            bid,
            ask,
            mid,
            spread_percent,
        ) = option_quote_values(
            snapshot
        )

        if (
            mid
            < MIN_OPTION_MID_PRICE
        ):
            continue

        if ask <= 0:
            continue

        if (
            spread_percent
            is not None
            and spread_percent
            > MAX_OPTION_SPREAD_PERCENT
        ):
            continue

        distance = abs(
            strike
            - stock_price
        )

        choices.append(
            {
                "symbol": (
                    option_symbol
                ),
                "strike": strike,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread_percent": (
                    spread_percent
                ),
                "distance": (
                    distance
                ),
            }
        )

    if not choices:
        return None

    choices.sort(
        key=lambda item: (
            item["distance"],
            (
                item[
                    "spread_percent"
                ]
                if item[
                    "spread_percent"
                ]
                is not None
                else 999
            ),
        )
    )

    return choices[0]


def get_latest_option_quote(
    option_symbol,
):
    data = market_get(
        "/v1beta1/options/"
        "quotes/latest",
        params={
            "symbols": (
                option_symbol
            ),
            "feed": "indicative",
        },
    )

    quotes = (
        data.get("quotes")
        or {}
    )

    quote = (
        quotes.get(
            option_symbol
        )
        or {}
    )

    bid = safe_float(
        quote.get("bp")
        or quote.get(
            "bid_price"
        ),
        0.0,
    )

    ask = safe_float(
        quote.get("ap")
        or quote.get(
            "ask_price"
        ),
        0.0,
    )

    if (
        bid > 0
        and ask > 0
    ):

        mid = (
            bid + ask
        ) / 2.0

    elif bid > 0:
        mid = bid

    elif ask > 0:
        mid = ask

    else:
        mid = 0.0

    return {
        "bid": bid,
        "ask": ask,
        "mid": mid,
    }


# ============================================================
# RISK / POSITION SIZING
# ============================================================

def open_option_symbols():
    return {
        str(
            position.get(
                "symbol",
                "",
            )
        ).upper()
        for position
        in get_option_positions()
    }


def position_exists_for_underlying(
    underlying,
):
    underlying = (
        underlying.upper()
    )

    with STATE_LOCK:

        for state in (
            TRADE_STATE.values()
        ):

            if (
                state.get(
                    "underlying"
                )
                == underlying
                and state.get(
                    "active",
                    True,
                )
            ):
                return True

    open_symbols = (
        open_option_symbols()
    )

    with STATE_LOCK:

        for (
            option_symbol,
            state,
        ) in TRADE_STATE.items():

            if (
                state.get(
                    "underlying"
                )
                == underlying
                and option_symbol
                in open_symbols
            ):
                return True

    return False


def cooldown_active(
    underlying,
):
    with STATE_LOCK:

        last_time = (
            LAST_ENTRY_TIME.get(
                underlying
            )
        )

    if not last_time:
        return False

    return (
        now_et()
        - last_time
        < timedelta(
            minutes=(
                ENTRY_COOLDOWN_MINUTES
            )
        )
    )


def calculate_contract_qty(
    option_price,
):
    if option_price <= 0:
        return 0

    equity = (
        account_equity()
    )

    buying_power = (
        account_buying_power()
    )

    if (
        equity <= 0
        or buying_power <= 0
    ):
        return 0

    risk_budget = (
        equity
        * RISK_PER_TRADE_PERCENT
    )

    risk_per_contract = (
        option_price
        * 100.0
        * HARD_STOP_PERCENT
    )

    if risk_per_contract <= 0:
        return 0

    qty_by_risk = math.floor(
        risk_budget
        / risk_per_contract
    )

    maximum_position_value = (
        equity
        * MAX_POSITION_VALUE_PERCENT
    )

    contract_value = (
        option_price
        * 100.0
    )

    qty_by_exposure = math.floor(
        maximum_position_value
        / contract_value
    )

    qty_by_buying_power = (
        math.floor(
            (
                buying_power
                * 0.90
            )
            / contract_value
        )
    )

    qty = min(
        qty_by_risk,
        qty_by_exposure,
        qty_by_buying_power,
        MAX_CONTRACTS_PER_TRADE,
    )

    if (
        qty
        < MIN_CONTRACTS_FOR_RUNNER
    ):
        return 0

    return qty


def risk_allows_new_trade(
    underlying,
):
    if daily_loss_limit_reached():

        return (
            False,
            "daily loss limit reached",
        )

    if (
        len(
            get_option_positions()
        )
        >= MAX_OPEN_POSITIONS
    ):

        return (
            False,
            "maximum open positions reached",
        )

    if not entry_window_open():

        return (
            False,
            "outside entry window",
        )

    if force_exit_time():

        return (
            False,
            "force-exit time reached",
        )

    if position_exists_for_underlying(
        underlying
    ):

        return (
            False,
            "already in this underlying",
        )

    if cooldown_active(
        underlying
    ):

        return (
            False,
            "underlying cooldown active",
        )

    return (
        True,
        "ok",
    )


# ============================================================
# ENTRY
# ============================================================

def enter_signal(
    signal,
):
    underlying = (
        signal["symbol"]
    )

    option_type = (
        signal["side"]
    )

    stock_price = safe_float(
        signal[
            "underlying_price"
        ],
        0.0,
    )

    (
        allowed,
        reason,
    ) = risk_allows_new_trade(
        underlying
    )

    if not allowed:

        return {
            "entered": False,
            "symbol": underlying,
            "reason": reason,
        }

    contract = (
        choose_option_contract(
            underlying,
            option_type,
            stock_price,
        )
    )

    if not contract:

        return {
            "entered": False,
            "symbol": underlying,
            "reason": (
                "no acceptable "
                "0DTE contract"
            ),
        }

    entry_reference = (
        contract["ask"]
        or contract["mid"]
    )

    qty = calculate_contract_qty(
        entry_reference
    )

    if (
        qty
        < MIN_CONTRACTS_FOR_RUNNER
    ):

        return {
            "entered": False,
            "symbol": underlying,
            "reason": (
                "position size too "
                "small for TP + runner"
            ),
        }

    if not AUTO_TRADE:

        return {
            "entered": False,
            "paper_order_disabled": True,
            "underlying": underlying,
            "option_symbol": (
                contract["symbol"]
            ),
            "option_type": (
                option_type
            ),
            "qty": qty,
            "reference_price": (
                entry_reference
            ),
        }

    order = submit_market_order(
        contract["symbol"],
        qty,
        "buy",
    )

    order_id = (
        order.get("id")
    )

    if not order_id:

        return {
            "entered": False,
            "symbol": underlying,
            "reason": (
                f"order missing id: "
                f"{order}"
            ),
        }

    filled = wait_for_order_fill(
        order_id
    )

    if (
        str(
            filled.get(
                "status",
                "",
            )
        ).lower()
        != "filled"
    ):

        return {
            "entered": False,
            "symbol": underlying,
            "reason": (
                "entry not filled: "
                f"{filled.get('status')}"
            ),
            "order": filled,
        }

    filled_qty = safe_int(
        filled.get(
            "filled_qty"
        ),
        qty,
    )

    fill_price = safe_float(
        filled.get(
            "filled_avg_price"
        ),
        entry_reference,
    )

    if (
        filled_qty < 1
        or fill_price <= 0
    ):

        return {
            "entered": False,
            "symbol": underlying,
            "reason": "invalid fill",
            "order": filled,
        }

    with STATE_LOCK:

        TRADE_STATE[
            contract["symbol"]
        ] = {
            "active": True,
            "underlying": underlying,
            "option_type": option_type,
            "entry_price": (
                fill_price
            ),
            "original_qty": (
                filled_qty
            ),
            "remaining_qty": (
                filled_qty
            ),
            "tp_hit": False,
            "runner_high": None,
            "partial_exit_qty": 0,
            "entry_time": (
                now_et()
            ),
            "entry_bar_time": (
                signal.get(
                    "bar_time"
                )
            ),
        }

        LAST_ENTRY_TIME[
            underlying
        ] = now_et()

    print(
        f"[ENTRY] "
        f"{underlying} "
        f"{option_type.upper()} "
        f"{contract['symbol']} "
        f"qty={filled_qty} "
        f"@ {fill_price:.2f}",
        flush=True,
    )

    return {
        "entered": True,
        "underlying": underlying,
        "option_symbol": (
            contract["symbol"]
        ),
        "option_type": option_type,
        "qty": filled_qty,
        "fill_price": fill_price,
    }


# ============================================================
# POSITION MANAGEMENT
# ============================================================

def sync_state_from_positions():
    positions = (
        get_option_positions()
    )

    open_map = {
        str(
            position.get(
                "symbol",
                "",
            )
        ).upper(): position
        for position
        in positions
    }

    with STATE_LOCK:

        for (
            option_symbol,
            state,
        ) in list(
            TRADE_STATE.items()
        ):

            if (
                option_symbol
                not in open_map
            ):

                state[
                    "active"
                ] = False

                state[
                    "remaining_qty"
                ] = 0

        for (
            option_symbol,
            position,
        ) in open_map.items():

            if (
                option_symbol
                in TRADE_STATE
            ):

                TRADE_STATE[
                    option_symbol
                ][
                    "remaining_qty"
                ] = abs(
                    safe_int(
                        position.get(
                            "qty"
                        ),
                        0,
                    )
                )


def exit_entire_trade(
    option_symbol,
    reason,
):
    with STATE_LOCK:

        state = (
            TRADE_STATE.get(
                option_symbol
            )
        )

        if (
            not state
            or not state.get(
                "active",
                False,
            )
        ):
            return None

        qty = safe_int(
            state.get(
                "remaining_qty"
            ),
            0,
        )

    if qty <= 0:
        return None

    if AUTO_TRADE:

        order = (
            close_option_position(
                option_symbol,
                qty,
            )
        )

    else:

        order = {
            "simulated": True
        }

    with STATE_LOCK:

        if (
            option_symbol
            in TRADE_STATE
        ):

            TRADE_STATE[
                option_symbol
            ][
                "active"
            ] = False

            TRADE_STATE[
                option_symbol
            ][
                "remaining_qty"
            ] = 0

            TRADE_STATE[
                option_symbol
            ][
                "exit_reason"
            ] = reason

            TRADE_STATE[
                option_symbol
            ][
                "exit_time"
            ] = now_et()

    print(
        f"[EXIT] "
        f"{option_symbol} "
        f"qty={qty} "
        f"reason={reason}",
        flush=True,
    )

    return order


def manage_trade(
    option_symbol,
    state,
):
    if not state.get(
        "active",
        False,
    ):
        return

    remaining_qty = safe_int(
        state.get(
            "remaining_qty"
        ),
        0,
    )

    if remaining_qty <= 0:
        return

    entry_price = safe_float(
        state.get(
            "entry_price"
        ),
        0.0,
    )

    if entry_price <= 0:
        return

    # Force all remaining 0DTE
    # positions out at cutoff.
    if force_exit_time():

        exit_entire_trade(
            option_symbol,
            "0DTE force exit",
        )

        return

    quote = (
        get_latest_option_quote(
            option_symbol
        )
    )

    # Use bid first because we are
    # managing a long option that
    # would need to be sold.
    option_price = (
        quote["bid"]
        or quote["mid"]
    )

    if option_price <= 0:
        return

    pnl_percent = (
        option_price
        - entry_price
    ) / entry_price

    tp_hit = bool(
        state.get(
            "tp_hit",
            False,
        )
    )

    # ========================================================
    # HARD STOP BEFORE TP
    # ========================================================

    if (
        not tp_hit
        and pnl_percent
        <= -HARD_STOP_PERCENT
    ):

        exit_entire_trade(
            option_symbol,
            "hard premium stop",
        )

        return

    # ========================================================
    # EMA 9 INVALIDATION BEFORE TP
    # ========================================================

    if not tp_hit:

        try:

            invalidated = (
                technical_exit_trigger(
                    state[
                        "underlying"
                    ],
                    state[
                        "option_type"
                    ],
                )
            )

            if invalidated:

                exit_entire_trade(
                    option_symbol,
                    (
                        "4-minute EMA9 "
                        "invalidation"
                    ),
                )

                return

        except Exception as exc:

            print(
                f"[TECH EXIT ERROR] "
                f"{option_symbol}: "
                f"{exc}",
                flush=True,
            )

    # ========================================================
    # FIRST TAKE PROFIT
    # ========================================================

    if (
        not tp_hit
        and pnl_percent
        >= TAKE_PROFIT_PERCENT
    ):

        original_qty = safe_int(
            state.get(
                "original_qty"
            ),
            remaining_qty,
        )

        sell_qty = max(
            1,
            math.floor(
                original_qty
                * TAKE_PROFIT_FRACTION
            ),
        )

        # Always leave at least
        # one runner.
        sell_qty = min(
            sell_qty,
            max(
                0,
                remaining_qty - 1,
            ),
        )

        if sell_qty <= 0:

            exit_entire_trade(
                option_symbol,
                "take profit",
            )

            return

        if AUTO_TRADE:

            close_option_position(
                option_symbol,
                sell_qty,
            )

        new_remaining = (
            remaining_qty
            - sell_qty
        )

        with STATE_LOCK:

            trade = (
                TRADE_STATE.get(
                    option_symbol
                )
            )

            if trade:

                trade[
                    "tp_hit"
                ] = True

                trade[
                    "partial_exit_qty"
                ] = sell_qty

                trade[
                    "remaining_qty"
                ] = new_remaining

                trade[
                    "runner_high"
                ] = option_price

        print(
            f"[TP] "
            f"{option_symbol} "
            f"sold={sell_qty} "
            f"remaining="
            f"{new_remaining} "
            f"price="
            f"{option_price:.2f}",
            flush=True,
        )

        return

    # ========================================================
    # RUNNER
    # ========================================================

    if tp_hit:

        runner_high = safe_float(
            state.get(
                "runner_high"
            ),
            option_price,
        )

        runner_high = max(
            runner_high,
            option_price,
        )

        with STATE_LOCK:

            trade = (
                TRADE_STATE.get(
                    option_symbol
                )
            )

            if trade:

                trade[
                    "runner_high"
                ] = runner_high

        trailing_exit = (
            runner_high
            * (
                1.0
                - RUNNER_TRAIL_PERCENT
            )
        )

        if (
            option_price
            <= trailing_exit
        ):

            exit_entire_trade(
                option_symbol,
                (
                    "runner "
                    "trailing stop"
                ),
            )

            return


def manage_all_trades():
    sync_state_from_positions()

    with STATE_LOCK:

        snapshot = {
            symbol: dict(
                state
            )
            for (
                symbol,
                state,
            )
            in TRADE_STATE.items()
            if state.get(
                "active",
                False,
            )
        }

    for (
        option_symbol,
        state,
    ) in snapshot.items():

        try:

            manage_trade(
                option_symbol,
                state,
            )

        except Exception as exc:

            print(
                f"[MANAGE ERROR] "
                f"{option_symbol}: "
                f"{exc}",
                flush=True,
            )


# ============================================================
# ENTRY SCANNER
# ============================================================

def scan_for_entries():
    if not entry_window_open():
        return []

    if daily_loss_limit_reached():
        return []

    candidates = (
        refresh_scanner()
    )

    results = []

    for candidate in candidates:

        if (
            len(
                get_option_positions()
            )
            >= MAX_OPEN_POSITIONS
        ):
            break

        symbol = (
            candidate["symbol"]
        )

        if (
            position_exists_for_underlying(
                symbol
            )
        ):
            continue

        if cooldown_active(
            symbol
        ):
            continue

        try:

            signal = (
                detect_signal(
                    symbol
                )
            )

            if not signal:
                continue

            result = (
                enter_signal(
                    signal
                )
            )

            result[
                "signal"
            ] = signal

            result[
                "scanner"
            ] = candidate

            results.append(
                result
            )

        except Exception as exc:

            results.append(
                {
                    "symbol": symbol,
                    "entered": False,
                    "error": str(
                        exc
                    ),
                }
            )

    return results


# ============================================================
# BOT LOOP
# ============================================================

def run_bot_cycle():
    global LAST_LOOP_ERROR
    global LAST_LOOP_TIME

    LAST_LOOP_TIME = (
        now_et().isoformat()
    )

    LAST_LOOP_ERROR = None

    try:

        # Manage existing trades first.
        manage_all_trades()

        # Then look for new entries.
        if (
            entry_window_open()
            and not
            daily_loss_limit_reached()
        ):

            scan_for_entries()

    except Exception as exc:

        LAST_LOOP_ERROR = str(
            exc
        )

        print(
            f"[BOT CYCLE ERROR] "
            f"{exc}",
            flush=True,
        )


def bot_loop():
    print(
        "[BOT] "
        "0DTE paper bot "
        "loop started.",
        flush=True,
    )

    while True:

        run_bot_cycle()

        time.sleep(
            BOT_LOOP_SECONDS
        )


def start_bot_thread():
    global BOT_THREAD_STARTED

    if not RUN_BOT_LOOP:
        return

    with BOT_THREAD_LOCK:

        if BOT_THREAD_STARTED:
            return

        thread = threading.Thread(
            target=bot_loop,
            daemon=True,
            name="alpaca-0dte-bot",
        )

        thread.start()

        BOT_THREAD_STARTED = True


# ============================================================
# FLASK ROUTES
# ============================================================

@app.get("/")
def home():
    return jsonify(
        {
            "ok": True,
            "name": (
                "Alpaca 0DTE "
                "Paper Trading Bot"
            ),
            "paper_only": True,
            "auto_trade": (
                AUTO_TRADE
            ),
            "run_bot_loop": (
                RUN_BOT_LOOP
            ),
            "strategy": (
                "4m EMA5 / EMA9 / "
                "EMA30 / VWAP"
            ),
            "take_profit_percent": (
                TAKE_PROFIT_PERCENT
            ),
            "take_profit_fraction": (
                TAKE_PROFIT_FRACTION
            ),
            "runner_trail_percent": (
                RUNNER_TRAIL_PERCENT
            ),
            "hard_stop_percent": (
                HARD_STOP_PERCENT
            ),
        }
    )


@app.get("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "paper_only": True,
            "last_loop_time": (
                LAST_LOOP_TIME
            ),
            "last_loop_error": (
                LAST_LOOP_ERROR
            ),
            "bot_thread_started": (
                BOT_THREAD_STARTED
            ),
        }
    )


@app.get("/status")
def status():
    try:

        account = (
            get_account()
        )

        positions = (
            get_option_positions()
        )

        with STATE_LOCK:

            trade_state = {}

            for (
                key,
                value,
            ) in TRADE_STATE.items():

                copied = dict(
                    value
                )

                if isinstance(
                    copied.get(
                        "entry_time"
                    ),
                    datetime,
                ):

                    copied[
                        "entry_time"
                    ] = (
                        copied[
                            "entry_time"
                        ].isoformat()
                    )

                if isinstance(
                    copied.get(
                        "exit_time"
                    ),
                    datetime,
                ):

                    copied[
                        "exit_time"
                    ] = (
                        copied[
                            "exit_time"
                        ].isoformat()
                    )

                trade_state[
                    key
                ] = copied

        return jsonify(
            {
                "ok": True,
                "paper_only": True,
                "auto_trade": (
                    AUTO_TRADE
                ),
                "equity": (
                    account.get(
                        "equity"
                    )
                ),
                "buying_power": (
                    account.get(
                        "options_buying_power"
                    )
                    or account.get(
                        "buying_power"
                    )
                ),
                "daily_pnl": (
                    daily_pnl()
                ),
                "option_positions": (
                    positions
                ),
                "trade_state": (
                    trade_state
                ),
                "last_loop_time": (
                    LAST_LOOP_TIME
                ),
                "last_loop_error": (
                    LAST_LOOP_ERROR
                ),
            }
        )

    except Exception as exc:

        return jsonify(
            {
                "ok": False,
                "error": str(
                    exc
                ),
            }
        ), 500


@app.get("/scan")
def scan():
    try:

        force = (
            request.args.get(
                "force",
                "false",
            )
            .strip()
            .lower()
            == "true"
        )

        candidates = (
            refresh_scanner(
                force=force
            )
        )

        return jsonify(
            {
                "ok": True,
                "count": len(
                    candidates
                ),
                "candidates": (
                    candidates
                ),
            }
        )

    except Exception as exc:

        return jsonify(
            {
                "ok": False,
                "error": str(
                    exc
                ),
            }
        ), 500


@app.post("/run-once")
def run_once():
    try:

        run_bot_cycle()

        return jsonify(
            {
                "ok": True,
                "last_loop_time": (
                    LAST_LOOP_TIME
                ),
                "last_loop_error": (
                    LAST_LOOP_ERROR
                ),
            }
        )

    except Exception as exc:

        return jsonify(
            {
                "ok": False,
                "error": str(
                    exc
                ),
            }
        ), 500


# ============================================================
# START BACKGROUND BOT
# ============================================================

start_bot_thread()


# ============================================================
# LOCAL / RENDER START
# ============================================================

if __name__ == "__main__":

    port = safe_int(
        os.environ.get(
            "PORT"
        ),
        10000,
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )
import os
import time
import threading
from collections import deque
from datetime import datetime, timedelta, time as dt_time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, request

# ============================================================
# CONFIG
# ============================================================
app = Flask(__name__)
NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

TRADING_BASE_URL = "https://paper-api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "").strip()
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "").strip()
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()

AUTO_TRADE = os.getenv("AUTO_TRADE", "false").lower() == "true"
DATA_FEED = os.getenv("DATA_FEED", "iex").lower()
OPTION_FEED = os.getenv("OPTION_FEED", "indicative").lower()

TIMEFRAME_MINUTES = 4
EMA_FAST = 5
EMA_SLOW = 9
EMA_TREND = 30
PREMARKET_START = dt_time(4, 0)
PREMARKET_END = dt_time(9, 30)
RTH_START = dt_time(9, 30)
RTH_END = dt_time(16, 0)

# EXACT FILTER REQUESTED
ROLLING_TRADE_COUNT = 64
MIN_WIN_RATE = float(os.getenv("MIN_WIN_RATE", "0.90"))
MIN_DIRECTION_TRADES = int(os.getenv("MIN_DIRECTION_TRADES", "64"))
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "2"))
BACKTEST_LOOKBACK_DAYS = int(os.getenv("BACKTEST_LOOKBACK_DAYS", "365"))

# Scanner size. 0 = attempt entire tradable US-equity universe.
SCAN_LIMIT = int(os.getenv("SCAN_LIMIT", "150"))
SCAN_REFRESH_MINUTES = int(os.getenv("SCAN_REFRESH_MINUTES", "30"))

# Options execution / protection
POSITION_DOLLARS = float(os.getenv("POSITION_DOLLARS", "500"))
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "0.20"))
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "3"))
STOP_MONITOR_SECONDS = int(os.getenv("STOP_MONITOR_SECONDS", "5"))

# Pine targetMove behavior: IWM = $0.50, everything else = $1.00
IWM_TARGET_MOVE = float(os.getenv("IWM_TARGET_MOVE", "0.50"))
DEFAULT_TARGET_MOVE = float(os.getenv("DEFAULT_TARGET_MOVE", "1.00"))

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    "Content-Type": "application/json",
}

PRIORITY_SYMBOLS = [
    "SPY", "QQQ", "IWM", "AAPL", "NVDA", "TSLA", "AMD", "AMZN",
    "META", "MSFT", "GOOGL", "NFLX", "AVGO", "PLTR", "COIN", "MSTR", "AMAT"
]

# ============================================================
# STATE
# ============================================================
state_lock = threading.RLock()
approved = {}  # symbol -> {CALL: stats|None, PUT: stats|None}
managed_positions = {}  # underlying -> option position metadata
last_webhook = None
last_scan = None
scanner_running = False
errors = deque(maxlen=50)


def now_et():
    return datetime.now(NY)


def log(msg):
    print(f"[{now_et():%Y-%m-%d %H:%M:%S} ET] {msg}", flush=True)


def add_error(msg):
    errors.append(str(msg))
    log(f"ERROR: {msg}")


# ============================================================
# ALPACA HTTP
# ============================================================
def alpaca_get(path, params=None, data_api=False):
    base = DATA_BASE_URL if data_api else TRADING_BASE_URL
    r = requests.get(f"{base}{path}", headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def alpaca_post(path, payload):
    r = requests.post(f"{TRADING_BASE_URL}{path}", headers=HEADERS, json=payload, timeout=30)
    if not r.ok:
        raise RuntimeError(f"Alpaca POST {path} HTTP {r.status_code}: {r.text}")
    return r.json() if r.text else {}


def alpaca_delete(path):
    r = requests.delete(f"{TRADING_BASE_URL}{path}", headers=HEADERS, timeout=30)
    if not r.ok and r.status_code != 404:
        raise RuntimeError(f"Alpaca DELETE {path} HTTP {r.status_code}: {r.text}")
    return r.json() if r.text else {}


# ============================================================
# DATA
# ============================================================
def bars_to_df(bars):
    if not bars:
        return None
    df = pd.DataFrame(bars)
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert(NY)
    for src, dst in [("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"), ("v", "volume")]:
        df[dst] = pd.to_numeric(df[src], errors="coerce")
    return (
        df[["timestamp", "open", "high", "low", "close", "volume"]]
        .dropna()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def get_historical_bars(symbol, days=BACKTEST_LOOKBACK_DAYS):
    end = datetime.now(UTC)
    start = end - timedelta(days=days)
    bars = []
    page_token = None
    for _ in range(50):
        params = {
            "timeframe": f"{TIMEFRAME_MINUTES}Min",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 10000,
            "adjustment": "raw",
            "feed": DATA_FEED,
        }
        if page_token:
            params["page_token"] = page_token
        data = alpaca_get(f"/v2/stocks/{symbol}/bars", params=params, data_api=True)
        bars.extend(data.get("bars", []))
        page_token = data.get("next_page_token")
        if not page_token:
            break
    return bars_to_df(bars)


def get_stock_universe():
    assets = alpaca_get("/v2/assets", params={"status": "active", "asset_class": "us_equity"})
    symbols = []
    for a in assets:
        s = a.get("symbol")
        if s and a.get("tradable") and "." not in s:
            symbols.append(s)
    ordered = list(dict.fromkeys(PRIORITY_SYMBOLS + symbols))
    return ordered if SCAN_LIMIT == 0 else ordered[:SCAN_LIMIT]


# ============================================================
# EXACT PINE-LIKE INDICATORS
# ============================================================
def calculate_indicators(df):
    if df is None or len(df) < 40:
        return None
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["ema5"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema9"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["ema30"] = df["close"].ewm(span=EMA_TREND, adjust=False).mean()

    session_date = df["timestamp"].dt.date
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    cumulative_pv = pv.groupby(session_date).cumsum()
    cumulative_v = df["volume"].groupby(session_date).cumsum().replace(0, np.nan)
    df["vwap"] = cumulative_pv / cumulative_v

    df["pm_high"] = np.nan
    df["pm_low"] = np.nan
    for _, idxs in df.groupby(session_date).groups.items():
        idxs = list(idxs)
        rows = df.loc[idxs]
        t = rows["timestamp"].dt.time
        pm = rows[(t >= PREMARKET_START) & (t < PREMARKET_END)]
        if pm.empty:
            continue
        df.loc[idxs, "pm_high"] = float(pm["high"].max())
        df.loc[idxs, "pm_low"] = float(pm["low"].min())
    return df


def target_move(symbol):
    return IWM_TARGET_MOVE if symbol.upper() == "IWM" else DEFAULT_TARGET_MOVE


# ============================================================
# PINE-MATCHING BACKTEST
# Entry:
#   CALL: close breaks above PM high + EMA5>EMA9>EMA30 + close>VWAP
#   PUT : close breaks below PM low  + EMA5<EMA9<EMA30 + close<VWAP
# Exit:
#   TP on underlying target move; otherwise EMA9 close exit.
# Max 2 entries/day total, one position at a time.
# ============================================================
def pine_backtest(symbol, raw_df):
    df = calculate_indicators(raw_df)
    if df is None:
        return []

    trades = []
    in_trade = False
    direction = None
    entry = None
    target = None
    current_day = None
    trades_today = 0
    move = target_move(symbol)

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        ts = row["timestamp"]
        day = ts.date()
        tm = ts.time()

        if day != current_day:
            current_day = day
            trades_today = 0

        if not (RTH_START <= tm < RTH_END):
            continue

        pm_high = row["pm_high"]
        pm_low = row["pm_low"]
        if pd.isna(pm_high) or pd.isna(pm_low) or pd.isna(row["vwap"]):
            continue

        # Manage existing position first, exactly like Pine's bar-by-bar state.
        if in_trade:
            if direction == "CALL":
                if float(row["high"]) >= target:
                    trades.append({
                        "symbol": symbol, "direction": "CALL", "entry": entry,
                        "exit": target, "win": True, "entry_time": entry_ts,
                        "exit_time": ts, "reason": "TP"
                    })
                    in_trade = False
                elif float(row["close"]) <= float(row["ema9"]):
                    exit_price = float(row["close"])
                    trades.append({
                        "symbol": symbol, "direction": "CALL", "entry": entry,
                        "exit": exit_price, "win": exit_price > entry,
                        "entry_time": entry_ts, "exit_time": ts, "reason": "EMA9"
                    })
                    in_trade = False
            else:
                if float(row["low"]) <= target:
                    trades.append({
                        "symbol": symbol, "direction": "PUT", "entry": entry,
                        "exit": target, "win": True, "entry_time": entry_ts,
                        "exit_time": ts, "reason": "TP"
                    })
                    in_trade = False
                elif float(row["close"]) >= float(row["ema9"]):
                    exit_price = float(row["close"])
                    trades.append({
                        "symbol": symbol, "direction": "PUT", "entry": entry,
                        "exit": exit_price, "win": exit_price < entry,
                        "entry_time": entry_ts, "exit_time": ts, "reason": "EMA9"
                    })
                    in_trade = False

            if in_trade:
                continue

        if trades_today >= MAX_TRADES_PER_DAY:
            continue

        bull = float(row["ema5"]) > float(row["ema9"]) > float(row["ema30"])
        bear = float(row["ema5"]) < float(row["ema9"]) < float(row["ema30"])
        long_filter = bull and float(row["close"]) > float(row["vwap"])
        short_filter = bear and float(row["close"]) < float(row["vwap"])

        long_entry = (
            float(row["close"]) > float(pm_high)
            and float(prev["close"]) <= float(pm_high)
            and long_filter
        )
        short_entry = (
            float(row["close"]) < float(pm_low)
            and float(prev["close"]) >= float(pm_low)
            and short_filter
        )

        if long_entry:
            in_trade = True
            direction = "CALL"
            entry = float(row["close"])
            target = entry + move
            entry_ts = ts
            trades_today += 1
        elif short_entry:
            in_trade = True
            direction = "PUT"
            entry = float(row["close"])
            target = entry - move
            entry_ts = ts
            trades_today += 1

    return trades


def direction_stats(trades, direction):
    d = [t for t in trades if t["direction"] == direction]
    d = d[-ROLLING_TRADE_COUNT:]
    total = len(d)
    wins = sum(1 for t in d if t["win"])
    rate = wins / total if total else 0.0
    return {
        "trades": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": round(rate * 100.0, 2),
        "qualified": total >= MIN_DIRECTION_TRADES and rate >= MIN_WIN_RATE,
    }


def scan_symbol(symbol):
    try:
        df = get_historical_bars(symbol)
        trades = pine_backtest(symbol, df)
        call = direction_stats(trades, "CALL")
        put = direction_stats(trades, "PUT")
        return {"CALL": call, "PUT": put, "total_completed": len(trades)}
    except Exception as e:
        add_error(f"{symbol} scan failed: {e}")
        return None


def run_scanner_once():
    global last_scan, scanner_running
    with state_lock:
        if scanner_running:
            return
        scanner_running = True
    try:
        universe = get_stock_universe()
        new_approved = {}
        log(f"SCANNER START | symbols={len(universe)} | threshold={MIN_WIN_RATE:.0%} | last={ROLLING_TRADE_COUNT}")
        for n, symbol in enumerate(universe, 1):
            stats = scan_symbol(symbol)
            if not stats:
                continue
            call_ok = stats["CALL"]["qualified"]
            put_ok = stats["PUT"]["qualified"]
            if call_ok or put_ok:
                new_approved[symbol] = {
                    "CALL": stats["CALL"] if call_ok else None,
                    "PUT": stats["PUT"] if put_ok else None,
                }
                log(
                    f"APPROVED {symbol} | "
                    f"CALL={stats['CALL']['win_rate']}% ({stats['CALL']['wins']}/{stats['CALL']['trades']}) | "
                    f"PUT={stats['PUT']['win_rate']}% ({stats['PUT']['wins']}/{stats['PUT']['trades']})"
                )
            if n % 25 == 0:
                log(f"SCANNER PROGRESS {n}/{len(universe)} | approved={len(new_approved)}")
        with state_lock:
            approved.clear()
            approved.update(new_approved)
            last_scan = now_et().isoformat()
        log(f"SCANNER DONE | approved symbols={len(new_approved)}")
    finally:
        with state_lock:
            scanner_running = False


def scanner_loop():
    while True:
        try:
            run_scanner_once()
        except Exception as e:
            add_error(f"scanner loop: {e}")
        time.sleep(max(5, SCAN_REFRESH_MINUTES) * 60)


# ============================================================
# OPTIONS HELPERS
# ============================================================
def today_ymd():
    return now_et().date().isoformat()


def get_underlying_price(symbol):
    data = alpaca_get(
        f"/v2/stocks/{symbol}/trades/latest",
        params={"feed": DATA_FEED},
        data_api=True,
    )
    trade = data.get("trade", {})
    return float(trade.get("p"))


def get_0dte_contract(symbol, direction):
    expiration = today_ymd()
    side = "call" if direction == "CALL" else "put"
    data = alpaca_get(
        "/v2/options/contracts",
        params={
            "underlying_symbols": symbol,
            "expiration_date": expiration,
            "type": side,
            "status": "active",
            "limit": 1000,
        },
    )
    contracts = data.get("option_contracts", [])
    if not contracts:
        raise RuntimeError(f"No {expiration} {direction} contracts found for {symbol}")
    px = get_underlying_price(symbol)
    best = min(contracts, key=lambda c: abs(float(c.get("strike_price", 0)) - px))
    return best["symbol"]


def latest_option_mid(option_symbol):
    # Alpaca option latest quotes endpoint.
    data = alpaca_get(
        f"/v1beta1/options/quotes/latest",
        params={"symbols": option_symbol, "feed": OPTION_FEED},
        data_api=True,
    )
    quotes = data.get("quotes", {})
    q = quotes.get(option_symbol, {}) if isinstance(quotes, dict) else {}
    bid = float(q.get("bp") or 0)
    ask = float(q.get("ap") or 0)
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return ask or bid or 0.0


def option_position(option_symbol):
    try:
        return alpaca_get(f"/v2/positions/{option_symbol}")
    except Exception:
        return None


def close_option(option_symbol, qty=None):
    pos = option_position(option_symbol)
    if not pos:
        return None
    available = abs(float(pos.get("qty", 0)))
    sell_qty = int(available if qty is None else min(available, qty))
    if sell_qty <= 0:
        return None
    if not AUTO_TRADE:
        log(f"PAPER-DRY-RUN CLOSE {option_symbol} qty={sell_qty}")
        return {"dry_run": True}
    return alpaca_post("/v2/orders", {
        "symbol": option_symbol,
        "qty": sell_qty,
        "side": "sell",
        "type": "market",
        "time_in_force": "day",
    })


def enter_option(underlying, direction):
    with state_lock:
        if underlying in managed_positions:
            raise RuntimeError(f"Duplicate blocked: {underlying} already managed")
        if len(managed_positions) >= MAX_OPEN_POSITIONS:
            raise RuntimeError("Max open managed positions reached")

    option_symbol = get_0dte_contract(underlying, direction)
    mid = latest_option_mid(option_symbol)
    if mid <= 0:
        raise RuntimeError(f"No usable option quote for {option_symbol}")
    contracts = max(1, int(POSITION_DOLLARS // (mid * 100.0)))

    if AUTO_TRADE:
        order = alpaca_post("/v2/orders", {
            "symbol": option_symbol,
            "qty": contracts,
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
        })
    else:
        order = {"dry_run": True, "symbol": option_symbol, "qty": contracts}
        log(f"PAPER-DRY-RUN BUY {option_symbol} qty={contracts} @ approx {mid:.2f}")

    with state_lock:
        managed_positions[underlying] = {
            "underlying": underlying,
            "direction": direction,
            "option_symbol": option_symbol,
            "qty": contracts,
            "reference_entry_mid": mid,
            "stop_price": round(mid * (1.0 - STOP_LOSS_PERCENT), 4),
            "entered_at": now_et().isoformat(),
            "order": order,
        }
    log(f"ENTRY ACCEPTED {underlying} {direction} -> {option_symbol} | stop~{mid*(1-STOP_LOSS_PERCENT):.2f}")
    return managed_positions[underlying]


def exit_underlying(underlying, reason):
    with state_lock:
        meta = managed_positions.get(underlying)
    if not meta:
        return {"ignored": True, "reason": "no managed position"}
    result = close_option(meta["option_symbol"])
    with state_lock:
        managed_positions.pop(underlying, None)
    log(f"EXIT {underlying} | reason={reason}")
    return result or {"closed": True}


def stop_monitor_loop():
    while True:
        try:
            with state_lock:
                snapshot = list(managed_positions.items())
            for underlying, meta in snapshot:
                mid = latest_option_mid(meta["option_symbol"])
                if mid > 0 and mid <= float(meta["stop_price"]):
                    log(f"EMERGENCY STOP {underlying} | option={mid:.2f} <= {meta['stop_price']:.2f}")
                    exit_underlying(underlying, "OPTION_STOP")
        except Exception as e:
            add_error(f"stop monitor: {e}")
        time.sleep(max(2, STOP_MONITOR_SECONDS))


# ============================================================
# WEBHOOK GATES
# TradingView is the live-entry authority.
# Scanner is the 90% historical qualification authority.
# ============================================================
def normalize_symbol(s):
    return str(s or "").upper().replace("NASDAQ:", "").replace("NYSE:", "").replace("AMEX:", "")


def validate_tv_stats(payload, direction):
    # If Pine sends direction-specific TradingView stats, enforce them as a final exact-TV gate.
    # This is optional until you add the corresponding fields to the Pine webhook.
    key = "callWinRate" if direction == "CALL" else "putWinRate"
    trades_key = "callTrades" if direction == "CALL" else "putTrades"
    if key not in payload:
        return True, "not supplied"
    try:
        rate = float(payload[key])
        trades = int(float(payload.get(trades_key, 0)))
    except Exception:
        return False, "invalid TradingView stats"
    if trades < MIN_DIRECTION_TRADES:
        return False, f"TradingView sample too small ({trades}/{MIN_DIRECTION_TRADES})"
    if rate < MIN_WIN_RATE * 100:
        return False, f"TradingView direction win rate {rate:.2f}% < {MIN_WIN_RATE*100:.0f}%"
    return True, f"TradingView {rate:.2f}%/{trades}"


@app.post("/webhook")
def webhook():
    global last_webhook
    payload = request.get_json(silent=True) or {}
    last_webhook = {"received_at": now_et().isoformat(), "payload": payload}

    if WEBHOOK_SECRET and payload.get("secret") != WEBHOOK_SECRET:
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    if payload.get("source") != "TRADINGVIEW":
        return jsonify({"ok": False, "error": "source must be TRADINGVIEW"}), 400

    symbol = normalize_symbol(payload.get("symbol"))
    action = str(payload.get("action", "")).upper()
    direction = str(payload.get("direction", "")).upper()

    if not symbol or action not in {"ENTRY", "TAKE_PROFIT", "EXIT"}:
        return jsonify({"ok": False, "error": "invalid signal"}), 400

    # Exit messages are allowed even if the symbol later falls off the approved list.
    if action in {"TAKE_PROFIT", "EXIT"}:
        result = exit_underlying(symbol, action)
        return jsonify({"ok": True, "symbol": symbol, "action": action, "result": result})

    if direction not in {"CALL", "PUT"}:
        return jsonify({"ok": False, "error": "direction must be CALL or PUT"}), 400

    with state_lock:
        symbol_stats = approved.get(symbol)
        direction_approval = symbol_stats.get(direction) if symbol_stats else None

    if not direction_approval:
        return jsonify({
            "ok": False,
            "blocked": True,
            "reason": f"{symbol} {direction} is not on the 90% approved list",
        }), 403

    tv_ok, tv_reason = validate_tv_stats(payload, direction)
    if not tv_ok:
        return jsonify({"ok": False, "blocked": True, "reason": tv_reason}), 403

    try:
        meta = enter_option(symbol, direction)
        return jsonify({
            "ok": True,
            "executed": AUTO_TRADE,
            "symbol": symbol,
            "direction": direction,
            "scanner_stats": direction_approval,
            "tv_stats_gate": tv_reason,
            "position": meta,
        })
    except Exception as e:
        add_error(f"webhook entry {symbol} {direction}: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


# ============================================================
# STATUS ROUTES
# ============================================================
@app.get("/")
def root():
    return jsonify({
        "service": "TradingView-gated Alpaca 0DTE bot",
        "auto_trade": AUTO_TRADE,
        "min_win_rate": MIN_WIN_RATE,
        "rolling_trades": ROLLING_TRADE_COUNT,
        "approved_symbols": len(approved),
        "last_scan": last_scan,
    })


@app.get("/watchlist")
def watchlist():
    with state_lock:
        return jsonify({
            "threshold": MIN_WIN_RATE * 100,
            "required_direction_trades": MIN_DIRECTION_TRADES,
            "approved": approved,
            "last_scan": last_scan,
            "scanner_running": scanner_running,
        })


@app.get("/status")
def status():
    with state_lock:
        return jsonify({
            "auto_trade": AUTO_TRADE,
            "approved": approved,
            "managed_positions": managed_positions,
            "last_webhook": last_webhook,
            "last_scan": last_scan,
            "scanner_running": scanner_running,
            "errors": list(errors),
        })


@app.post("/scan-now")
def scan_now():
    threading.Thread(target=run_scanner_once, daemon=True).start()
    return jsonify({"ok": True, "message": "scan started"})


# ============================================================
# STARTUP
# ============================================================
def startup():
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        add_error("ALPACA_API_KEY / ALPACA_SECRET_KEY missing")
        return
    try:
        acct = alpaca_get("/v2/account")
        log(f"ALPACA CONNECTED | equity={acct.get('equity')} | AUTO_TRADE={AUTO_TRADE}")
    except Exception as e:
        add_error(f"credential check failed: {e}")
        return

    threading.Thread(target=scanner_loop, daemon=True).start()
    threading.Thread(target=stop_monitor_loop, daemon=True).start()


startup()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)

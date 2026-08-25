import os, time, math, threading, logging
from datetime import datetime, timedelta, time as dt_time
from zoneinfo import ZoneInfo

import requests
import pandas as pd
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
NY = ZoneInfo('America/New_York')
UTC = ZoneInfo('UTC')

ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '').strip()
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '').strip()
TRADING_URL = 'https://paper-api.alpaca.markets'
DATA_URL = 'https://data.alpaca.markets'
DATA_FEED = os.getenv('DATA_FEED', 'iex').strip().lower()
HEADERS = {'APCA-API-KEY-ID': ALPACA_API_KEY, 'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY}

TIMEFRAME = '4Min'
EMA_FAST, EMA_SLOW, EMA_TREND = 5, 9, 30
PREMARKET_START, PREMARKET_END = dt_time(4, 0), dt_time(9, 30)
RTH_START, RTH_END = dt_time(9, 30), dt_time(16, 0)

MIN_PRICE = float(os.getenv('MIN_PRICE', '5'))
MIN_DOLLAR_VOLUME = float(os.getenv('MIN_DOLLAR_VOLUME', '5000000'))
MAX_LIVE_UNIVERSE = int(os.getenv('MAX_LIVE_UNIVERSE', '250'))
WATCHLIST_SIZE = int(os.getenv('WATCHLIST_SIZE', '10'))
MIN_SETUP_SCORE = float(os.getenv('MIN_SETUP_SCORE', '70'))
SCAN_SECONDS = int(os.getenv('SCAN_SECONDS', '240'))
SNAPSHOT_BATCH = int(os.getenv('SNAPSHOT_BATCH', '200'))
BAR_BATCH = int(os.getenv('BAR_BATCH', '50'))
LEVEL_LOOKBACK = int(os.getenv('LEVEL_LOOKBACK', '90'))

PRIORITY = ['SPY','QQQ','IWM','AAPL','NVDA','TSLA','AMD','AMZN','META','MSFT','GOOGL','NFLX','AVGO','PLTR','COIN','MSTR']

lock = threading.Lock()
STATE = {
    'status': 'STARTING', 'last_scan': None, 'universe_count': 0,
    'liquid_count': 0, 'watchlist_count': 0, 'watchlist': [], 'error': None,
}


def req(method, url, params=None, timeout=45):
    r = requests.request(method, url, headers=HEADERS, params=params, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f'{r.status_code}: {r.text[:400]}')
    return r.json() if r.text else {}


def chunks(items, n):
    for i in range(0, len(items), n):
        yield items[i:i+n]


def get_universe():
    assets = req('GET', f'{TRADING_URL}/v2/assets', params={'status':'active','asset_class':'us_equity'})
    out = []
    for a in assets:
        s = str(a.get('symbol','')).upper().strip()
        if not s or not a.get('tradable', False) or '/' in s or '.' in s:
            continue
        out.append(s)
    return list(dict.fromkeys(PRIORITY + sorted(set(out))))


def liquid_universe(symbols):
    ranked = []
    for batch in chunks(symbols, SNAPSHOT_BATCH):
        data = req('GET', f'{DATA_URL}/v2/stocks/snapshots', params={'symbols': ','.join(batch), 'feed': DATA_FEED})
        for s, snap in (data or {}).items():
            day = snap.get('dailyBar') or {}
            prev = snap.get('prevDailyBar') or {}
            price = float((snap.get('latestTrade') or {}).get('p') or day.get('c') or 0)
            vol = float(day.get('v') or prev.get('v') or 0)
            dv = price * vol
            if price >= MIN_PRICE and dv >= MIN_DOLLAR_VOLUME:
                ranked.append((s, dv, price))
        time.sleep(0.05)
    ranked.sort(key=lambda x: x[1], reverse=True)
    keep = {s for s,_,_ in ranked[:MAX_LIVE_UNIVERSE]}
    keep.update(s for s in PRIORITY if s in symbols)
    meta = {s:{'dollar_volume':dv,'snapshot_price':p} for s,dv,p in ranked if s in keep}
    return list(keep), meta


def get_batch_bars(symbols, days=3):
    if not symbols: return {}
    end = datetime.now(UTC)
    start = end - timedelta(days=days)
    params = {
        'symbols': ','.join(symbols), 'timeframe': TIMEFRAME,
        'start': start.isoformat(), 'end': end.isoformat(), 'adjustment':'raw',
        'feed': DATA_FEED, 'sort':'asc', 'limit':10000,
    }
    out = {s:[] for s in symbols}
    token = None
    while True:
        if token: params['page_token'] = token
        elif 'page_token' in params: params.pop('page_token')
        data = req('GET', f'{DATA_URL}/v2/stocks/bars', params=params)
        for s, bars in (data.get('bars') or {}).items(): out.setdefault(s, []).extend(bars)
        token = data.get('next_page_token')
        if not token: break
    return out


def to_df(bars):
    if not bars: return pd.DataFrame()
    df = pd.DataFrame(bars).rename(columns={'t':'timestamp','o':'open','h':'high','l':'low','c':'close','v':'volume'})
    if not {'timestamp','open','high','low','close','volume'}.issubset(df.columns): return pd.DataFrame()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.set_index('timestamp').tz_convert(NY).sort_index()
    for c in ['open','high','low','close','volume']: df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna(subset=['open','high','low','close'])


def add_indicators(df):
    df = df.copy()
    df['ema5'] = df.close.ewm(span=EMA_FAST, adjust=False).mean()
    df['ema9'] = df.close.ewm(span=EMA_SLOW, adjust=False).mean()
    df['ema30'] = df.close.ewm(span=EMA_TREND, adjust=False).mean()
    prev = df.close.shift(1)
    tr = pd.concat([(df.high-df.low).abs(), (df.high-prev).abs(), (df.low-prev).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14, min_periods=5).mean()
    dates = pd.Series(df.index.date, index=df.index)
    tp = (df.high + df.low + df.close)/3
    cumv = df.volume.groupby(dates).cumsum().replace(0, math.nan)
    df['vwap'] = (tp*df.volume).groupby(dates).cumsum()/cumv
    df['vol_sma20'] = df.volume.rolling(20, min_periods=5).mean()
    return df


def closed_only(df):
    if df.empty: return df
    now = datetime.now(NY)
    return df[df.index + pd.Timedelta(minutes=4) <= now]


def cluster_levels(values, tolerance):
    vals = sorted(float(v) for v in values if pd.notna(v))
    clusters = []
    for v in vals:
        if not clusters or abs(v - sum(clusters[-1])/len(clusters[-1])) > tolerance:
            clusters.append([v])
        else:
            clusters[-1].append(v)
    return [(sum(c)/len(c), len(c)) for c in clusters]


def levels_from_df(df, price, atr):
    w = df.tail(LEVEL_LOOKBACK)
    lows, highs = [], []
    for i in range(2, len(w)-2):
        lo = float(w.low.iloc[i]); hi = float(w.high.iloc[i])
        if lo <= float(w.low.iloc[i-2:i+3].min()): lows.append(lo)
        if hi >= float(w.high.iloc[i-2:i+3].max()): highs.append(hi)
    tol = max(price*0.0008, atr*0.18, 0.02)
    supports = cluster_levels(lows, tol)
    resistances = cluster_levels(highs, tol)
    below = [(x,n) for x,n in supports if x < price]
    above = [(x,n) for x,n in resistances if x > price]
    support = max(below, default=(None,0), key=lambda z:z[0])
    resistance = min(above, default=(None,0), key=lambda z:z[0])
    return support, resistance, tol


def previous_day_levels(df, today):
    prev = df[df.index.date < today]
    if prev.empty: return None, None
    d = prev.index.date[-1]
    p = prev[prev.index.date == d]
    return float(p.high.max()), float(p.low.min())


def analyze(symbol, raw, meta):
    df = closed_only(add_indicators(to_df(raw)))
    if len(df) < 35: return None
    today = datetime.now(NY).date()
    td = df[df.index.date == today]
    if len(td) < 3: return None
    row, prev = td.iloc[-1], td.iloc[-2]
    price = float(row.close); atr = float(row.atr if pd.notna(row.atr) else max(price*0.003,0.05))
    vwap = float(row.vwap) if pd.notna(row.vwap) else price
    rvol = float(row.volume / row.vol_sma20) if pd.notna(row.vol_sma20) and row.vol_sma20 else 1.0
    pm = td[(td.index.time >= PREMARKET_START) & (td.index.time < PREMARKET_END)]
    pmh = float(pm.high.max()) if not pm.empty else None
    pml = float(pm.low.min()) if not pm.empty else None
    pdh, pdl = previous_day_levels(df, today)

    support, resistance, tol = levels_from_df(df, price, atr)
    s, stouches = support; r, rtouches = resistance
    if pml and pml < price and (s is None or pml > s): s, stouches = pml, 3
    if pdl and pdl < price and (s is None or pdl > s): s, stouches = pdl, max(stouches,2)
    if pmh and pmh > price and (r is None or pmh < r): r, rtouches = pmh, 3
    if pdh and pdh > price and (r is None or pdh < r): r, rtouches = pdh, max(rtouches,2)

    bull = row.ema5 > row.ema9 > row.ema30
    bear = row.ema5 < row.ema9 < row.ema30
    mom = (price - float(td.close.iloc[-3])) / max(atr, 1e-6)
    direction = 'CALL' if (bull and price > vwap) else 'PUT' if (bear and price < vwap) else ('CALL' if mom > 0.35 else 'PUT' if mom < -0.35 else 'NONE')
    if direction == 'NONE': return None

    trigger = r if direction == 'CALL' else s
    if trigger is None: return None
    distance = abs(price-trigger)
    proximity = max(0.0, 1.0 - distance/max(atr*1.25, 0.05))
    target = None
    if direction == 'CALL':
        above = sorted([x for x,_ in cluster_levels(df.tail(LEVEL_LOOKBACK).high.tolist(), tol) if x > max(price, trigger)])
        target = above[0] if above else trigger + max(atr, price*0.002)
    else:
        below = sorted([x for x,_ in cluster_levels(df.tail(LEVEL_LOOKBACK).low.tolist(), tol) if x < min(price, trigger)], reverse=True)
        target = below[0] if below else trigger - max(atr, price*0.002)

    score = 0.0
    if (direction=='CALL' and bull) or (direction=='PUT' and bear): score += 25
    if (direction=='CALL' and price>vwap) or (direction=='PUT' and price<vwap): score += 15
    score += min(max(rvol-0.8,0)/1.7, 1)*15
    score += min(abs(mom)/1.2,1)*10
    score += proximity*20
    touches = rtouches if direction=='CALL' else stouches
    score += min(touches/3,1)*10
    room = abs(target-trigger) if target else 0
    if room >= atr*0.6: score += 5
    score = round(min(score,100),1)

    prev_close = float(prev.close)
    if direction == 'CALL':
        crossed = prev_close <= trigger and price > trigger
        status = 'BREAK_CONFIRMED' if crossed else ('WAITING_FOR_BREAK' if price <= trigger + tol else 'ABOVE_LEVEL')
    else:
        crossed = prev_close >= trigger and price < trigger
        status = 'BREAK_CONFIRMED' if crossed else ('WAITING_FOR_BREAK' if price >= trigger - tol else 'BELOW_LEVEL')

    return {
        'symbol': symbol, 'direction': direction, 'score': score, 'status': status,
        'price': round(price,4), 'trigger': round(trigger,4),
        'support': round(s,4) if s else None, 'resistance': round(r,4) if r else None,
        'target': round(target,4) if target else None, 'ema5': round(float(row.ema5),4),
        'ema9': round(float(row.ema9),4), 'ema30': round(float(row.ema30),4),
        'vwap': round(vwap,4), 'atr': round(atr,4), 'rvol': round(rvol,2),
        'dollar_volume': round(float(meta.get(symbol,{}).get('dollar_volume',0)),2),
        'bar_time': td.index[-1].isoformat(), 'touches': int(touches),
    }


def run_scan():
    with lock: STATE.update(status='SCANNING', error=None)
    symbols = get_universe()
    live, meta = liquid_universe(symbols)
    results = []
    for batch in chunks(live, BAR_BATCH):
        bars = get_batch_bars(batch)
        for s in batch:
            try:
                item = analyze(s, bars.get(s,[]), meta)
                if item and item['score'] >= MIN_SETUP_SCORE: results.append(item)
            except Exception as e:
                logging.warning('%s analyze error: %s', s, e)
        time.sleep(0.05)
    results.sort(key=lambda x:(x['score'], x['dollar_volume']), reverse=True)
    watch = results[:WATCHLIST_SIZE]
    with lock:
        STATE.update(status='READY', last_scan=datetime.now(NY).isoformat(), universe_count=len(symbols),
                     liquid_count=len(live), watchlist_count=len(watch), watchlist=watch, error=None)
    logging.info('scan ready | universe=%s liquid=%s watch=%s', len(symbols), len(live), len(watch))


def loop():
    while True:
        try: run_scan()
        except Exception as e:
            logging.exception('scan failed')
            with lock: STATE.update(status='ERROR', error=str(e)[:500])
        time.sleep(SCAN_SECONDS)

HTML = '''<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="20"><title>AI Trade Watchlist</title>
<style>body{font-family:Arial;background:#0b0f14;color:#e7edf4;margin:18px}.card{background:#141a22;border:1px solid #263241;border-radius:12px;padding:14px;margin:10px 0}.big{font-size:28px;font-weight:800}.muted{color:#9fb0c2}.call{color:#41d17d}.put{color:#ff6464}.score{font-size:22px;font-weight:800}code{color:#cfe4ff}</style></head><body>
<h2>AI Confirmation Watchlist</h2><div class="muted">{{state.status}} · {{state.last_scan}} · liquid {{state.liquid_count}}</div>
{% for x in state.watchlist %}<div class="card"><div class="big">{{x.symbol}} <span class="{{'call' if x.direction=='CALL' else 'put'}}">{{x.direction}}</span> <span class="score">{{x.score}}/100</span></div>
<div>{{x.status}} · price {{x.price}} · trigger <b>{{x.trigger}}</b> · target {{x.target}}</div><div class="muted">S {{x.support}} · R {{x.resistance}} · VWAP {{x.vwap}} · RVOL {{x.rvol}} · touches {{x.touches}}</div></div>{% endfor %}
</body></html>'''

@app.get('/')
def home():
    with lock: snap = dict(STATE); snap['watchlist'] = list(STATE['watchlist'])
    return render_template_string(HTML, state=snap)

@app.get('/api/watchlist')
def watchlist():
    with lock: return jsonify(dict(STATE))

@app.get('/health')
def health():
    return jsonify({'ok':True,'status':STATE['status'],'last_scan':STATE['last_scan']})

if __name__ == '__main__':
    threading.Thread(target=loop, daemon=True).start()
    app.run(host='0.0.0.0', port=int(os.getenv('PORT','10000')))
else:
    threading.Thread(target=loop, daemon=True).start()

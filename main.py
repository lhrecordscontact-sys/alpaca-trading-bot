import os, time, math, threading, logging
from datetime import datetime, timedelta, time as dt_time
from zoneinfo import ZoneInfo

import requests
import pandas as pd
from flask import Flask, jsonify
from ai_confirmation import ask_ai

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
NY, UTC = ZoneInfo('America/New_York'), ZoneInfo('UTC')
TRADING_URL = 'https://paper-api.alpaca.markets'
DATA_URL = 'https://data.alpaca.markets'
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY','').strip()
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY','').strip()
DATA_FEED = os.getenv('DATA_FEED','iex').strip().lower()
OPTION_FEED = os.getenv('OPTION_FEED','opra').strip().lower()
SCANNER_URL = os.getenv('SCANNER_URL','https://nine0-percent-scanner.onrender.com/api/watchlist').strip()
HEADERS = {'APCA-API-KEY-ID':ALPACA_API_KEY,'APCA-API-SECRET-KEY':ALPACA_SECRET_KEY,'Content-Type':'application/json'}

AUTO_TRADE = os.getenv('AUTO_TRADE','false').strip().lower() == 'true'
LOOP_SECONDS = int(os.getenv('LOOP_SECONDS','20'))
MIN_SETUP_SCORE = float(os.getenv('BOT_MIN_SETUP_SCORE','82'))
MAX_OPEN_POSITIONS = int(os.getenv('MAX_OPEN_POSITIONS','3'))
MAX_NEW_TRADES_PER_CYCLE = int(os.getenv('MAX_NEW_TRADES_PER_CYCLE','1'))
MAX_TRADES_PER_SYMBOL_DAY = int(os.getenv('MAX_TRADES_PER_SYMBOL_DAY','2'))
OPTION_QTY = int(os.getenv('OPTION_QTY','1'))
LAST_ENTRY = dt_time.fromisoformat(os.getenv('LAST_ENTRY','14:45'))
FORCE_EXIT = dt_time.fromisoformat(os.getenv('FORCE_EXIT','15:15'))
TIMEFRAME='4Min'; EMA_FAST=5; EMA_SLOW=9; EMA_TREND=30

lock = threading.Lock()
STATE = {'status':'STARTING','last_cycle':None,'watching':[],'last_ai_decision':None,'last_order':None,'errors':[],'auto_trade':AUTO_TRADE}
seen_ai_bars = set(); daily_trade_counts = {}; managed = {}


def log_error(e):
    msg=str(e)[:500]; logging.error(msg)
    with lock: STATE['errors']=(STATE['errors']+[msg])[-20:]


def req(method, path, base=TRADING_URL, params=None, data=None, timeout=30):
    r=requests.request(method, f'{base}{path}', headers=HEADERS, params=params, json=data, timeout=timeout)
    if not r.ok: raise RuntimeError(f'{method} {path} {r.status_code}: {r.text[:500]}')
    return r.json() if r.text else {}


def scanner_watchlist():
    r=requests.get(SCANNER_URL, timeout=30); r.raise_for_status(); p=r.json()
    items=p.get('watchlist') or p.get('qualified') or []
    out=[]
    for x in items:
        try:
            if float(x.get('score',0)) >= MIN_SETUP_SCORE and str(x.get('direction','')).upper() in ('CALL','PUT'):
                out.append(x)
        except Exception: pass
    return out


def get_bars(symbol, hours=12):
    now=datetime.now(UTC); start=now-timedelta(hours=hours)
    p={'timeframe':TIMEFRAME,'start':start.isoformat(),'end':now.isoformat(),'adjustment':'raw','feed':DATA_FEED,'sort':'asc','limit':1000}
    bars=req('GET',f'/v2/stocks/{symbol}/bars',base=DATA_URL,params=p).get('bars') or []
    if not bars:return pd.DataFrame()
    df=pd.DataFrame(bars).rename(columns={'t':'timestamp','o':'open','h':'high','l':'low','c':'close','v':'volume'})
    df['timestamp']=pd.to_datetime(df.timestamp,utc=True); df=df.set_index('timestamp').tz_convert(NY).sort_index()
    for c in ['open','high','low','close','volume']:df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.dropna(); nowet=datetime.now(NY); return df[df.index+pd.Timedelta(minutes=4)<=nowet]


def enrich(df):
    d=df.copy(); d['ema5']=d.close.ewm(span=5,adjust=False).mean(); d['ema9']=d.close.ewm(span=9,adjust=False).mean(); d['ema30']=d.close.ewm(span=30,adjust=False).mean()
    dates=pd.Series(d.index.date,index=d.index); tp=(d.high+d.low+d.close)/3; cv=d.volume.groupby(dates).cumsum().replace(0,math.nan)
    d['vwap']=(tp*d.volume).groupby(dates).cumsum()/cv; d['vol20']=d.volume.rolling(20,min_periods=5).mean(); return d


def hard_confirmation(item, df):
    if len(df)<4:return None
    d=enrich(df); last=d.iloc[-1]; prev=d.iloc[-2]; direction=str(item['direction']).upper(); trigger=float(item['trigger']); price=float(last.close)
    bull=last.ema5>last.ema9 and price>last.vwap; bear=last.ema5<last.ema9 and price<last.vwap
    broke = (price>trigger and float(prev.close)<=trigger and bull) if direction=='CALL' else (price<trigger and float(prev.close)>=trigger and bear)
    already_beyond = (price>trigger and bull) if direction=='CALL' else (price<trigger and bear)
    if not (broke or already_beyond): return None
    candles=[]
    for ts,row in d.tail(6).iterrows():
        candles.append({'time':ts.isoformat(),'o':round(float(row.open),4),'h':round(float(row.high),4),'l':round(float(row.low),4),'c':round(float(row.close),4),'v':int(row.volume)})
    return {
        'symbol':item['symbol'],'direction':direction,'scanner_score':item.get('score'),'scanner_status':item.get('status'),
        'trigger':trigger,'support':item.get('support'),'resistance':item.get('resistance'),'scanner_target':item.get('target'),
        'price':round(price,4),'ema5':round(float(last.ema5),4),'ema9':round(float(last.ema9),4),'ema30':round(float(last.ema30),4),
        'vwap':round(float(last.vwap),4),'volume_ratio':round(float(last.volume/last.vol20),2) if pd.notna(last.vol20) and last.vol20 else 1.0,
        'bar_time':d.index[-1].isoformat(),'recent_candles':candles,
        'rule_note':'The latest bar is closed. Decide ENTER only if the breakout remains valid and reward to next level is adequate.'
    }


def positions():
    p=req('GET','/v2/positions'); return p if isinstance(p,list) else []


def underlying_open(symbol):
    for p in positions():
        if str(p.get('symbol','')).startswith(symbol): return True
    return False


def option_contract(symbol,direction,price):
    today=datetime.now(NY).date().isoformat(); typ='call' if direction=='CALL' else 'put'
    params={'underlying_symbols':symbol,'status':'active','type':typ,'expiration_date':today,'limit':1000}
    data=req('GET','/v2/options/contracts',params=params)
    contracts=data.get('option_contracts') or data.get('contracts') or []
    if not contracts: return None
    def strike(c):
        try:return float(c.get('strike_price') or 0)
        except:return 1e12
    contracts=[c for c in contracts if strike(c)>0]
    if not contracts:return None
    return min(contracts,key=lambda c:abs(strike(c)-price))


def submit_option(symbol,direction,price,decision):
    contract=option_contract(symbol,direction,price)
    if not contract: raise RuntimeError(f'No same-day {direction} option contract for {symbol}')
    opt=contract.get('symbol') or contract.get('id')
    if not opt: raise RuntimeError('Option contract missing symbol')
    order=req('POST','/v2/orders',data={'symbol':opt,'qty':str(max(1,OPTION_QTY)),'side':'buy','type':'market','time_in_force':'day'})
    managed[opt]={'underlying':symbol,'direction':direction,'trigger':decision.get('entry') or price,'stop':decision.get('stop'),'tp1':decision.get('tp1'),'tp2':decision.get('tp2'),'tp1_done':False}
    return order


def close_option(opt, qty=None):
    if qty is None:
        return req('DELETE',f'/v2/positions/{opt}')
    return req('POST','/v2/orders',data={'symbol':opt,'qty':str(qty),'side':'sell','type':'market','time_in_force':'day'})


def latest_stock_price(symbol):
    d=get_bars(symbol,4); return float(d.close.iloc[-1]) if not d.empty else None


def manage_positions():
    now=datetime.now(NY)
    for opt,info in list(managed.items()):
        try:
            pos=next((p for p in positions() if p.get('symbol')==opt),None)
            if not pos: managed.pop(opt,None); continue
            if now.time()>=FORCE_EXIT: close_option(opt); managed.pop(opt,None); continue
            px=latest_stock_price(info['underlying']);
            if px is None: continue
            qty=max(1,int(float(pos.get('qty') or 1))); d=info['direction']; stop=info.get('stop'); tp1=info.get('tp1'); tp2=info.get('tp2')
            stop_hit=stop is not None and ((d=='CALL' and px<=float(stop)) or (d=='PUT' and px>=float(stop)))
            tp2_hit=tp2 is not None and ((d=='CALL' and px>=float(tp2)) or (d=='PUT' and px<=float(tp2)))
            tp1_hit=tp1 is not None and ((d=='CALL' and px>=float(tp1)) or (d=='PUT' and px<=float(tp1)))
            if stop_hit or tp2_hit: close_option(opt); managed.pop(opt,None)
            elif tp1_hit and not info['tp1_done']:
                sell=max(1,qty//2) if qty>1 else qty; close_option(opt,sell); info['tp1_done']=True
                if sell>=qty: managed.pop(opt,None)
        except Exception as e: log_error(f'manage {opt}: {e}')


def cycle():
    now=datetime.now(NY); manage_positions()
    if now.time()>=LAST_ENTRY:
        with lock: STATE.update(status='NO_NEW_ENTRIES',last_cycle=now.isoformat())
        return
    watch=scanner_watchlist(); withhold=[]; new_trades=0
    for item in watch:
        if new_trades>=MAX_NEW_TRADES_PER_CYCLE or len(positions())>=MAX_OPEN_POSITIONS: break
        symbol=str(item.get('symbol','')).upper(); keyday=(symbol,now.date().isoformat())
        if daily_trade_counts.get(keyday,0)>=MAX_TRADES_PER_SYMBOL_DAY or underlying_open(symbol): continue
        df=get_bars(symbol); setup=hard_confirmation(item,df)
        if not setup: continue
        ai_key=(symbol,setup['direction'],setup['bar_time'])
        if ai_key in seen_ai_bars: continue
        seen_ai_bars.add(ai_key); decision=ask_ai(setup)
        with lock: STATE['last_ai_decision']={'setup':setup,'decision':decision,'time':now.isoformat()}
        logging.info('AI %s %s -> %s %.2f | %s',symbol,setup['direction'],decision.get('decision'),decision.get('confidence',0),decision.get('reason'))
        if decision.get('decision')!='ENTER': continue
        if not AUTO_TRADE:
            withhold.append({'symbol':symbol,'direction':setup['direction'],'reason':'AUTO_TRADE=false','decision':decision}); continue
        order=submit_option(symbol,setup['direction'],setup['price'],decision); daily_trade_counts[keyday]=daily_trade_counts.get(keyday,0)+1; new_trades+=1
        with lock: STATE['last_order']=order
    with lock: STATE.update(status='RUNNING',last_cycle=now.isoformat(),watching=watch)


def loop():
    while True:
        try: cycle()
        except Exception as e: log_error(e)
        time.sleep(LOOP_SECONDS)

@app.get('/')
def home():
    with lock:return jsonify(STATE)
@app.get('/health')
def health():
    return jsonify({'ok':True,'status':STATE['status'],'auto_trade':AUTO_TRADE,'last_cycle':STATE['last_cycle']})

if __name__=='__main__':
    threading.Thread(target=loop,daemon=True).start(); app.run(host='0.0.0.0',port=int(os.getenv('PORT','10000')))
else:
    threading.Thread(target=loop,daemon=True).start()

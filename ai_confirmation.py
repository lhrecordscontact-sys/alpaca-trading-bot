import os, json, requests

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '').strip()
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-5.6-luna').strip()
AI_MIN_CONFIDENCE = float(os.getenv('AI_MIN_CONFIDENCE', '0.72'))

SYSTEM = '''You are a trade-confirmation gate for a PAPER options bot. You do not search for trades; you only judge a supplied 4-minute setup.
Return strict JSON only. Use decision ENTER, WAIT, or CANCEL. Prefer WAIT when confirmation is incomplete. Never override hard risk rules.
For CALL: require a confirmed close above the trigger/support-resistance level and evidence the level is holding or continuation is present.
For PUT: require a confirmed close below the trigger/support-resistance level and evidence the level is holding as resistance or continuation is present.
Avoid entries directly into the next support/resistance when reward is too small. If the breakout was immediately reclaimed, WAIT or CANCEL.
Use the supplied trigger, support, resistance, target, EMA5/9/30, VWAP, volume and recent candles.''' 


def _extract_text(payload):
    if isinstance(payload, dict):
        if payload.get('output_text'):
            return payload['output_text']
        for item in payload.get('output', []):
            if item.get('type') == 'message':
                for c in item.get('content', []):
                    if c.get('type') in ('output_text','text') and c.get('text'):
                        return c['text']
    return ''


def ask_ai(setup):
    if not OPENAI_API_KEY:
        return {'decision':'WAIT','confidence':0.0,'reason':'OPENAI_API_KEY missing','entry':None,'stop':None,'tp1':None,'tp2':None}

    prompt = {
        'task':'Decide whether this already-detected setup is ready to enter now.',
        'allowed_decisions':['ENTER','WAIT','CANCEL'],
        'setup':setup,
        'required_output':{
            'decision':'ENTER|WAIT|CANCEL','confidence':'0.0-1.0','entry':'number|null','stop':'number|null',
            'tp1':'number|null','tp2':'number|null','reason':'short sentence'
        }
    }
    body = {
        'model': OPENAI_MODEL,
        'input': [
            {'role':'system','content':SYSTEM},
            {'role':'user','content':json.dumps(prompt, separators=(',',':'))},
        ],
        'max_output_tokens': 250,
    }
    r = requests.post('https://api.openai.com/v1/responses', headers={
        'Authorization':f'Bearer {OPENAI_API_KEY}','Content-Type':'application/json'
    }, json=body, timeout=45)
    if not r.ok:
        return {'decision':'WAIT','confidence':0.0,'reason':f'AI HTTP {r.status_code}','entry':None,'stop':None,'tp1':None,'tp2':None}
    text = _extract_text(r.json()).strip()
    if text.startswith('```'):
        text = text.strip('`').replace('json\n','',1)
    try:
        out = json.loads(text)
    except Exception:
        return {'decision':'WAIT','confidence':0.0,'reason':'AI returned invalid JSON','entry':None,'stop':None,'tp1':None,'tp2':None}
    decision = str(out.get('decision','WAIT')).upper()
    confidence = float(out.get('confidence') or 0)
    if decision not in ('ENTER','WAIT','CANCEL'):
        decision = 'WAIT'
    if decision == 'ENTER' and confidence < AI_MIN_CONFIDENCE:
        decision = 'WAIT'
        out['reason'] = f"Below AI confidence gate ({confidence:.2f} < {AI_MIN_CONFIDENCE:.2f})"
    out['decision'] = decision
    out['confidence'] = confidence
    return out

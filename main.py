//@version=6
indicator("SPY + IWM Premarket Break + Win Rate Stats + ALPACA WEBHOOK", overlay=true, max_labels_count=500, max_lines_count=500)

//────────────────────────────────────
// INPUTS
//────────────────────────────────────
showEMAs      = input.bool(true, "Show EMAs")
showVWAP      = input.bool(true, "Show VWAP")
showPMLevels  = input.bool(true, "Show Premarket High/Low")
showSignals   = input.bool(true, "Show SELL / BUY / EXIT")
showStats     = input.bool(true, "Show Large Stats Box")

pmSession     = input.session("0400-0930", "Premarket Session")
rthSession    = input.session("0930-1600", "Regular Session")

emaFastLen    = input.int(5, "Fast EMA")
emaSlowLen    = input.int(9, "Slow EMA")
emaTrendLen   = input.int(30, "Trend EMA")

spyTarget     = input.float(1.00, "SPY Target Move ($)", step=0.05)
iwmTarget     = input.float(0.50, "IWM Target Move ($)", step=0.05)

useTrend      = input.bool(true, "Require EMA Trend Filter")
useVWAPFilter = input.bool(true, "Require VWAP Filter")
maxTradesDay  = input.int(2, "Max Trades Per Day", minval=1, maxval=10)

//────────────────────────────────────
// SYMBOL DETECTION
//────────────────────────────────────
isSPY = str.contains(syminfo.ticker, "SPY")
isIWM = str.contains(syminfo.ticker, "IWM")

targetMove = isIWM ? iwmTarget : spyTarget

//────────────────────────────────────
// SESSION DETECTION
//────────────────────────────────────
inPM  = not na(time(timeframe.period, pmSession, "America/New_York"))
inRTH = not na(time(timeframe.period, rthSession, "America/New_York"))

newPM  = inPM and not inPM[1]
newDay = ta.change(time("D")) != 0

//────────────────────────────────────
// PREMARKET HIGH / LOW
//────────────────────────────────────
var float pmHigh = na
var float pmLow  = na

if newDay
    pmHigh := na
    pmLow  := na

if newPM
    pmHigh := high
    pmLow  := low
else if inPM
    pmHigh := na(pmHigh) ? high : math.max(pmHigh, high)
    pmLow  := na(pmLow) ? low : math.min(pmLow, low)

//────────────────────────────────────
// INDICATORS
//────────────────────────────────────
ema5  = ta.ema(close, emaFastLen)
ema9  = ta.ema(close, emaSlowLen)
ema30 = ta.ema(close, emaTrendLen)
vwap  = ta.vwap(hlc3)

plot(showEMAs ? ema5 : na, "EMA 5", color=color.green, linewidth=2)
plot(showEMAs ? ema9 : na, "EMA 9", color=color.blue, linewidth=2)
plot(showEMAs ? ema30 : na, "EMA 30", color=color.orange, linewidth=2)
plot(showVWAP ? vwap : na, "VWAP", color=color.purple, linewidth=3)

plot(showPMLevels ? pmHigh : na, "Premarket High", color=color.green, linewidth=2, style=plot.style_linebr)
plot(showPMLevels ? pmLow : na, "Premarket Low", color=color.red, linewidth=2, style=plot.style_linebr)

//────────────────────────────────────
// TRADE FILTERS
//────────────────────────────────────
bullTrend = ema5 > ema9 and ema9 > ema30
bearTrend = ema5 < ema9 and ema9 < ema30

bullVWAP = close > vwap
bearVWAP = close < vwap

longFilters =
     (not useTrend or bullTrend) and
     (not useVWAPFilter or bullVWAP)

shortFilters =
     (not useTrend or bearTrend) and
     (not useVWAPFilter or bearVWAP)

//────────────────────────────────────
// BREAK SIGNALS
//────────────────────────────────────
longBreak =
     inRTH and
     not na(pmHigh) and
     close > pmHigh and
     close[1] <= pmHigh and
     longFilters

shortBreak =
     inRTH and
     not na(pmLow) and
     close < pmLow and
     close[1] >= pmLow and
     shortFilters

//────────────────────────────────────
// TRADE MANAGEMENT
//────────────────────────────────────
var bool  inTrade       = false
var bool  longTrade     = false
var float entryPrice    = na
var float targetPrice   = na
var float stopPrice     = na
var int   tradesToday   = 0

var int wins            = 0
var int losses          = 0
var int totalTrades     = 0
var float totalMove     = 0.0

var int callWins        = 0
var int callLosses      = 0
var int callTrades      = 0
var float callMove      = 0.0

var int putWins         = 0
var int putLosses       = 0
var int putTrades       = 0
var float putMove       = 0.0

if newDay
    tradesToday := 0

canTrade = not inTrade and tradesToday < maxTradesDay

// CONFIRMED 4-MINUTE CANDLE SIGNALS
longEntry  = longBreak and canTrade and barstate.isconfirmed
shortEntry = shortBreak and canTrade and barstate.isconfirmed

//────────────────────────────────────
// WEBHOOK MESSAGE BUILDERS
//────────────────────────────────────
makeMessage(action, direction, signalPrice) =>
    "{" +
    "\"source\":\"TRADINGVIEW\"," +
    "\"action\":\"" + action + "\"," +
    "\"direction\":\"" + direction + "\"," +
    "\"symbol\":\"" + syminfo.ticker + "\"," +
    "\"timeframe\":\"" + timeframe.period + "\"," +
    "\"price\":" + str.tostring(signalPrice) + "," +
    "\"ema5\":" + str.tostring(ema5) + "," +
    "\"ema9\":" + str.tostring(ema9) + "," +
    "\"ema30\":" + str.tostring(ema30) + "," +
    "\"vwap\":" + str.tostring(vwap) + "," +
    "\"pmHigh\":" + str.tostring(pmHigh) + "," +
    "\"pmLow\":" + str.tostring(pmLow) + "," +
    "\"barTime\":" + str.tostring(time) +
    "}"

//────────────────────────────────────
// ENTRIES
//────────────────────────────────────
if longEntry
    inTrade     := true
    longTrade   := true
    entryPrice  := close
    targetPrice := close + targetMove
    stopPrice   := ema9
    tradesToday += 1

    // EXACT SAME EVENT SENT TO ALPACA BOT
    alert(
         makeMessage("ENTRY", "CALL", close),
         alert.freq_once_per_bar_close)

    if showSignals
        label.new(
             bar_index,
             low,
             "BUY\n" + str.tostring(entryPrice, format.mintick),
             style=label.style_label_up,
             color=color.green,
             textcolor=color.white,
             size=size.normal)

if shortEntry
    inTrade     := true
    longTrade   := false
    entryPrice  := close
    targetPrice := close - targetMove
    stopPrice   := ema9
    tradesToday += 1

    // EXACT SAME EVENT SENT TO ALPACA BOT
    alert(
         makeMessage("ENTRY", "PUT", close),
         alert.freq_once_per_bar_close)

    if showSignals
        label.new(
             bar_index,
             high,
             "SELL\n" + str.tostring(entryPrice, format.mintick),
             style=label.style_label_down,
             color=color.red,
             textcolor=color.white,
             size=size.normal)

//────────────────────────────────────
// EXIT LOGIC
//────────────────────────────────────
longTP  = inTrade and longTrade and high >= targetPrice
shortTP = inTrade and not longTrade and low <= targetPrice

// EMA9 EXIT MUST BE CONFIRMED AT CANDLE CLOSE
longStop =
     inTrade and
     longTrade and
     close <= ema9 and
     barstate.isconfirmed

shortStop =
     inTrade and
     not longTrade and
     close >= ema9 and
     barstate.isconfirmed

if longTP
    profitMove = targetPrice - entryPrice

    wins        += 1
    totalTrades += 1
    totalMove   += profitMove

    callWins   += 1
    callTrades += 1
    callMove   += profitMove

    alert(
         makeMessage("TAKE_PROFIT", "CALL", targetPrice),
         alert.freq_once_per_bar_close)

    if showSignals
        label.new(
             bar_index,
             high,
             "TP\n+$" + str.tostring(profitMove, "#.##"),
             style=label.style_label_down,
             color=color.green,
             textcolor=color.white,
             size=size.normal)

    inTrade     := false
    entryPrice  := na
    targetPrice := na
    stopPrice   := na

else if shortTP
    profitMove = entryPrice - targetPrice

    wins        += 1
    totalTrades += 1
    totalMove   += profitMove

    putWins   += 1
    putTrades += 1
    putMove   += profitMove

    alert(
         makeMessage("TAKE_PROFIT", "PUT", targetPrice),
         alert.freq_once_per_bar_close)

    if showSignals
        label.new(
             bar_index,
             low,
             "TP\n+$" + str.tostring(profitMove, "#.##"),
             style=label.style_label_up,
             color=color.green,
             textcolor=color.white,
             size=size.normal)

    inTrade     := false
    entryPrice  := na
    targetPrice := na
    stopPrice   := na

else if longStop
    exitMove = close - entryPrice

    if exitMove > 0
        wins     += 1
        callWins += 1
    else
        losses     += 1
        callLosses += 1

    totalTrades += 1
    totalMove   += exitMove

    callTrades += 1
    callMove   += exitMove

    pctMove = entryPrice != 0 ? (exitMove / entryPrice) * 100 : 0

    alert(
         makeMessage("EXIT", "CALL", close),
         alert.freq_once_per_bar_close)

    if showSignals
        label.new(
             bar_index,
             low,
             "EXIT\n" +
             (exitMove >= 0 ? "+" : "") +
             "$" + str.tostring(exitMove, "#.##") +
             "\n" +
             (pctMove >= 0 ? "+" : "") +
             str.tostring(pctMove, "#.##") + "%",
             style=label.style_label_up,
             color=exitMove >= 0 ? color.green : color.red,
             textcolor=color.white,
             size=size.normal)

    inTrade     := false
    entryPrice  := na
    targetPrice := na
    stopPrice   := na

else if shortStop
    exitMove = entryPrice - close

    if exitMove > 0
        wins    += 1
        putWins += 1
    else
        losses    += 1
        putLosses += 1

    totalTrades += 1
    totalMove   += exitMove

    putTrades += 1
    putMove   += exitMove

    pctMove = entryPrice != 0 ? (exitMove / entryPrice) * 100 : 0

    alert(
         makeMessage("EXIT", "PUT", close),
         alert.freq_once_per_bar_close)

    if showSignals
        label.new(
             bar_index,
             high,
             "EXIT\n" +
             (exitMove >= 0 ? "+" : "") +
             "$" + str.tostring(exitMove, "#.##") +
             "\n" +
             (pctMove >= 0 ? "+" : "") +
             str.tostring(pctMove, "#.##") + "%",
             style=label.style_label_down,
             color=exitMove >= 0 ? color.green : color.red,
             textcolor=color.white,
             size=size.normal)

    inTrade     := false
    entryPrice  := na
    targetPrice := na
    stopPrice   := na

//────────────────────────────────────
// ACTIVE TARGET
//────────────────────────────────────
plot(
     inTrade ? targetPrice : na,
     "Active Target",
     color=color.yellow,
     linewidth=2,
     style=plot.style_linebr)

//────────────────────────────────────
// WIN RATES
//────────────────────────────────────
winRate     = totalTrades > 0 ? (wins * 100.0) / totalTrades : 0.0
callWinRate = callTrades > 0 ? (callWins * 100.0) / callTrades : 0.0
putWinRate  = putTrades > 0 ? (putWins * 100.0) / putTrades : 0.0

//────────────────────────────────────
// LARGE STATS TABLE
//────────────────────────────────────
var table stats = table.new(
     position.top_right,
     2,
     13,
     border_width=2,
     frame_width=2)

if barstate.islast
    table.clear(stats, 0, 0, 1, 12)

    if showStats
        table.cell(stats, 0, 0, "PREMARKET STATS",
             bgcolor=color.rgb(20,20,20), text_color=color.white, text_size=size.large)

        table.cell(stats, 1, 0,
             isSPY ? "SPY" : isIWM ? "IWM" : syminfo.ticker,
             bgcolor=color.rgb(20,20,20), text_color=color.yellow, text_size=size.large)

        table.cell(stats, 0, 1, "OVERALL WIN RATE",
             bgcolor=color.rgb(30,30,30), text_color=color.white, text_size=size.large)

        table.cell(stats, 1, 1,
             str.tostring(winRate, "#.0") + "%",
             bgcolor=winRate >= 60 ? color.green : winRate >= 50 ? color.orange : color.red,
             text_color=color.white, text_size=size.large)

        table.cell(stats, 0, 2, "CALL WIN RATE",
             bgcolor=color.rgb(30,30,30), text_color=color.white)

        table.cell(stats, 1, 2,
             str.tostring(callWinRate, "#.0") + "%",
             bgcolor=callWinRate >= 60 ? color.green : callWinRate >= 50 ? color.orange : color.red,
             text_color=color.white)

        table.cell(stats, 0, 3, "PUT WIN RATE",
             bgcolor=color.rgb(30,30,30), text_color=color.white)

        table.cell(stats, 1, 3,
             str.tostring(putWinRate, "#.0") + "%",
             bgcolor=putWinRate >= 60 ? color.green : putWinRate >= 50 ? color.orange : color.red,
             text_color=color.white)

        table.cell(stats, 0, 4, "CALL W / L",
             bgcolor=color.rgb(30,30,30), text_color=color.white)

        table.cell(stats, 1, 4,
             str.tostring(callWins) + " / " + str.tostring(callLosses),
             bgcolor=color.rgb(50,50,50), text_color=color.white)

        table.cell(stats, 0, 5, "PUT W / L",
             bgcolor=color.rgb(30,30,30), text_color=color.white)

        table.cell(stats, 1, 5,
             str.tostring(putWins) + " / " + str.tostring(putLosses),
             bgcolor=color.rgb(50,50,50), text_color=color.white)

        table.cell(stats, 0, 6, "CALL TRADES",
             bgcolor=color.rgb(30,30,30), text_color=color.white)

        table.cell(stats, 1, 6,
             str.tostring(callTrades),
             bgcolor=color.rgb(50,50,50), text_color=color.white)

        table.cell(stats, 0, 7, "PUT TRADES",
             bgcolor=color.rgb(30,30,30), text_color=color.white)

        table.cell(stats, 1, 7,
             str.tostring(putTrades),
             bgcolor=color.rgb(50,50,50), text_color=color.white)

        table.cell(stats, 0, 8, "WINS",
             bgcolor=color.rgb(30,30,30), text_color=color.white)

        table.cell(stats, 1, 8,
             str.tostring(wins),
             bgcolor=color.green, text_color=color.white)

        table.cell(stats, 0, 9, "LOSSES",
             bgcolor=color.rgb(30,30,30), text_color=color.white)

        table.cell(stats, 1, 9,
             str.tostring(losses),
             bgcolor=color.red, text_color=color.white)

        table.cell(stats, 0, 10, "TOTAL TRADES",
             bgcolor=color.rgb(30,30,30), text_color=color.white)

        table.cell(stats, 1, 10,
             str.tostring(totalTrades),
             bgcolor=color.rgb(50,50,50), text_color=color.white)

        table.cell(stats, 0, 11, "NET MOVE",
             bgcolor=color.rgb(30,30,30), text_color=color.white)

        table.cell(stats, 1, 11,
             (totalMove >= 0 ? "+" : "") +
             "$" + str.tostring(totalMove, "#.##"),
             bgcolor=totalMove >= 0 ? color.green : color.red,
             text_color=color.white)

        table.cell(stats, 0, 12, "STATUS",
             bgcolor=color.rgb(30,30,30), text_color=color.white)

        table.cell(stats, 1, 12,
             inTrade ? (longTrade ? "CALL ACTIVE" : "PUT ACTIVE") : "WAITING",
             bgcolor=inTrade ? color.orange : color.rgb(50,50,50),
             text_color=color.white)

//────────────────────────────────────
// ALERT CONDITIONS
//────────────────────────────────────
// These remain available for normal TradingView notifications.
// Alpaca automation will use the alert() calls above.
alertcondition(longEntry, "Premarket CALL", "Premarket high breakout CALL signal")
alertcondition(shortEntry, "Premarket PUT", "Premarket low breakdown PUT signal")
alertcondition(longTP or shortTP, "Take Profit", "Take-profit target reached")
alertcondition(longStop or shortStop, "Exit", "EMA 9 exit triggered")
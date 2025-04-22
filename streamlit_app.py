import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf
import warnings
import traceback
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from io import BytesIO

warnings.filterwarnings("ignore", message="Series.__getitem__", category=FutureWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Ticker mapping for indices and display names
ticker_mapping = {
    "nifty": "^NSEI",
    "banknifty": "^NSEBANK",
    "sensex": "^BSESN",
    "^NSEI": "^NSEI",
    "^NSEBANK": "^NSEBANK",
    "^BSESN": "^BSESN"
}

# Reverse mapping for display
display_mapping = {
    "^NSEI": "Nifty",
    "^NSEBANK": "Banknifty",
    "^BSESN": "Sensex"
}
display_mapping.update({ticker: ticker for ticker in [
    'AARTIIND','ABB','ABBOTINDIA','ABCAPITAL','ABFRL','ACC','ADANIENSOL','ADANIENT','ADANIGREEN','ADANIPORTS',
    'ALKEM','AMBUJACEM','ANGELONE','APLAPOLLO','APOLLOHOSP','APOLLOTYRE','ASHOKLEY','ASIANPAINT','ASTRAL','ATGL','ATUL','AUBANK','AUROPHARMA',
    'AXISBANK','BAJAJ-AUTO','BAJAJFINSV','BAJFINANCE','BALKRISIND','BANDHANBNK','BANKBARODA','BANKINDIA','BATAINDIA','BEL','BERGEPAINT',
    'BHARATFORG','BHARTIARTL','BHEL','BIOCON','BOSCHLTD','BPCL','BRITANNIA','BSE','BSOFT','CAMS','CANBK','CANFINHOME','CDSL','CESC',
    'CGPOWER','CHAMBLFERT','CHOLAFIN','CIPLA','COALINDIA','COFORGE','COLPAL','CONCOR','COROMANDEL','CROMPTON','CUB','CUMMINSIND','CYIENT',
    'DABUR','DALBHARAT','DEEPAKNTR','DELHIVERY','DIVISLAB','DIXON','DLF','DMART','DRREDDY','EICHERMOT','ESCORTS','EXIDEIND','FEDERALBNK',
    'GAIL','GLENMARK','GMRAIRPORT','GNFC','GODREJCP','GODREJPROP','GRANULES','GRASIM','GUJGASLTD','HAL','HAVELLS','HCLTECH','HDFCAMC',
    'HDFCBANK','HDFCLIFE','HEROMOTOCO','HFCL','HINDALCO','HINDCOPPER','HINDPETRO','HINDUNILVR','HUDCO','ICICIBANK','ICICIGI','ICICIPRULI',
    'IDEA','IDFCFIRSTB','IEX','IGL','INDHOTEL','INDIAMART','INDIANB','INDIGO','INDUSINDBK','INDUSTOWER','INFY','IOC','IPCALAB','IRB',
    'IRCTC','IRFC','ITC','JINDALSTEL','JKCEMENT','JSL','JSWENERGY','JSWSTEEL','JUBLFOOD','KALYANKJIL','KEI','KOTAKBANK',
    'KPITTECH','LALPATHLAB','LAURUSLABS','LICHSGFIN','LICI','LODHA','LT','LTF','LTIM','LTTS','LUPIN','M&M','M&MFIN','MANAPPURAM',
    'MARICO','MARUTI','MAXHEALTH','MCX','METROPOLIS','MFSL','MGL','MOTHERSON','MPHASIS','MRF','MUTHOOTFIN','NATIONALUM','NAUKRI',
    'NAVINFLUOR','NCC','NESTLEIND','NHPC','NMDC','NTPC','NYKAA','OBEROIRLTY','OFSS','OIL','ONGC','PAGEIND','PAYTM','PEL','PERSISTENT',
    'PETRONET','PFC','PIDILITIND','PIIND','PNB','POLICYBZR','POLYCAB','POONAWALLA','POWERGRID','PRESTIGE','PVRINOX','RAMCOCEM','RBLBANK',
    'RECLTD','RELIANCE','SAIL','SBICARD','SBILIFE','SBIN','SHREECEM','SHRIRAMFIN','SIEMENS','SJVN','SONACOMS','SRF','SUNPHARMA','SUNTV',
    'SUPREMEIND','SYNGENE','TATACHEM','TATACOMM','TATACONSUM','TATAELXSI','TATAMOTORS','TATAPOWER','TATASTEEL','TCS','TECHM','TIINDIA','TITAN',
    'TORNTPHARM','TRENT','TVSMOTOR','UBL','ULTRACEMCO','UNIONBANK','UNITDSPR','UPL','VBL','VEDL','VOLTAS','WIPRO','YESBANK','ZOMATO',
    'ZYDUSLIFE'
]})

# Fetch and process data with EMA and RSI
def fetch_and_process_data(tickers, period, interval, use_ema, ema_period, use_rsi, trade_log):
    all_data = {}
    for ticker in tickers:
        try:
            mapped_ticker = ticker_mapping.get(ticker.lower(), ticker)
            if not mapped_ticker.startswith('^'):
                mapped_ticker = f"{ticker.upper()}.NS"
            trade_log.append(f"Fetching data for {ticker} ({mapped_ticker}, Period: {period}, Interval: {interval})")
            data = yf.download(mapped_ticker, period=period, interval=interval)
            if data.empty:
                trade_log.append(f"No data returned for {ticker} ({mapped_ticker})")
                continue
            data.reset_index(inplace=True)
            data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
            data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            data = data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
            data.set_index('Date', inplace=True)
            trade_log.append(f"Fetched {len(data)} data points for {ticker}")
            if use_ema:
                data['ema'] = data['close'].ewm(span=ema_period, adjust=False).mean()
            if use_rsi:
                data['rsi'] = compute_rsi(data['close'], 14)
            data.dropna(inplace=True)
            trade_log.append(f"After processing, {len(data)} data points remain for {ticker}")
            all_data[ticker] = data
        except Exception as e:
            trade_log.append(f"Error fetching data for {ticker}: {str(e)}")
    return all_data

def compute_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def identify_zones(df):
    zones = []
    for i in range(1, len(df) - 1):
        if df['low'].iloc[i] < df['low'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i+1]:
            zones.append({'date': df.index[i], 'type': 'demand', 'level': df['low'].iloc[i]})
        if df['high'].iloc[i] > df['high'].iloc[i-1] and df['high'].iloc[i] > df['high'].iloc[i+1]:
            zones.append({'date': df.index[i], 'type': 'supply', 'level': df['high'].iloc[i]})
    return zones

def identify_super_zones(ticker, trade_log):
    super_zones = []
    timeframe_configs = [
        {'period': '1y', 'interval': '1wk'},
        {'period': '6mo', 'interval': '1wk'},
        {'period': '6mo', 'interval': '1d'},
        {'period': '3mo', 'interval': '1d'}
    ]
    
    all_zones = []
    mapped_ticker = ticker_mapping.get(ticker.lower(), ticker)
    trade_log.append(f"Identifying super zones for {ticker} (mapped to {mapped_ticker})")
    for config in timeframe_configs:
        period = config['period']
        interval = config['interval']
        data = fetch_and_process_data([mapped_ticker], period, interval, False, 20, False, trade_log)
        if mapped_ticker in data:
            zones = identify_zones(data[mapped_ticker])
            for zone in zones:
                zone['period'] = period
                zone['interval'] = interval
            all_zones.extend(zones)
        else:
            trade_log.append(f"No data for {mapped_ticker} at {period}/{interval}")
    
    demand_zones = [z for z in all_zones if z['type'] == 'demand']
    supply_zones = [z for z in all_zones if z['type'] == 'supply']
    
    for zone_type in ['demand', 'supply']:
        zones = demand_zones if zone_type == 'demand' else supply_zones
        i = 0
        while i < len(zones):
            cluster = [zones[i]]
            j = i + 1
            while j < len(zones):
                avg_level = np.mean([z['level'] for z in cluster])
                if abs(zones[j]['level'] - avg_level) <= avg_level * 0.01:
                    cluster.append(zones[j])
                    zones.pop(j)
                else:
                    j += 1
            intervals = set(z['interval'] for z in cluster)
            has_weekly = '1wk' in intervals
            has_daily = '1d' in intervals
            if has_weekly and has_daily:
                avg_level = np.mean([z['level'] for z in cluster])
                super_zones.append({
                    'date': min(z['date'] for z in cluster),
                    'type': zone_type,
                    'level': avg_level,
                    'periods': list(set(z['period'] for z in cluster)),
                    'intervals': list(intervals)
                })
            i += 1
    
    trade_log.append(f"Found {len(super_zones)} super zones for {ticker}")
    return super_zones

def calculate_super_zone_probability(ticker, super_zones, trade_log):
    probabilities = []
    mapped_ticker = ticker_mapping.get(ticker.lower(), ticker)
    for sz in super_zones:
        zone_level = sz['level']
        zone_type = sz['type']
        data = fetch_and_process_data([mapped_ticker], 'max', '1d', True, 20, True, trade_log)
        if mapped_ticker not in data:
            continue
        df = data[mapped_ticker]
        instances = []
        future_data = df[df.index > sz['date']]
        if zone_type == 'demand':
            approaches = future_data[(future_data['low'] >= zone_level * 0.99) & (future_data['low'] <= zone_level * 1.01)]
        else:
            approaches = future_data[(future_data['high'] >= zone_level * 0.99) & (future_data['high'] <= zone_level * 1.01)]
        for approach_date in approaches.index:
            approach_price = df.loc[approach_date, 'close']
            post_approach = df[df.index > approach_date]
            if zone_type == 'demand':
                break_level = zone_level * 0.995
                target_level = approach_price * 1.02
                hit_break = (post_approach['low'] <= break_level).any()
                hit_target = (post_approach['high'] >= target_level).any()
            else:
                break_level = zone_level * 1.005
                target_level = approach_price * 0.98
                hit_break = (post_approach['high'] >= break_level).any()
                hit_target = (post_approach['low'] <= target_level).any()
            if hit_break and hit_target:
                break_idx = post_approach[post_approach['low'] <= break_level].index[0] if zone_type == 'demand' else post_approach[post_approach['high'] >= break_level].index[0]
                target_idx = post_approach[post_approach['high'] >= target_level].index[0] if zone_type == 'demand' else post_approach[post_approach['low'] <= target_level].index[0]
                outcome = 'held' if target_idx < break_idx else 'broke'
            elif hit_target:
                outcome = 'held'
            elif hit_break:
                outcome = 'broke'
            else:
                continue
            instances.append({'outcome': outcome})
        held_count = sum(1 for inst in instances if inst['outcome'] == 'held')
        total = len(instances)
        prob_held = (held_count / total * 100) if total > 0 else 0
        probabilities.append({
            'level': zone_level,
            'type': zone_type,
            'probability_held': prob_held,
            'approaches': total
        })
    return probabilities

def find_approaches_and_labels(df, zones, use_ema, use_rsi):
    instances = []
    for zone in zones:
        limit_price = zone['level']
        base_time = zone['date']
        side = 'BUY' if zone['type'] == 'demand' else 'SELL'
        future_data = df[df.index > base_time]
        if side == 'BUY':
            approaches = future_data[(future_data['low'] >= limit_price * 0.99) & (future_data['low'] <= limit_price * 1.01)]
        else:
            approaches = future_data[(future_data['high'] >= limit_price * 0.99) & (future_data['high'] <= limit_price * 1.01)]
        for approach_date in approaches.index:
            approach_price = df.loc[approach_date, 'close']
            post_approach = df[df.index > approach_date]
            if side == 'BUY':
                break_level = limit_price * 0.995
                target_level = approach_price * 1.02
                hit_break = (post_approach['low'] <= break_level).any()
                hit_target = (post_approach['high'] >= target_level).any()
            else:
                break_level = limit_price * 1.005
                target_level = approach_price * 0.98
                hit_break = (post_approach['high'] >= break_level).any()
                hit_target = (post_approach['low'] <= target_level).any()
            if hit_break and hit_target:
                break_idx = post_approach[post_approach['low'] <= break_level].index[0] if side == 'BUY' else post_approach[post_approach['high'] >= break_level].index[0]
                target_idx = post_approach[post_approach['high'] >= target_level].index[0] if side == 'BUY' else post_approach[post_approach['low'] <= target_level].index[0]
                outcome = 1 if target_idx < break_idx else 0
            elif hit_target:
                outcome = 1
            elif hit_break:
                outcome = 0
            else:
                continue
            features = {'prev_approaches': len(approaches[approaches.index < approach_date])}
            if use_ema:
                features['ema'] = df.loc[approach_date, 'ema']
            if use_rsi:
                features['rsi'] = df.loc[approach_date, 'rsi']
            instances.append({'features': features, 'label': outcome})
    return instances

def train_model(instances, use_ema, use_rsi):
    if not instances:
        return None, 0
    X = []
    y = []
    for inst in instances:
        features = inst['features']
        X.append([features.get('prev_approaches', 0)] + 
                 ([features.get('ema', 0)] if use_ema else []) + 
                 ([features.get('rsi', 0)] if use_rsi else []))
        y.append(inst['label'])
    if len(X) < 2:
        return None, 0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, accuracy

def check_signals(df, zones, model, use_ema, use_rsi, trade_log):
    signals = []
    for zone in zones:
        limit_price = zone['level']
        base_time = zone['date']
        side = 'BUY' if zone['type'] == 'demand' else 'SELL'
        last_candle = df.iloc[-1]
        approach_condition = (last_candle['low'] >= limit_price * 0.99 and last_candle['low'] <= limit_price * 1.01) if side == 'BUY' else \
                            (last_candle['high'] >= limit_price * 0.99 and last_candle['high'] <= limit_price * 1.01)
        if approach_condition:
            prev_approaches = len(df[(df.index > base_time) & (df.index < df.index[-1]) & 
                                    (df['low' if side == 'BUY' else 'high'] >= limit_price * 0.99) & 
                                    (df['low' if side == 'BUY' else 'high'] <= limit_price * 1.01)])
            features = {'prev_approaches': prev_approaches}
            if use_ema:
                features['ema'] = last_candle['ema']
            if use_rsi:
                features['rsi'] = last_candle['rsi']
            pred_df = pd.DataFrame([features], columns=features.keys())
            pred = model.predict(pred_df)[0]
            pred_prob = model.predict_proba(pred_df)[0]
            signal = 'Buy' if side == 'BUY' and pred == 1 else 'Sell' if side == 'SELL' and pred == 1 else 'Avoid'
            signals.append({
                'ticker': df.name,
                'signal': signal,
                'limit_price': limit_price,
                'probability': pred_prob[1] if pred == 1 else pred_prob[0]
            })
    return signals

def update_chart(df, ax, ticker, use_ema, use_rsi, super_zones, trade_log):
    ax.clear()
    df_plot = df.copy()
    df_plot.index.name = 'Date'
    addplots = []
    if use_ema and 'ema' in df_plot.columns:
        addplots.append(mpf.make_addplot(df_plot['ema'], color='orange', label='EMA'))
    if use_rsi and 'rsi' in df_plot.columns:
        addplots.append(mpf.make_addplot(df_plot['rsi'], panel=1, color='purple', label='RSI', ylabel='RSI'))
    mpf.plot(df_plot, type='candle', ax=ax, volume=False, addplot=addplots, style='classic')
    ax.set_title(f"{ticker} ({st.session_state.period}/{st.session_state.interval})", fontsize=12)
    plot_zones(ax, df, super_zones, trade_log, super_zones=True)

def plot_zones(ax, df, zones, trade_log, super_zones=False):
    show_limit_lines = st.session_state.limit_lines
    show_mean_lines = st.session_state.mean_lines
    show_prices = st.session_state.show_prices
    enable_ai = st.session_state.enable_ai
    use_ema = st.session_state.use_ema
    use_rsi = st.session_state.use_rsi
    ema_period = st.session_state.ema_period

    model, accuracy = None, None
    if enable_ai and not super_zones:
        instances = find_approaches_and_labels(df, zones, use_ema, use_rsi)
        model, accuracy = train_model(instances, use_ema, use_rsi)

    for zone in zones:
        limit_price = zone['level']
        base_time = zone['date']
        side = 'BUY' if zone['type'] == 'demand' else 'SELL'
        color = 'blue' if side == 'BUY' else 'red'
        linewidth = 2 if super_zones else 1

        if base_time not in df.index:
            trade_log.append(f"Skipping {'Super ' if super_zones else ''}Zone at {limit_price:.2f} ({side}): Date {base_time} not in DataFrame index")
            continue

        idx = df.index.get_loc(base_time)
        if show_limit_lines:
            ax.axhline(y=limit_price, color=color, linestyle='--', alpha=0.5, linewidth=linewidth)
        if show_mean_lines:
            ax.axhline(y=limit_price, color='black', linestyle='--', linewidth=linewidth)
        if show_prices:
            ax.text(len(df) - 1, limit_price, f'{limit_price:.2f}', ha='right', va='center', fontsize=12, color=color)

        if enable_ai and model and not super_zones:
            last_candle = df.iloc[-1]
            approach_condition = (last_candle['low'] >= limit_price * 0.99 and last_candle['low'] <= limit_price * 1.01) if side == 'BUY' else \
                                (last_candle['high'] >= limit_price * 0.99 and last_candle['high'] <= limit_price * 1.01)
            if approach_condition:
                prev_approaches = len(df[(df.index > base_time) & (df.index < df.index[-1]) & 
                                        (df['low' if side == 'BUY' else 'high'] >= limit_price * 0.99) & 
                                        (df['low' if side == 'BUY' else 'high'] <= limit_price * 1.01)])
                features = {'prev_approaches': prev_approaches}
                if use_ema:
                    features['ema'] = last_candle['ema']
                if use_rsi:
                    features['rsi'] = last_candle['rsi']
                pred_df = pd.DataFrame([features], columns=features.keys())
                pred = model.predict(pred_df)[0]
                pred_prob = model.predict_proba(pred_df)[0]
                signal = 'Buy' if side == 'BUY' and pred == 1 else 'Sell' if side == 'SELL' and pred == 1 else 'Avoid'
                trade_log.append(f"AI Signal: {signal} at {limit_price:.2f} (Accuracy: {accuracy:.2f}, Probs: {pred_prob})")
                if show_prices:
                    ax.text(len(df) - 1, limit_price + (limit_price * 0.005), f"{signal}", color=color, fontsize=12, ha='right')

        trade_log.append(f"{'Super ' if super_zones else ''}Zone: {side} at {limit_price:.2f}, Latest: {df['close'].iloc[-1]:.2f}")

def save_chart(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf

# Streamlit app
def main():
    st.set_page_config(page_title="Stock Charting Tool", layout="wide")
    
    # Custom CSS for small fonts
    st.markdown("""
        <style>
        .css-1y0tads, .css-1v0mbdj, .css-1v3fvcr, .css-1r6slb0, .css-17e7dxy, .css-1d391kg {
            font-size: 12px !important;
        }
        .stButton>button {
            font-size: 12px !important;
            padding: 5px 10px !important;
        }
        .stTextInput input, .stSelectbox select, .stNumberInput input {
            font-size: 12px !important;
        }
        .stTextArea textarea {
            font-size: 12px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'ticker' not in st.session_state:
        st.session_state.ticker = ""
    if 'period' not in st.session_state:
        st.session_state.period = "1mo"
    if 'interval' not in st.session_state:
        st.session_state.interval = "1d"
    if 'limit_lines' not in st.session_state:
        st.session_state.limit_lines = True
    if 'mean_lines' not in st.session_state:
        st.session_state.mean_lines = False
    if 'show_prices' not in st.session_state:
        st.session_state.show_prices = False
    if 'enable_ai' not in st.session_state:
        st.session_state.enable_ai = False
    if 'live_update' not in st.session_state:
        st.session_state.live_update = False
    if 'use_ema' not in st.session_state:
        st.session_state.use_ema = False
    if 'ema_period' not in st.session_state:
        st.session_state.ema_period = 20
    if 'use_rsi' not in st.session_state:
        st.session_state.use_rsi = False
    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = []
    if 'show_sidebar' not in st.session_state:
        st.session_state.show_sidebar = True

    # Sidebar
    with st.sidebar:
        st.subheader("Stocks")
        if st.button("Toggle Sidebar"):
            st.session_state.show_sidebar = not st.session_state.show_sidebar

        if st.session_state.show_sidebar:
            for stock in sorted(display_mapping.keys()):
                if st.button(display_mapping[stock], key=stock):
                    st.session_state.ticker = stock
                    st.session_state.trade_log.append(f"Selected {stock}")

    # Main area
    st.title("Stock Charting Tool")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ticker = st.text_input("Ticker", value=st.session_state.ticker, placeholder="e.g., RELIANCE, Nifty")
        st.session_state.ticker = ticker
    with col2:
        st.session_state.period = st.selectbox("Period", 
            ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'], 
            index=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'].index(st.session_state.period))
    with col3:
        st.session_state.interval = st.selectbox("Interval", 
            ['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo'], 
            index=['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo'].index(st.session_state.interval))

    col4, col5, col6, col7, col8 = st.columns(5)
    with col4:
        st.session_state.limit_lines = st.checkbox("Limit Lines", value=st.session_state.limit_lines)
    with col5:
        st.session_state.mean_lines = st.checkbox("Mean Lines", value=st.session_state.mean_lines)
    with col6:
        st.session_state.show_prices = st.checkbox("Prices", value=st.session_state.show_prices)
    with col7:
        st.session_state.enable_ai = st.checkbox("AI", value=st.session_state.enable_ai)
    with col8:
        st.session_state.live_update = st.checkbox("Live", value=st.session_state.live_update)

    col9, col10, col11 = st.columns([1, 1, 1])
    with col9:
        st.session_state.use_ema = st.checkbox("EMA", value=st.session_state.use_ema)
    with col10:
        st.session_state.ema_period = st.number_input("EMA Period", min_value=1, value=st.session_state.ema_period, step=1)
    with col11:
        st.session_state.use_rsi = st.checkbox("RSI", value=st.session_state.use_rsi)

    if st.button("Plot"):
        if not st.session_state.ticker:
            st.session_state.trade_log.append("No ticker specified")
        else:
            plot_chart(st.session_state.ticker)

    # Trade log
    with st.expander("Trade Log"):
        st.text_area("Log", value="\n".join(st.session_state.trade_log[-50:]), height=100, disabled=True)

    # Live update simulation
    if st.session_state.live_update and st.session_state.ticker:
        plot_chart(st.session_state.ticker)

def plot_chart(ticker):
    try:
        period = st.session_state.period
        interval = st.session_state.interval
        use_ema = st.session_state.use_ema
        use_rsi = st.session_state.use_rsi
        ema_period = st.session_state.ema_period
        mapped_ticker = ticker_mapping.get(ticker.lower(), ticker)

        st.session_state.trade_log.append(f"Plotting {ticker} (Period: {period}, Interval: {interval})")
        data = fetch_and_process_data([mapped_ticker], period, interval, use_ema, ema_period, use_rsi, st.session_state.trade_log)
        if mapped_ticker not in data:
            st.session_state.trade_log.append(f"No data for {ticker}")
            return
        df = data[mapped_ticker]

        super_zones = identify_super_zones(ticker, st.session_state.trade_log)
        probabilities = calculate_super_zone_probability(ticker, super_zones, st.session_state.trade_log)
        for prob in probabilities:
            st.session_state.trade_log.append(f"Super Zone {prob['type']} at {prob['level']:.2f}: {prob['probability_held']:.1f}% chance to hold ({prob['approaches']} approaches)")

        fig, ax = plt.subplots(figsize=(8, 4))
        update_chart(df, ax, ticker, use_ema, use_rsi, super_zones, st.session_state.trade_log)
        st.pyplot(fig)

        # Save chart
        buf = save_chart(fig)
        st.download_button("Save Chart", data=buf, file_name=f"{ticker}_super_zones.png", mime="image/png")

    except Exception as e:
        st.session_state.trade_log.append(f"Error plotting {ticker}: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()

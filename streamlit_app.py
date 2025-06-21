import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import warnings
from datetime import datetime
import pytz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Series.__getitem__", category=FutureWarning)

# Ticker mapping
ticker_mapping = {
    "nifty50": "^NSEI",
    "banknifty": "^NSEBANK",
    "sensex": "^BSESN"
}

# Timeframe and period mapping
timeframes = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "4h": "4h", "1d": "1d", "1wk": "1wk", "1mo": "1mo"
}
periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y"]

# Initialize session state
if 'dfs' not in st.session_state:
    st.session_state.dfs = {}
if 'zones_list' not in st.session_state:
    st.session_state.zones_list = {}
if 'trades_list' not in st.session_state:
    st.session_state.trades_list = {}
if 'metrics_list' not in st.session_state:
    st.session_state.metrics_list = {}
if 'equity_list' not in st.session_state:
    st.session_state.equity_list = {}
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = []
if 'aligned_zones' not in st.session_state:
    st.session_state.aligned_zones = {}
if 'recommendation' not in st.session_state:
    st.session_state.recommendation = {}
if 'plot_data_ready' not in st.session_state:
    st.session_state.plot_data_ready = False
if 'backtest_data_ready' not in st.session_state:
    st.session_state.backtest_data_ready = False

# Function to fetch and clean data
def fetch_and_process_data(tickers, periods_intervals):
    all_data = {}
    for ticker in tickers:
        all_data[ticker] = {}
        for period, intervals in periods_intervals.items():
            for interval in intervals:
                try:
                    data = yf.download(ticker, period=period, interval=interval)
                    st.session_state.trade_log.append(f"Fetched data for {ticker} ({period}, {interval}): Index type={type(data.index)}, Length={len(data)}, Tz={data.index.tz}")
                    if data.empty:
                        st.session_state.trade_log.append(f"No data fetched for {ticker} ({period}, {interval})")
                        continue
                    data.reset_index(inplace=True)
                    data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
                    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                    data = data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
                    data.set_index('Date', inplace=True)
                    if len(data) >= 10000:
                        st.session_state.trade_log.append(f"Data exceeds 10000 candles for {ticker} ({period}, {interval}), truncated")
                        data = data.iloc[-10000:]
                    all_data[ticker][(period, interval)] = data
                except Exception as e:
                    st.session_state.trade_log.append(f"Error fetching {ticker} ({period}, {interval}): {str(e)}")
    return all_data

# Zone identification
def identify_zones(data):
    zones = []
    if len(data) < 2:
        st.session_state.trade_log.append("Insufficient data for zone identification")
        return zones
    for i in range(1, len(data) - 1):
        body_size_base = abs(data['open'].iloc[i] - data['close'].iloc[i])
        total_height_base = data['high'].iloc[i] - data['low'].iloc[i]
        if body_size_base > 0.5 * total_height_base or total_height_base == 0:
            continue
        body_size_follow = abs(data['open'].iloc[i+1] - data['close'].iloc[i+1])
        total_height_follow = data['high'].iloc[i+1] - data['low'].iloc[i+1]
        if body_size_follow < 0.75 * total_height_follow or total_height_follow == 0:
            continue
        date = data.index[i].to_pydatetime()
        ist = pytz.timezone('Asia/Kolkata')
        index_tz = data.index.tz
        if index_tz:
            zone_date = date.replace(tzinfo=index_tz) if date.tzinfo is None else date.astimezone(index_tz)
        else:
            zone_date = ist.localize(date) if date.tzinfo is None else date.astimezone(ist)
        if data['close'].iloc[i+1] > data['open'].iloc[i+1]:
            zones.append({'date': zone_date, 'type': 'demand', 'level': data['low'].iloc[i]})
        elif data['close'].iloc[i+1] < data['open'].iloc[i+1]:
            zones.append({'date': zone_date, 'type': 'supply', 'level': data['high'].iloc[i]})
    st.session_state.trade_log.append(f"Identified {len(zones)} zones for data length {len(data)}")
    return zones

# Approach detection and labeling
def find_approaches_and_labels(data, zones):
    instances = []
    if not zones or len(data) < 2:
        st.session_state.trade_log.append("No zones or insufficient data for approach detection")
        return instances
    for zone in zones:
        zone_date = zone['date']
        zone_level = zone['level']
        zone_type = zone['type']
        index_tz = data.index.tz
        if index_tz:
            zone_date = zone_date.astimezone(index_tz)
        future_data = data[data.index > zone_date]
        if zone_type == 'demand':
            approaches = future_data[(future_data['low'] >= zone_level * 0.99) & (future_data['low'] <= zone_level * 1.01)]
        else:
            approaches = future_data[(future_data['high'] >= zone_level * 0.99) & (future_data['high'] <= zone_level * 1.01)]
        for approach_date in approaches.index:
            approach_price = data.loc[approach_date, 'close']
            post_approach = data[data.index > approach_date]
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
            prev_approaches = len(data[(data.index > zone_date) & (data.index < approach_date) & 
                                      (data['low' if zone_type == 'demand' else 'high'] >= zone_level * 0.99) & 
                                      (data['low' if zone_type == 'demand' else 'high'] <= zone_level * 1.01)])
            features = {'prev_approaches': prev_approaches}
            instances.append({'features': features, 'outcome': outcome})
    st.session_state.trade_log.append(f"Found {len(instances)} approach instances for {len(zones)} zones")
    return instances

# Calculate zone significance
def calculate_zone_significance(data, zone, timeframe_idx):
    approaches = len(find_approaches_and_labels(data, [zone]))
    ist = pytz.timezone('Asia/Kolkata')
    age = (datetime.now(ist) - zone['date'].astimezone(ist)).total_seconds() / (3600 * 24)  # Age in days
    timeframe_weight = {0: 1.5, 1: 1.2, 2: 1.0, 3: 0.8}.get(timeframe_idx, 1.0)
    score = (approaches * 10 + age * 0.1) * timeframe_weight
    return score, approaches, age

# Find aligned zones
def find_aligned_zones(dfs, zones_list, timeframes_list):
    aligned_zones = {}
    for ticker in dfs:
        aligned_zones[ticker] = []  # Explicitly initialize as empty list
        st.session_state.trade_log.append(f"Processing aligned zones for {ticker}: dfs={len(dfs[ticker])}, zones_list={len(zones_list.get(ticker, []))}, timeframes={len(timeframes_list)}")
        if not any(zones for zones in zones_list.get(ticker, [])):
            st.session_state.trade_log.append(f"No zones found for {ticker}, skipping alignment")
            continue
        
        all_zones = []
        for idx, (df, zones, tf) in enumerate(zip(dfs[ticker], zones_list.get(ticker, [None]*4), timeframes_list)):
            st.session_state.trade_log.append(f"Processing timeframe {tf} for {ticker}: df={df is not None}, zones={zones is not None}, len(zones)={len(zones) if zones else 0}")
            if zones and df is not None:
                for zone in zones:
                    try:
                        score, approaches, age = calculate_zone_significance(df, zone, idx)
                        all_zones.append({
                            'level': zone['level'],
                            'type': zone['type'],
                            'timeframe': tf,
                            'score': score,
                            'approaches': approaches,
                            'age': age,
                            'date': zone['date']
                        })
                    except Exception as e:
                        st.session_state.trade_log.append(f"Error calculating significance for {ticker} (TF {tf}): {str(e)}")
        
        if not all_zones:
            st.session_state.trade_log.append(f"No valid zones processed for {ticker}")
            continue
        
        all_zones.sort(key=lambda x: x['level'])
        current_group = []
        for zone in all_zones:
            if not current_group or abs(zone['level'] - current_group[0]['level']) / current_group[0]['level'] <= 0.01:
                current_group.append(zone)
            else:
                if len(current_group) >= 2:
                    avg_level = np.mean([z['level'] for z in current_group])
                    zone_type = current_group[0]['type']
                    timeframes = [z['timeframe'] for z in current_group]
                    total_score = sum(z['score'] for z in current_group)
                    approaches = sum(z['approaches'] for z in current_group)
                    age = np.mean([z['age'] for z in current_group])
                    aligned_zones[ticker].append({
                        'level': avg_level,
                        'type': zone_type,
                        'timeframes': timeframes,
                        'score': total_score,
                        'approaches': approaches,
                        'age': age,
                        'count': len(current_group)
                    })
                current_group = [zone]
        
        if len(current_group) >= 2:
            avg_level = np.mean([z['level'] for z in current_group])
            zone_type = current_group[0]['type']
            timeframes = [z['timeframe'] for z in current_group]
            total_score = sum(z['score'] for z in current_group)
            approaches = sum(z['approaches'] for z in current_group)
            age = np.mean([z['age'] for z in current_group])
            aligned_zones[ticker].append({
                'level': avg_level,
                'type': zone_type,
                'timeframes': timeframes,
                'score': total_score,
                'approaches': approaches,
                'age': age,
                'count': len(current_group)
            })
    
    st.session_state.trade_log.append(f"Aligned zones for all tickers: {list(aligned_zones.keys())} with lengths {[len(z) for z in aligned_zones.values()]}")
    return aligned_zones

# Generate trade recommendation
def generate_trade_recommendation(dfs, aligned_zones, symbol):
    if not aligned_zones or symbol not in aligned_zones or not aligned_zones[symbol]:
        st.session_state.trade_log.append(f"No aligned zones available for {symbol}")
        return {"signal": "Hold", "confidence": 0, "details": "No aligned zones found"}
    if not isinstance(aligned_zones[symbol], list):
        st.session_state.trade_log.append(f"Invalid aligned_zones type for {symbol}: {type(aligned_zones[symbol])}")
        return {"signal": "Hold", "confidence": 0, "details": "Invalid aligned zones data"}
    
    latest_df = dfs[symbol][3]  # Lowest timeframe
    if latest_df is None:
        st.session_state.trade_log.append(f"No data for lowest timeframe for {symbol}")
        return {"signal": "Hold", "confidence": 0, "details": "No data for lowest timeframe"}
    
    current_price = latest_df['close'].iloc[-1]
    latest_candle = latest_df.iloc[-1]
    is_bullish = latest_candle['close'] > latest_candle['open']
    is_bearish = latest_candle['close'] < latest_candle['open']
    
    for zone in aligned_zones[symbol][:3]:
        zone_level = zone['level']
        zone_type = zone['type']
        proximity = abs(current_price - zone_level) / current_price
        if proximity > 0.02:
            continue
        
        confidence = min(zone['score'] / 50, 1.0) * 100
        if zone['count'] == 4:
            confidence *= 1.2
        elif zone['count'] == 3:
            confidence *= 1.1
        
        if zone_type == 'demand' and is_bullish:
            target = zone_level * 1.02
            stop_loss = zone_level * 0.995
            return {
                "signal": "Buy",
                "confidence": confidence,
                "details": f"Strong demand zone at ~{zone_level:.2f} ({zone['count']} TFs: {', '.join(zone['timeframes'])}). "
                          f"Target: {target:.2f}, Stop Loss: {stop_loss:.2f}. Approaches: {zone['approaches']}, Age: {zone['age']:.1f} days."
            }
        elif zone_type == 'supply' and is_bearish:
            target = zone_level * 0.98
            stop_loss = zone_level * 1.005
            return {
                "signal": "Sell",
                "confidence": confidence,
                "details": f"Strong supply zone at ~{zone_level:.2f} ({zone['count']} TFs: {', '.join(zone['timeframes'])}). "
                          f"Target: {target:.2f}, Stop Loss: {stop_loss:.2f}. Approaches: {zone['approaches']}, Age: {zone['age']:.1f} days."
            }
    
    return {"signal": "Hold", "confidence": 0, "details": "No clear trading opportunity"}

# Train the model
def train_model(instances):
    if not instances or len(instances) < 2:
        return None, None
    feature_cols = ['prev_approaches']
    df = pd.DataFrame([dict(inst['features'], outcome=1 if inst['outcome'] == 'held' else 0) for inst in instances])
    st.session_state.trade_log.append(f"Training data distribution: Held={df['outcome'].sum()}, Broke={len(df) - df['outcome'].sum()}")
    X = df[feature_cols]
    y = df['outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    st.session_state.trade_log.append(f"Model accuracy: {accuracy:.2f}")
    return model, accuracy

# Backtesting strategy
def backtest_strategy(data, zones):
    trades = []
    equity = [100000]
    position = None
    zones = sorted(zones, key=lambda x: x['level'])

    for i in range(2, len(data)):
        current_time = data.index[i]
        current_candle = data.iloc[i]
        open_zones = [z for z in zones if z['date'] < current_time]
        
        for zone in open_zones:
            zone_level = zone['level']
            zone_type = zone['type']
            is_demand = zone_type == 'demand'
            body_size = abs(current_candle['close'] - current_candle['open'])
            candle_range = current_candle['high'] - current_candle['low']
            is_strong = body_size >= 0.7 * candle_range if candle_range > 0 else False
            
            touch_condition = (current_candle['low'] >= zone_level * 0.99 and current_candle['low'] <= zone_level * 1.01) if is_demand else \
                              (current_candle['high'] >= zone_level * 0.99 and current_candle['high'] <= zone_level * 1.01)
            if touch_condition and is_strong and position is None:
                if is_demand and current_candle['close'] > current_candle['open']:
                    targets = [z['level'] for z in open_zones if z['type'] == 'supply' and z['level'] > zone_level]
                    target = min(targets) if targets else zone_level * 1.02
                    position = {
                        'type': 'bounce_long',
                        'entry_time': current_time,
                        'entry_price': zone_level,
                        'target': target,
                        'stop_loss': zone_level * 0.99,
                        'highest_price': zone_level,
                        'candle_count': 0
                    }
                elif not is_demand and current_candle['close'] < current_candle['open']:
                    targets = [z['level'] for z in open_zones if z['type'] == 'demand' and z['level'] < zone_level]
                    target = max(targets) if targets else zone_level * 0.98
                    position = {
                        'type': 'bounce_short',
                        'entry_time': current_time,
                        'entry_price': zone_level,
                        'target': target,
                        'stop_loss': zone_level * 1.01,
                        'lowest_price': zone_level,
                        'candle_count': 0
                    }

        if position:
            position['candle_count'] += 1
            if 'long' in position['type']:
                position['highest_price'] = max(position['highest_price'], current_candle['high'])
                position['stop_loss'] = max(position['stop_loss'], position['highest_price'] * 0.995)
            else:
                position['lowest_price'] = min(position['lowest_price'], current_candle['low'])
                position['stop_loss'] = min(position['stop_loss'], position['lowest_price'] * 1.005)

            exit_trade = False
            if position['type'] in ['bounce_long']:
                if current_candle['high'] >= position['target']:
                    exit_price = position['target']
                    outcome = 'win'
                    exit_type = 'Target'
                    exit_trade = True
                elif current_candle['low'] <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    outcome = 'loss'
                    exit_type = 'Stop Loss'
                    exit_trade = True
            else:
                if current_candle['low'] <= position['target']:
                    exit_price = position['target']
                    outcome = 'win'
                    exit_type = 'Target'
                    exit_trade = True
                elif current_candle['high'] >= position['stop_loss']:
                    exit_price = position['stop_loss']
                    outcome = 'loss'
                    exit_type = 'Stop Loss'
                    exit_trade = True
            if position['candle_count'] >= 20:
                exit_price = current_candle['close']
                outcome = 'neutral'
                exit_type = 'Timeout'
                exit_trade = True
            if exit_trade:
                pl = (exit_price - position['entry_price']) * 1000 if 'long' in position['type'] else (position['entry_price'] - exit_price) * 1000
                equity.append(equity[-1] + pl)
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'type': position['type'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'pl': pl,
                    'outcome': outcome,
                    'exit_type': exit_type
                })
                position = None

    equity = np.array(equity)
    returns = np.diff(equity) / equity[:-1]
    metrics = {
        'total_return': (equity[-1] - equity[0]) / equity[0] * 100 if equity.size > 1 else 0,
        'win_rate': len([t for t in trades if t['outcome'] == 'win']) / len(trades) * 100 if trades else 0,
        'max_drawdown': max([(max(equity[:i+1]) - min(equity[:i+1])) / max(equity[:i+1]) * 100 for i in range(len(equity))] or [0]),
        'num_trades': len(trades)
    }
    return trades, metrics, equity

# Plotting function
def plot_chart(df, zones, symbol, timeframe, period, show_buy_zones, show_sell_zones, show_limit_lines, show_prices, aligned_zones):
    if df is None or zones is None:
        st.session_state.trade_log.append(f"plot_chart failed for {symbol} (Timeframe: {timeframe}, Period: {period}): df or zones is None")
        return None
    fig, ax = mpf.plot(df, type='candle', style='charles', returnfig=True, figsize=(10, 5))
    ax[0].set_title(f'{symbol} Candlestick (Timeframe: {timeframe}, Period: {period})')
    ax[0].set_ylabel('Price')

    for zone in zones:
        limit_price = zone['level']
        side = 'BUY' if zone['type'] == 'demand' else 'SELL'
        if side == 'BUY' and not show_buy_zones:
            continue
        if side == 'SELL' and not show_sell_zones:
            continue
        color = 'blue' if side == 'BUY' else 'red'
        linewidth = 1
        alpha = 0.5

        for az in aligned_zones.get(symbol, []):
            if az and isinstance(az, dict) and 'level' in az and 'type' in az:
                if abs(az['level'] - limit_price) / limit_price < 0.01 and az['type'] == zone['type']:
                    linewidth = 2
                    alpha = 0.8
                    break

        if show_limit_lines:
            ax[0].axhline(y=limit_price, color=color, linestyle='--', alpha=alpha, linewidth=linewidth)
        if show_prices:
            ax[0].text(len(df) - 1, limit_price, f'{limit_price:.2f}', ha='right', va='center', fontsize=10, color=color)

    return fig

def plot_trade_chart(df, zones, trades, symbol, timeframe, period, aligned_zones):
    if df is None or zones is None or not trades:
        st.session_state.trade_log.append(f"plot_trade_chart failed for {symbol} (Timeframe: {timeframe}, Period: {period}): df, zones, or trades invalid")
        return None
    fig, ax = mpf.plot(df, type='candle', style='charles', returnfig=True, figsize=(10, 5))
    ax[0].set_title(f'{symbol} Trades (Timeframe: {timeframe}, Period: {period})')
    ax[0].set_ylabel('Price')

    for zone in zones:
        limit_price = zone['level']
        side = 'BUY' if zone['type'] == 'demand' else 'SELL'
        color = 'blue' if side == 'BUY' else 'red'
        linewidth = 1
        alpha = 0.5

        for az in aligned_zones.get(symbol, []):
            if az and isinstance(az, dict) and 'level' in az and 'type' in az:
                if abs(az['level'] - limit_price) / limit_price < 0.01 and az['type'] == zone['type']:
                    linewidth = 2
                    alpha = 0.8
                    break

        ax[0].axhline(y=limit_price, color=color, linestyle='--', alpha=alpha, linewidth=linewidth)

    entry_times = []
    entry_prices = []
    exit_times = []
    exit_prices = []
    stop_loss_times = []
    stop_loss_prices = []
    for trade in trades:
        try:
            entry_idx = df.index.get_loc(trade['entry_time'])
            exit_idx = df.index.get_loc(trade['exit_time'])
            entry_times.append(entry_idx)
            entry_prices.append(trade['entry_price'])
            if trade['exit_type'] == 'Stop Loss':
                stop_loss_times.append(exit_idx)
                stop_loss_prices.append(trade['exit_price'])
            else:
                exit_times.append(exit_idx)
                exit_prices.append(trade['exit_price'])
        except KeyError:
            continue

    if entry_times:
        ax[0].scatter(entry_times, entry_prices, marker='^', color='#2ca02c', label='Entry', s=100)
    if exit_times:
        ax[0].scatter(exit_times, exit_prices, marker='v', color='#d62728', label='Exit', s=100)
    if stop_loss_times:
        ax[0].scatter(stop_loss_times, stop_loss_prices, marker='v', color='#1f77b4', label='Stop Loss Exit', s=100)
    ax[0].legend()
    return fig

# Streamlit UI
st.set_page_config(page_title="Indian Stocks Trading App", layout="wide")
st.title("Indian Stocks Trading App")

# Guidelines
st.markdown("""
**Guidelines**: 
- Enter valid tickers (e.g., `nifty50, banknifty`) separated by commas.
- Use consistent timeframes and periods (e.g., `1d` with `1mo`, `1wk` with `6mo`).
- Uncheck `NSE stocks` for indices like `nifty50` (`^NSEI`); check for stocks (e.g., `RELIANCE` → `RELIANCE.NS`).
""")

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    tickers = st.text_input("Enter stock symbols (e.g., nifty50, banknifty)", "nifty50")
    append_ns = st.checkbox("NSE stocks", value=True)
    
    st.subheader("Timeframe 1")
    timeframe_1 = st.selectbox("Timeframe 1", list(timeframes.keys()), index=6, key="tf1")  # Default to 1d
    period_1 = st.selectbox("Period 1", periods, index=4, key="p1")
    
    st.subheader("Timeframe 2")
    timeframe_2 = st.selectbox("Timeframe 2", list(timeframes.keys()), index=7, key="tf2")  # Default to 1wk
    period_2 = st.selectbox("Period 2", periods, index=2, key="p2")
    
    st.subheader("Timeframe 3")
    timeframe_3 = st.selectbox("Timeframe 3", list(timeframes.keys()), index=3, key="tf3")
    period_3 = st.selectbox("Period 3", periods, index=1, key="p3")
    
    st.subheader("Timeframe 4")
    timeframe_4 = st.selectbox("Timeframe 4", list(timeframes.keys()), index=2, key="tf4")
    period_4 = st.selectbox("Period 4", periods, index=0, key="p4")
    
    show_buy_zones = st.checkbox("Show Demand Zones", value=True)
    show_sell_zones = st.checkbox("Show Supply Zones", value=True)
    show_limit_lines = st.checkbox("Show Limit Lines", value=True)
    show_prices = st.checkbox("Show Zone Prices", value=True)
    plot_button = st.button("Plot Chart")
    backtest_button = st.button("Run Backtest")

# Process tickers
st.session_state.trade_log.append(f"Raw ticker input: {tickers} (type: {type(tickers)})")
tickers_clean = tickers.strip()
st.session_state.trade_log.append(f"Cleaned ticker input: {tickers_clean}")
if not tickers_clean:
    tickers_clean = "nifty50"
    st.session_state.trade_log.append("No valid tickers provided, defaulting to 'nifty50'")

entered_tickers = [ticker.strip().lower() for ticker in tickers_clean.split(",") if ticker.strip()]
st.session_state.trade_log.append(f"Entered tickers: {entered_tickers}")
if not entered_tickers:
    entered_tickers = ["nifty50"]
    st.session_state.trade_log.append("No valid tickers parsed, defaulting to ['nifty50']")

mapped_tickers = [ticker_mapping.get(ticker, ticker) for ticker in entered_tickers]
final_ticker_list = [symbol.upper() + ".NS" if append_ns and not symbol.startswith('^') else symbol.upper() for symbol in mapped_tickers]
st.session_state.trade_log.append(f"Final ticker list: {final_ticker_list}")

# Validate timeframes
timeframes_list = [timeframe_1, timeframe_2, timeframe_3, timeframe_4]
if isinstance(timeframes_list, tuple):
    timeframes_list = list(timeframes_list)
    st.session_state.trade_log.append("Converted timeframes_list from tuple to list")
st.session_state.trade_log.append(f"Timeframes list: {timeframes_list}")

# Define periods_intervals
periods_list = [period_1, period_2, period_3, period_4]
periods_intervals = {periods_list[i]: [timeframes_list[i]] for i in range(4)}
st.session_state.trade_log.append(f"Periods intervals: {periods_intervals}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Chart", "Backtest", "Trade Chart", "Decision"])

# Chart Tab
with tab1:
    if plot_button and not st.session_state.plot_data_ready:
        st.session_state.dfs = {ticker: [None] * 4 for ticker in final_ticker_list}
        st.session_state.zones_list = {ticker: [None] * 4 for ticker in final_ticker_list}
        st.session_state.aligned_zones = {}
        st.session_state.recommendation = {}
        st.session_state.plot_data_ready = True
        
        all_data = fetch_and_process_data(final_ticker_list, periods_intervals)
        for ticker in final_ticker_list:
            st.session_state.trade_log.append(f"Processing ticker: {ticker}")
            data_fetched = False
            if ticker in all_data:
                for idx, (tf, period) in enumerate(zip(timeframes_list, periods_list)):
                    if (period, tf) in all_data[ticker]:
                        df = all_data[ticker][(period, tf)]
                        st.session_state.dfs[ticker][idx] = df
                        st.session_state.zones_list[ticker][idx] = identify_zones(df)
                        data_fetched = True
                        st.session_state.trade_log.append(f"Found {len(st.session_state.zones_list[ticker][idx])} zones for {ticker} (Timeframe: {tf}, Period: {period})")
                    else:
                        st.session_state.trade_log.append(f"No data found for {ticker} (Timeframe: {tf}, Period: {period})")
            
            if not data_fetched:
                st.error(f"No valid data fetched for {ticker}. Please check ticker or timeframe settings.")
            else:
                st.session_state.aligned_zones[ticker] = find_aligned_zones({ticker: st.session_state.dfs[ticker]}, 
                                                                           {ticker: st.session_state.zones_list[ticker]}, 
                                                                           timeframes_list)
                if ticker in st.session_state.aligned_zones and st.session_state.aligned_zones[ticker]:
                    st.session_state.recommendation[ticker] = generate_trade_recommendation({ticker: st.session_state.dfs[ticker]}, 
                                                                                           st.session_state.aligned_zones, ticker)
                    st.session_state.trade_log.append(f"Found {len(st.session_state.aligned_zones[ticker])} aligned zones for {ticker}. Recommendation: {st.session_state.recommendation[ticker]['signal']}")
                else:
                    st.session_state.trade_log.append(f"No aligned zones found for {ticker}, skipping recommendation")
        
        if final_ticker_list:  # Ensure final_ticker_list is valid
            cols = st.columns(min(4, len(final_ticker_list) * 2))  # Up to 4 columns
            idx = 0
            for ticker in final_ticker_list:
                for tf, period, df, zones in zip(timeframes_list, periods_list, st.session_state.dfs[ticker], st.session_state.zones_list[ticker]):
                    if df is not None and zones is not None:
                        with cols[idx % 4]:
                            fig = plot_chart(df, zones, ticker, tf, period, show_buy_zones, show_sell_zones, show_limit_lines, show_prices, st.session_state.aligned_zones)
                            if fig:
                                st.pyplot(fig)
                                st.session_state.trade_log.append(f"Chart plotted successfully for {ticker} (Timeframe: {tf})!")
                                plt.close(fig)
                        idx += 1
        else:
            st.error("No valid tickers to plot. Please check ticker input.")

# Backtest Tab
with tab2:
    if backtest_button and not st.session_state.backtest_data_ready:
        st.session_state.trades_list = {ticker: [[] for _ in range(4)] for ticker in final_ticker_list}
        st.session_state.metrics_list = {ticker: [{} for _ in range(4)] for ticker in final_ticker_list}
        st.session_state.equity_list = {ticker: [[100000] for _ in range(4)] for ticker in final_ticker_list}
        st.session_state.backtest_data_ready = True
        
        timeframes_list = [timeframe_1, timeframe_2, timeframe_3, timeframe_4]
        periods_list = [period_1, period_2, period_3, period_4]
        for ticker in final_ticker_list:
            for idx, (df, zones, tf, period) in enumerate(zip(st.session_state.dfs[ticker], st.session_state.zones_list[ticker], timeframes_list, periods_list)):
                if df is None or zones is None:
                    continue
                filtered_zones = [z for z in zones if (z['type'] == 'demand' and show_buy_zones) or (z['type'] == 'supply' and show_sell_zones)]
                trades, metrics, equity = backtest_strategy(df, filtered_zones)
                st.session_state.trades_list[ticker][idx] = trades
                st.session_state.metrics_list[ticker][idx] = metrics
                st.session_state.equity_list[ticker][idx] = equity
                st.session_state.trade_log.append(f"Backtest completed for {ticker} (Timeframe {idx+1}: {tf}): {metrics['num_trades']} trades")
        
        if final_ticker_list:  # Ensure final_ticker_list is valid
            cols = st.columns(min(4, len(final_ticker_list) * 2))
            idx = 0
            for ticker in final_ticker_list:
                for idx_tf, (metrics, equity, trades, tf, period) in enumerate(zip(st.session_state.metrics_list[ticker], st.session_state.equity_list[ticker], st.session_state.trades_list[ticker], timeframes_list, periods_list)):
                    if metrics:
                        with cols[idx % 4]:
                            st.subheader(f"{ticker} - Timeframe {idx_tf+1}: {tf}, Period: {period}")
                            col3, col4 = st.columns([1, 1])
                            with col3:
                                st.metric("Total Return", f"{metrics.get('total_return', 0):.2f}%")
                                st.metric("Win Rate", f"{metrics.get('win_rate', 0):.2f}%")
                                st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")
                                st.metric("Number of Trades", metrics.get('num_trades', 0))
                            with col4:
                                fig, ax = plt.subplots(figsize=(5, 3))
                                ax.plot(equity, color='blue', label='Equity')
                                ax.set_title(f"{ticker} Equity Curve (TF {idx_tf+1})")
                                ax.set_xlabel("Trade")
                                ax.set_ylabel("Equity ($)")
                                ax.grid(True)
                                ax.legend()
                                st.pyplot(fig)
                                plt.close(fig)
                            
                                trade_df = pd.DataFrame(trades)
                                if not trade_df.empty:
                                    trade_df['entry_time'] = trade_df['entry_time'].astype(str)
                                    trade_df['exit_time'] = trade_df['exit_time'].astype(str)
                                    st.dataframe(trade_df[['entry_time', 'exit_time', 'type', 'entry_price', 'exit_price', 'pl', 'exit_type']])
                        idx += 1
        else:
            st.error("No valid tickers to backtest. Please check ticker input.")

# Trade Chart Tab
with tab3:
    if any(any(trades for trades in tls.values()) for tls in st.session_state.trades_list.values()) and any(any(df is not None for df in dfs.values()) for dfs in st.session_state.dfs.values()):
        timeframes_list = [timeframe_1, timeframe_2, timeframe_3, timeframe_4]
        periods_list = [period_1, period_2, period_3, period_4]
        if final_ticker_list:  # Ensure final_ticker_list is valid
            cols = st.columns(min(4, len(final_ticker_list) * 2))
            idx = 0
            for ticker in final_ticker_list:
                for idx_tf, (df, zones, trades, tf, period) in enumerate(zip(st.session_state.dfs[ticker], st.session_state.zones_list[ticker], st.session_state.trades_list[ticker], timeframes_list, periods_list)):
                    if df is not None and zones is not None and trades:
                        with cols[idx % 4]:
                            fig = plot_trade_chart(df, zones, trades, ticker, tf, period, st.session_state.aligned_zones)
                            if fig:
                                st.pyplot(fig)
                                st.session_state.trade_log.append(f"Trade Chart updated for {ticker} (Timeframe {idx_tf+1}: {tf})")
                                plt.close(fig)
                        idx += 1
    else:
        st.write("Run a backtest to view the trade charts.")

# Decision Tab
with tab4:
    if st.session_state.plot_data_ready:
        for ticker in final_ticker_list:
            if ticker in st.session_state.recommendation and ticker in st.session_state.aligned_zones:
                st.subheader(f"Trade Recommendation for {ticker}")
                rec = st.session_state.recommendation[ticker]
                st.write(f"**Signal**: {rec['signal']}")
                st.write(f"**Confidence**: {rec['confidence']:.1f}%")
                st.write(f"**Details**: {rec['details']}")
                
                st.subheader(f"Aligned Zones for {ticker}")
                zones_df = pd.DataFrame(st.session_state.aligned_zones[ticker])
                if not zones_df.empty:
                    zones_df['timeframes'] = zones_df['timeframes'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')
                    zones_df['level'] = zones_df['level'].round(2)
                    zones_df['age'] = zones_df['age'].round(1)
                    zones_df = zones_df[['level', 'type', 'timeframes', 'count', 'approaches', 'age', 'score']]
                    zones_df.columns = ['Price Level', 'Type', 'Timeframes', 'TF Count', 'Approaches', 'Age (days)', 'Score']
                    st.dataframe(zones_df)
            else:
                st.write(f"Plot charts to generate trade recommendations for {ticker}.")
    else:
        st.write("Click 'Plot Chart' to generate trade recommendations.")

# Trade Log
st.header("Trade Log")
st.text_area("Log", "\n".join(st.session_state.trade_log[-20:]), height=300, disabled=True)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf
import warnings
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

# Fetch and process data
def fetch_and_process_data(tickers, period, interval, trade_log):
    all_data = {}
    for ticker in tickers:
        try:
            mapped_ticker = ticker_mapping.get(ticker.lower(), ticker)
            if not mapped_ticker.startswith('^'):
                mapped_ticker = f"{ticker.upper()}.NS"
            trade_log.append(f"Fetching data for {ticker} ({mapped_ticker}, Period: {period}, Interval: {interval})")
            data = yf.download(mapped_ticker, period=period, interval=interval, progress=False)
            if data.empty:
                trade_log.append(f"No data returned for {ticker} ({mapped_ticker})")
                continue
            data.reset_index(inplace=True)
            if 'Datetime' in data.columns:
                data = data.rename(columns={'Datetime': 'Date'})
            data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
            data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            data = data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
            data.set_index('Date', inplace=True)
            data.index = data.index.tz_localize(None)
            trade_log.append(f"Fetched {len(data)} data points for {ticker}. First: {data.index[0]}, Last: {data.index[-1]}")
            data.dropna(inplace=True)
            trade_log.append(f"After processing, {len(data)} data points remain for {ticker}")
            all_data[ticker] = data
        except Exception as e:
            trade_log.append(f"Error fetching data for {ticker}: {str(e)}")
    return all_data

def identify_zones(df, interval, trade_log):
    try:
        zones = []
        window = 3 if interval == '5m' else 1
        for i in range(window, len(df) - window):
            if all(df['low'].iloc[i] < df['low'].iloc[i-j] for j in range(1, window + 1)) and \
               all(df['low'].iloc[i] < df['low'].iloc[i+j] for j in range(1, window + 1)):
                zones.append({'date': df.index[i], 'type': 'demand', 'level': df['low'].iloc[i]})
            if all(df['high'].iloc[i] > df['high'].iloc[i-j] for j in range(1, window + 1)) and \
               all(df['high'].iloc[i] > df['high'].iloc[i+j] for j in range(1, window + 1)):
                zones.append({'date': df.index[i], 'type': 'supply', 'level': df['high'].iloc[i]})
        trade_log.append(f"Identified {len(zones)} zones for interval {interval}")
        return zones
    except Exception as e:
        trade_log.append(f"Error identifying zones for interval {interval}: {str(e)}")
        return []

def identify_super_zones(ticker, trade_log):
    try:
        super_zones = []
        timeframe_configs = [
            {'period': '1y', 'interval': '1wk'},
            {'period': '6mo', 'interval': '1wk'},
            {'period': '6mo', 'interval': '1d'},
            {'period': '3mo', 'interval': '1d'},
            {'period': '1mo', 'interval': '1h'},
            {'period': '1mo', 'interval': '30m'},
            {'period': '5d', 'interval': '15m'},
            {'period': '1d', 'interval': '5m'}
        ]
        
        all_zones = []
        mapped_ticker = ticker_mapping.get(ticker.lower(), ticker)
        trade_log.append(f"Identifying super zones for {ticker} (mapped to {mapped_ticker})")
        for config in timeframe_configs:
            period = config['period']
            interval = config['interval']
            data = fetch_and_process_data([mapped_ticker], period, interval, trade_log)
            if mapped_ticker in data:
                zones = identify_zones(data[mapped_ticker], interval, trade_log)
                trade_log.append(f"Found {len(zones)} zones for {ticker} at {period}/{interval}")
                for zone in zones:
                    zone['date'] = zone['date'].round('5min')
                    zone['period'] = period
                    zone['interval'] = interval
                    trade_log.append(f"Zone at {zone['level']:.2f} ({zone['type']}) on {zone['date']} for {period}/{interval}")
                all_zones.extend(zones)
            else:
                trade_log.append(f"No data for {mapped_ticker} at {period}/{interval}")
        
        # Step 1: Weekly + daily/hourly/minute super zones
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
                    threshold = 0.015 if '5m' in [z['interval'] for z in cluster] else 0.01
                    if abs(zones[j]['level'] - avg_level) <= avg_level * threshold:
                        cluster.append(zones[j])
                        zones.pop(j)
                    else:
                        j += 1
                intervals = set(z['interval'] for z in cluster)
                has_weekly = '1wk' in intervals
                has_daily_or_shorter = '1d' in intervals or '1h' in intervals or '30m' in intervals or '15m' in intervals or '5m' in intervals
                if has_weekly and has_daily_or_shorter:
                    avg_level = np.mean([z['level'] for z in cluster])
                    super_zones.append({
                        'date': min(z['date'] for z in cluster),
                        'type': zone_type,
                        'level': avg_level,
                        'periods': list(set(z['period'] for z in cluster)),
                        'intervals': list(intervals)
                    })
                    trade_log.append(f"Super zone {zone_type} at {avg_level:.2f} with intervals {list(intervals)}")
                i += 1
        
        # Step 2: Intraday super zones (1mo/1h + 1mo/30m, 5d/15m + 1d/5m)
        intraday_zones = [z for z in all_zones if z['interval'] in ['1h', '30m', '15m', '5m']]
        intraday_demand = [z for z in intraday_zones if z['type'] == 'demand']
        intraday_supply = [z for z in intraday_zones if z['type'] == 'supply']
        
        for zone_type in ['demand', 'supply']:
            zones = intraday_demand if zone_type == 'demand' else intraday_supply
            i = 0
            while i < len(zones):
                cluster = [zones[i]]
                j = i + 1
                while j < len(zones):
                    avg_level = np.mean([z['level'] for z in cluster])
                    threshold = 0.01
                    if abs(zones[j]['level'] - avg_level) <= avg_level * threshold:
                        cluster.append(zones[j])
                        zones.pop(j)
                    else:
                        j += 1
                intervals = set(z['interval'] for z in cluster)
                # 1mo/1h + 1mo/30m
                if '1h' in intervals and '30m' in intervals:
                    avg_level = np.mean([z['level'] for z in cluster])
                    super_zones.append({
                        'date': min(z['date'] for z in cluster),
                        'type': zone_type,
                        'level': avg_level,
                        'periods': ['1mo'],
                        'intervals': list(intervals)
                    })
                    trade_log.append(f"Intraday super zone {zone_type} at {avg_level:.2f} with intervals {list(intervals)}")
                # 5d/15m + 1d/5m
                if '15m' in intervals and '5m' in intervals:
                    avg_level = np.mean([z['level'] for z in cluster])
                    super_zones.append({
                        'date': min(z['date'] for z in cluster),
                        'type': zone_type,
                        'level': avg_level,
                        'periods': ['5d', '1d'],
                        'intervals': list(intervals)
                    })
                    trade_log.append(f"Intraday super zone {zone_type} at {avg_level:.2f} with intervals {list(intervals)}")
                i += 1
        
        trade_log.append(f"Found {len(super_zones)} super zones for {ticker}")
        return super_zones
    except Exception as e:
        trade_log.append(f"Error identifying super zones for {ticker}: {str(e)}")
        return []

def find_approaches_and_labels(df, zones):
    try:
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
                instances.append({'features': features, 'label': outcome})
        return instances
    except Exception as e:
        st.session_state.trade_log.append(f"Error finding approaches and labels: {str(e)}")
        return []

def train_model(instances):
    try:
        if not instances:
            return None, 0
        X = []
        y = []
        for inst in instances:
            features = inst['features']
            X.append([features.get('prev_approaches', 0)])
            y.append(inst['label'])
        if len(X) < 2:
            return None, 0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        return model, accuracy
    except Exception as e:
        st.session_state.trade_log.append(f"Error training model: {str(e)}")
        return None, 0

def check_signals(df, zones, model, trade_log):
    try:
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
    except Exception as e:
        trade_log.append(f"Error checking signals: {str(e)}")
        return []

def update_chart(df, ax, ticker, super_zones, trade_log, period, interval):
    try:
        ax.clear()
        df_plot = df.copy()
        df_plot.index.name = 'Date'
        mpf.plot(df_plot, type='candle', ax=ax, volume=False, style='classic')
        ax.set_title(f"{ticker} ({period}/{interval})", fontsize=12)
        plot_zones(ax, df, super_zones, trade_log, super_zones=True)
        trade_log.append(f"Rendered {len(super_zones)} super zones for {ticker} at {period}/{interval}")
    except Exception as e:
        trade_log.append(f"Error updating chart for {ticker} ({period}/{interval}): {str(e)}")

def plot_zones(ax, df, zones, trade_log, super_zones=False):
    try:
        show_limit_lines = st.session_state.limit_lines
        show_prices = st.session_state.show_prices
        enable_ai = st.session_state.enable_ai

        model, accuracy = None, None
        if enable_ai and not super_zones:
            instances = find_approaches_and_labels(df, zones)
            model, accuracy = train_model(instances)

        for zone in zones:
            limit_price = zone['level']
            base_time = zone['date']
            side = 'BUY' if zone['type'] == 'demand' else 'SELL'
            color = 'blue' if side == 'BUY' else 'red'
            linewidth = 2 if super_zones else 1

            if base_time not in df.index:
                try:
                    closest_idx = (df.index - base_time).abs().argmin()
                    closest_date = df.index[closest_idx]
                    trade_log.append(f"Adjusting zone date from {base_time} to {closest_date} for {limit_price:.2f} ({side})")
                    base_time = closest_date
                except Exception as e:
                    trade_log.append(f"Skipping {'Super ' if super_zones else ''}Zone at {limit_price:.2f} ({side}): {str(e)}")
                    continue

            idx = df.index.get_loc(base_time)
            if show_limit_lines:
                ax.axhline(y=limit_price, color=color, linestyle='--', alpha=0.5, linewidth=linewidth)
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
                    pred_df = pd.DataFrame([features], columns=features.keys())
                    pred = model.predict(pred_df)[0]
                    pred_prob = model.predict_proba(pred_df)[0]
                    signal = 'Buy' if side == 'BUY' and pred == 1 else 'Sell' if side == 'SELL' and pred == 1 else 'Avoid'
                    trade_log.append(f"AI Signal: {signal} at {limit_price:.2f} (Accuracy: {accuracy:.2f}, Probs: {pred_prob})")
                    if show_prices:
                        ax.text(len(df) - 1, limit_price + (limit_price * 0.005), f"{signal}", color=color, fontsize=12, ha='right')

            trade_log.append(f"{'Super ' if super_zones else ''}Zone: {side} at {limit_price:.2f}, Latest: {df['close'].iloc[-1]:.2f}")
    except Exception as e:
        trade_log.append(f"Error plotting zones: {str(e)}")

def save_chart(fig):
    try:
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        return buf
    except Exception as e:
        st.session_state.trade_log.append(f"Error saving chart: {str(e)}")
        return None

def plot_chart(ticker, period=None, interval=None):
    try:
        period = period or st.session_state.period
        interval = interval or st.session_state.interval
        mapped_ticker = ticker_mapping.get(ticker.lower(), ticker)
        trade_log = st.session_state.trade_log

        trade_log.append(f"Starting plot for {ticker} (Period: {period}, Interval: {interval})")
        
        if not mapped_ticker:
            trade_log.append(f"Invalid ticker: {ticker}")
            return None, None

        trade_log.append(f"Fetching data for {ticker}")
        data = fetch_and_process_data([mapped_ticker], period, interval, trade_log)
        if mapped_ticker not in data:
            trade_log.append(f"No data for {ticker}")
            return None, None

        df = data[mapped_ticker]
        trade_log.append(f"Data fetched for {ticker}: {len(df)} rows")

        trade_log.append(f"Identifying super zones for {ticker}")
        super_zones = identify_super_zones(ticker, trade_log)
        trade_log.append(f"Found {len(super_zones)} super zones")

        trade_log.append(f"Creating chart for {ticker}")
        fig, ax = plt.subplots(figsize=(8, 4))
        update_chart(df, ax, ticker, super_zones, trade_log, period, interval)
        
        trade_log.append(f"Saving chart for {ticker}")
        buf = save_chart(fig)
        plt.close(fig)
        
        trade_log.append(f"Plot completed for {ticker}")
        return fig, buf
    except Exception as e:
        trade_log.append(f"Error plotting {ticker}: {str(e)}")
        return None, None

def plot_analysis_charts(ticker):
    try:
        trade_log = st.session_state.trade_log
        timeframes = [
            {'period': '1y', 'interval': '1wk'},
            {'period': '6mo', 'interval': '1wk'},
            {'period': '6mo', 'interval': '1d'},
            {'period': '3mo', 'interval': '1d'},
            {'period': '1mo', 'interval': '1h'},
            {'period': '1mo', 'interval': '30m'},
            {'period': '5d', 'interval': '15m'},
            {'period': '1d', 'interval': '5m'}
        ]

        trade_log.append(f"Plotting full analysis charts for {ticker}")
        for tf in timeframes:
            period = tf['period']
            interval = tf['interval']
            with st.container():
                st.subheader(f"{ticker} ({period}/{interval})", anchor=False)
                trade_log.append(f"Generating chart for {ticker} ({period}/{interval})")
                fig, buf = plot_chart(ticker, period, interval)
                if fig and buf:
                    st.pyplot(fig)
                    st.download_button(
                        f"Save {period}/{interval} Chart",
                        data=buf,
                        file_name=f"{ticker}_{period}_{interval}_super_zones.png",
                        mime="image/png",
                        help=f"Download the {period}/{interval} chart for {ticker}"
                    )
                else:
                    st.write(f"No data available for {period}/{interval}")
                    trade_log.append(f"Failed to generate chart for {ticker} ({period}/{interval})")
    except Exception as e:
        st.session_state.trade_log.append(f"Error plotting analysis charts for {ticker}: {str(e)}")

# Streamlit app
def main():
    st.set_page_config(page_title="Stock Charting Tool", layout="wide")
    
    st.markdown("""
        <style>
        .css-1y0tads, .css-1v0mbdj, .css-1v3fvcr, .css-1r6slb0, .css-17e7dxy, .css-1d391kg {
            font-size: 12px !important;
        }
        h1, h2, h3 {
            font-size: 14px !important;
            font-weight: bold;
        }
        .main .block-container {
            padding: 1rem;
            background-color: #f9f9f9;
            border-radius: 5px;
        }
        .stTextInput, .stSelectbox {
            margin-bottom: 0.5rem;
        }
        .stButton>button {
            font-size: 12px !important;
            padding: 6px 12px !important;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin: 0 5px;
        }
        .stButton>button:hover {
            background-color: #e0e0e0;
        }
        .css-1v3fvcr .stButton>button {
            width: 100%;
            text-align: left;
            margin-bottom: 0.3rem;
        }
        .css-1v3fvcr .stTextInput {
            margin-bottom: 1rem;
        }
        .stPyplot {
            width: 100% !important;
            max-width: 800px;
            margin: 1rem auto;
            padding: 0.5rem;
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .stExpander {
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <script>
        function copyTradeLog() {
            const textarea = document.querySelector('textarea');
            textarea.select();
            document.execCommand('copy');
            alert('Trade log copied to clipboard!');
        }
        </script>
    """, unsafe_allow_html=True)

    if 'ticker' not in st.session_state:
        st.session_state.ticker = ""
    if 'period' not in st.session_state:
        st.session_state.period = "1mo"
    if 'interval' not in st.session_state:
        st.session_state.interval = "1d"
    if 'limit_lines' not in st.session_state:
        st.session_state.limit_lines = True
    if 'show_prices' not in st.session_state:
        st.session_state.show_prices = False
    if 'enable_ai' not in st.session_state:
        st.session_state.enable_ai = False
    if 'live_update' not in st.session_state:
        st.session_state.live_update = False
    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = []
    if 'show_sidebar' not in st.session_state:
        st.session_state.show_sidebar = True
    if 'search' not in st.session_state:
        st.session_state.search = ""

    with st.sidebar:
        st.subheader("Select Stock")
        if st.button("Toggle Sidebar", help="Show or hide the stock selection panel"):
            st.session_state.show_sidebar = not st.session_state.show_sidebar

        if st.session_state.show_sidebar:
            st.text_input("Search Stocks", key="search", placeholder="e.g., Nifty, RELIANCE", help="Filter stocks by name")
            search_query = st.session_state.search.lower()
            filtered_tickers = [
                ticker for ticker, display_name in display_mapping.items()
                if search_query in display_name.lower()
            ]
            for stock in sorted(filtered_tickers):
                if st.button(display_mapping[stock], key=stock, help=f"View chart for {display_mapping[stock]}"):
                    st.session_state.ticker = stock
                    st.session_state.trade_log.append(f"Selected {stock}")
                    try:
                        fig, buf = plot_chart(stock)
                        if fig and buf:
                            st.session_state.main_fig = fig
                            st.session_state.main_buf = buf
                        else:
                            st.session_state.trade_log.append(f"Failed to plot chart for {stock}")
                    except Exception as e:
                        st.session_state.trade_log.append(f"Error in sidebar plot for {stock}: {str(e)}")

    st.title("Stock Charting Tool")
    
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        ticker = st.text_input("Ticker", value=st.session_state.ticker, placeholder="e.g., RELIANCE, Nifty", help="Enter a stock or index ticker")
        st.session_state.ticker = ticker
    with col2:
        st.session_state.period = st.selectbox("Period", 
            ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'], 
            index=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'].index(st.session_state.period),
            help="Select the time period for the chart")
    with col3:
        st.session_state.interval = st.selectbox("Interval", 
            ['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo'], 
            index=['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo'].index(st.session_state.interval),
            help="Select the time interval for the chart")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.session_state.limit_lines = st.checkbox("Limit Lines", value=st.session_state.limit_lines, help="Show horizontal lines for super zones")
    with col5:
        st.session_state.show_prices = st.checkbox("Prices", value=st.session_state.show_prices, help="Show price levels for super zones")
    with col6:
        st.session_state.enable_ai = st.checkbox("AI", value=st.session_state.enable_ai, help="Enable AI-based signal predictions")

    st.divider()

    col7, col8, col9 = st.columns([1, 1, 3])
    with col7:
        if st.button("Plot", help="Generate chart for the selected ticker"):
            if not st.session_state.ticker:
                st.session_state.trade_log.append("No ticker specified")
            else:
                try:
                    fig, buf = plot_chart(st.session_state.ticker)
                    if fig and buf:
                        st.session_state.main_fig = fig
                        st.session_state.main_buf = buf
                    else:
                        st.session_state.trade_log.append(f"Failed to plot chart for {st.session_state.ticker}")
                except Exception as e:
                    st.session_state.trade_log.append(f"Error in main plot for {st.session_state.ticker}: {str(e)}")
    with col8:
        if st.button("View Full Analysis", help="Show charts for multiple timeframes"):
            if not st.session_state.ticker:
                st.session_state.trade_log.append("No ticker specified")
            else:
                plot_analysis_charts(st.session_state.ticker)

    if 'main_fig' in st.session_state and st.session_state.main_fig:
        st.pyplot(st.session_state.main_fig)
        st.download_button(
            "Save Chart",
            data=st.session_state.main_buf,
            file_name=f"{st.session_state.ticker}_super_zones.png",
            mime="image/png",
            help="Download the main chart"
        )

    with st.expander("Trade Log"):
        trade_log_text = "\n".join(st.session_state.trade_log[-50:])
        st.text_area("Log", value=trade_log_text, height=150, help="View recent actions and errors")
        col_log1, col_log2 = st.columns(2)
        with col_log1:
            if st.button("Copy Log", help="Copy the trade log to clipboard"):
                st.markdown(f'<button onclick="copyTradeLog()">Copy to Clipboard</button>', unsafe_allow_html=True)
        with col_log2:
            st.download_button(
                "Download Log",
                data=trade_log_text,
                file_name="trade_log.txt",
                mime="text/plain",
                help="Download the trade log as a text file"
            )

    if st.session_state.live_update and st.session_state.ticker:
        try:
            fig, buf = plot_chart(st.session_state.ticker)
            if fig and buf:
                st.session_state.main_fig = fig
                st.session_state.main_buf = buf
        except Exception as e:
            st.session_state.trade_log.append(f"Error in live update for {st.session_state.ticker}: {str(e)}")

if __name__ == "__main__":
    main()

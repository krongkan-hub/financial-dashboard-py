import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, datetime
from app.constants import BOND_YIELD_MAP, BOND_BENCHMARK_MAP

# ==============================================================================
# DATA SIMULATION / FETCHING
# ==============================================================================
def fetch_daily_prices(tickers, start_date=datetime(2020, 1, 1).date()):
    """Mocks fetching daily price/yield data for given tickers."""
    end_date = date.today()
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    data = []
    for t in tickers:
        df_temp = pd.DataFrame({'date': date_range})
        
        # --- Yield Simulation ---
        if t in ['^IRX', '^STIX', '^GTII', '^TWS', '^FVX', '^TNX', '^TYX']:
            base_yield = 4.0 + (5 - 4) * (t in ['^IRX', '^STIX', '^GTII']) 
            if t == '^TYX': base_yield -= 0.5 
            df_temp['close'] = base_yield + (np.random.randn(len(date_range)) * 0.4)
            
        # --- Benchmark Simulation ---
        elif t in ['LQD', 'HYG', 'TIP']:
            base_yield = 6.0 if t == 'HYG' else 4.5
            df_temp['close'] = base_yield + (np.random.randn(len(date_range)) * 0.7)
            
        # --- S&P 500 (High Price) ---
        elif t == '^GSPC': 
            df_temp['close'] = 3000 + (np.random.randn(len(date_range)).cumsum() * 10)
        
        # --- Rates/Index (Low/Specific Value) ---
        elif t in ['^MOVE', '^SOFR', '^FFR', 'THB_RATE_1Y']:
            base_val = 5.0 if t in ['^SOFR', '^FFR'] else 120.0
            if t == 'THB_RATE_1Y': base_val = 2.5 
            df_temp['close'] = base_val + (np.random.randn(len(date_range)) * 0.2)
            
        else: # Fallback
            df_temp['close'] = 100 + (np.random.randn(len(date_range)) * 5)
            
        df_temp['ticker'] = t
        data.append(df_temp)
    
    if not data:
        return pd.DataFrame(columns=['date', 'ticker', 'close'])
        
    df = pd.concat(data)
    df['previous_close'] = df.groupby('ticker')['close'].shift(1)
    df['weekly_change'] = df.groupby('ticker')['close'].diff(periods=5) * 10000 
    df['ytd_change'] = df['close'] - df.groupby('ticker')['close'].transform('first')
    return df

def mock_individual_bond_metrics(ticker):
    """Generates mock data for individual bond metrics including Credit Rating, PAR, YTM, and Prices."""
    # Use ticker name to create some variance
    np.random.seed(hash(ticker) % 4294967295) 

    rating = np.random.choice(['AAA', 'AA+', 'A-', 'BBB', 'BB+', 'B-'])
    coupon_rate = np.random.uniform(2.0, 7.0)
    ytm = np.random.uniform(4.0, 6.5)
    
    duration = np.random.uniform(5.0, 15.0)
    convexity = np.random.uniform(0.5, 2.5)
    
    current_price = np.random.uniform(90.0, 110.0)
    
    # Logic for PAR status (Premium/Par/Discount)
    if current_price > 105: status = "Premium Bond"
    elif current_price < 95: status = "Discount Bond"
    else: status = "Par Bond"

    accrued_interest = np.random.uniform(0.1, 1.5)
    clean_price = current_price - accrued_interest
    
    intrinsic_value = current_price * (1 + np.random.uniform(-0.02, 0.05))

    return {
        'Ticker': ticker,
        'Maturity / Benchmark': '10-Year' if ticker == '^TNX' else '5-Year' if ticker == '^FVX' else '30-Year', 
        'Credit Rating (S&P)': rating,
        'Coupon Rate (%)': coupon_rate,
        'YTM (%)': ytm,
        'PAR': status,
        'Duration (Modified)': duration,
        'Convexity': convexity,
        'Clean Price ($)': clean_price,
        'Accrued Interest ($)': accrued_interest,
        'Dirty Price ($)': current_price,
        'Intrinsic Value ($)': intrinsic_value,
        'Valuation Spread (%)': (intrinsic_value / current_price - 1) * 100
    }

# ==============================================================================
# GRAPH GENERATION LOGIC
# ==============================================================================
def generate_yield_history_figure(tickers, indices):
    """
    Generates the 'HISTORICAL YIELDS' Plotly figure.
    This logic is shared between server-side initial render and client-side callbacks.
    """
    all_symbols = list(set(tickers + indices))
    
    if not all_symbols:
        return None # Or return an empty figure with a message

    df_all = fetch_daily_prices(all_symbols)
    if df_all.empty:
        return None

    # --- Combine Maps for Display Names ---
    full_map = {**BOND_YIELD_MAP, **BOND_BENCHMARK_MAP}
    df_all['display_name'] = df_all['ticker'].map(full_map).fillna(df_all['ticker'])
    
    # [MODIFIED LOGIC] Normalize to Cumulative Percentage Change
    # 1. Pivot ข้อมูลให้อยู่ในรูปแบบ Wide Format (Date x Ticker)
    df_pivot = df_all.pivot(index='date', columns='ticker', values='close')
    
    # 2. คำนวณ % Change สะสม: (ราคาปัจจุบัน / ราคา ณ วันแรก) - 1
    # ใช้ iloc[0] คือข้อมูล ณ วันแรกสุดของช่วงเวลานั้นเป็นฐาน (Base = 0%)
    df_normalized = (df_pivot / df_pivot.iloc[0]) - 1
    
    # 3. แปลงกลับเป็น Long Format
    df_normalized_long = df_normalized.reset_index().melt(id_vars='date', var_name='ticker', value_name='cumulative_change')
    
    # 4. Map ชื่อ Display Name กลับเข้าไปใหม่
    ticker_to_display = dict(zip(df_all['ticker'], df_all['display_name']))
    df_normalized_long['display_name'] = df_normalized_long['ticker'].map(ticker_to_display)

    fig = px.line(
        df_normalized_long, 
        x='date', 
        y='cumulative_change', 
        color='display_name',
        title="Comparative Cumulative Change (Normalized to Start Date)",
        labels={'cumulative_change': 'Cumulative Change (%)', 'date': 'Date'},
        template='plotly_dark' 
    )
    # จัดรูปแบบแกน Y เป็นเปอร์เซ็นต์ (เช่น 50%, -10%)
    fig.update_layout(
        yaxis_tickformat=".2%", 
        yaxis_title='Cumulative Change (%)', 
        legend_title='Instrument', 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

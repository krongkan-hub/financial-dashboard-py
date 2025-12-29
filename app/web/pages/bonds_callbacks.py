# [START] app/web/pages/bonds_callbacks.py

import dash
from dash import Dash, dcc, html, dash_table, callback_context
from dash.dependencies import Input, Output, State, ALL, MATCH
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc 
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, datetime, timedelta 
# [MODIFIED] Import BOND_CLASSIFICATIONS (ต้องมีใน app.constants.py)
from app.constants import BOND_YIELD_MAP, BOND_BENCHMARK_MAP, BOND_CLASSIFICATIONS 
from app.web.pages.bonds import BOND_METRIC_DEFINITIONS

# [NEW MOCK DATA] Mock Prices/Yields with all expanded tickers (Treasury, Corporate, Reference)
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

# [NEW MOCK] Mock function for advanced individual bond metrics
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
        'PAR': status, # <--- KEY CHANGED from 'Status' to 'PAR'
        'Duration (Modified)': duration,
        'Convexity': convexity,
        'Clean Price ($)': clean_price,
        'Accrued Interest ($)': accrued_interest,
        'Dirty Price ($)': current_price,
        'Intrinsic Value ($)': intrinsic_value,
        'Valuation Spread (%)': (intrinsic_value / current_price - 1) * 100
    }

# --- Utility Functions (Adapted for Bonds) ---
def get_user_symbols(data):
    """Helper to extract active symbols (Yields and Benchmarks)."""
    tickers = data.get('tickers', [])
    indices = data.get('indices', [])
    all_symbols = list(set(tickers + indices))
    return all_symbols, tickers, indices

# [CRITICAL FIX 1: Standalone Column Generation Helper]
def _generate_datatable_columns_detail(df):
    """Generates column definitions for the detail table with appropriate formatting."""
    columns = []
    for col in df.columns:
        col_def = {"name": col, "id": col}
        # Fixed point formatting for currency/price
        if 'Price' in col or 'Interest' in col or 'Intrinsic Value' in col:
            col_def.update({'type': 'numeric', 'format': Format(precision=3, scheme=Scheme.fixed)})
        # Fixed point formatting for percentages/ratios
        elif '%' in col or 'YTM' in col or 'Coupon' in col or 'Duration' in col or 'Convexity' in col:
            col_def.update({'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.fixed)})
        # Text formatting for Ticker, Rating, PAR status
        else:
             col_def.update({"type": "text"})
             
        columns.append(col_def)
    return columns

# [CRITICAL FIX 2: Simple Data Fetch Helper (returns DataFrame or Alert)]
def _get_detail_table_data(eligible_tickers):
    """Fetches or generates the detailed metrics DataFrame."""
    # Treasury and Corporate ETFs that can be analyzed in detail
    if not eligible_tickers:
        return dbc.Alert("Please select at least one Treasury or Corporate ETF Yield for in-depth analysis.", color="info", className="mt-3 text-center")

    df_details = pd.DataFrame([mock_individual_bond_metrics(t) for t in eligible_tickers])
    
    # Returns DataFrame (length 1) OR Alert (length 1)
    return df_details
    
# [CRITICAL FIX 3: Renderer Helper (always returns 3 elements)]
def _render_detailed_table(df_table, cols_to_show, sort_col):
    """Generates the DataTable and dropdown options based on the data and columns to show."""
    df_table = df_table[cols_to_show].copy()
    
    # Sort options: Exclude descriptive/text columns
    sort_options = [{'label': col, 'value': col} for col in cols_to_show if col not in ['Ticker', 'PAR', 'Credit Rating (S&P)']]

    # Sorting logic
    if sort_col and sort_col in df_table.columns:
        try:
            # Try numeric sort first
            df_table.sort_values(by=sort_col, ascending=False, inplace=True)
        except Exception:
             # Fallback to string sort if numeric fails
             df_table.sort_values(by=sort_col, ascending=False, key=lambda col: col.astype(str), inplace=True)

    table = dash_table.DataTable(
        id='bonds-individual-metrics-table',
        columns=_generate_datatable_columns_detail(df_table), # Uses the standalone helper
        data=df_table.to_dict('records'),
        style_header={'border': '0px', 'backgroundColor': 'transparent', 'fontWeight': '600', 'textTransform': 'uppercase', 'textAlign': 'right'},
        style_data={'border': '0px', 'backgroundColor': 'transparent'},
        style_cell={'textAlign': 'right', 'padding': '12px', 'border': '0px', 'borderBottom': '1px solid #334155', 'fontFamily': '"Open Sans", verdana, arial, sans-serif', 'fontSize': '14px'},
        style_header_conditional=[{'if': {'column_id': 'Ticker'}, 'textAlign': 'left'}],
        style_cell_conditional=[{'if': {'column_id': 'Ticker'}, 'textAlign': 'left', 'width': '10%', 'verticalAlign': 'middle'}],
        markdown_options={"html": True}
    )
    # Returns [children], options, value (Length 3)
    return [table], sort_options, sort_col


# --- Main Registration Function ---
def register_bonds_callbacks(app: Dash, BOND_METRIC_DEFINITIONS):

    # --- 8.1 Load Data Store (bonds-user-selections-store) ---
    @app.callback(
        Output('bonds-user-selections-store', 'data'),
        Input('bonds-add-yield-button', 'n_clicks'),
        Input('bonds-add-benchmark-button', 'n_clicks'),
        Input({'type': 'bonds-remove-ticker-btn', 'index': ALL}, 'n_clicks'),
        Input('url', 'pathname'), 
        State('bonds-user-selections-store', 'data'),
        State('bonds-yield-select-dropdown', 'value'),
        State('bonds-benchmark-select-dropdown', 'value'),
        prevent_initial_call=False
    )
    def load_bonds_data_to_store(add_ticker_clicks, add_index_clicks, remove_clicks, url_pathname, current_data, new_tickers_list, new_indices_list):
        """Initializes the store and manages adding/removing yields and benchmarks."""
        
        initial_data = {
            'tickers': ['^TNX', '^TWS', '^TYX'], 
            'indices': ['^GSPC'],
        }
        
        if url_pathname != '/bonds': return dash.no_update # Only update if we are on the bonds page or initial load
        if not current_data: current_data = initial_data

        ctx = dash.callback_context
        if not ctx.triggered: return initial_data 

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        tickers = set(current_data.get('tickers', []))
        indices = set(current_data.get('indices', []))

        def ensure_list(value):
            if value is None: return []
            return value if isinstance(value, list) else [value]

        if button_id == 'bonds-add-yield-button' and new_tickers_list:
            for t in ensure_list(new_tickers_list): tickers.add(t)
        elif button_id == 'bonds-add-benchmark-button' and new_indices_list:
            for i in ensure_list(new_indices_list): indices.add(i)
        elif 'bonds-remove-ticker-btn' in button_id:
            import json
            triggered_id_str = ctx.triggered[0]['prop_id'].split('.')[0]
            try:
                removed_ticker = json.loads(triggered_id_str)['index']
                if removed_ticker in tickers: tickers.remove(removed_ticker)
                elif removed_ticker in indices: indices.remove(removed_ticker)
            except (json.JSONDecodeError, KeyError): pass 

        if not tickers and not indices: current_data = initial_data
        else:
            current_data['tickers'] = sorted(list(tickers))
            current_data['indices'] = sorted(list(indices))

        return current_data

    # --- [MODIFIED] 8.2 Update the available options based on Sector Dropdown ---
    @app.callback(
        # Output('bonds-yield-select-dropdown', 'options') คือ Dropdown ตัวล่าง
        Output('bonds-yield-select-dropdown', 'options'),
        Input('bonds-yield-sector-dropdown', 'value'), # <<< Input จาก Dropdown Sector ใหม่
        State('bonds-user-selections-store', 'data'),
    )
    def update_bond_options(selected_sector, data):
        """
        Populates the Yields dropdown based on the selected sector/classification.
        """
        # ดึง Tickers ทั้งหมดที่ถูกเลือกอยู่แล้ว (เพื่อไม่ให้แสดงซ้ำ)
        selected_tickers = set(data.get('tickers', []))
        
        # 1. กรอง Tickers ตาม Sector ที่เลือก
        # ใช้ BOND_CLASSIFICATIONS เพื่อดึง Tickers ที่เกี่ยวข้อง
        # .get(selected_sector, {}) ใช้ป้องกัน Error หาก selected_sector เป็น None หรือไม่ถูกต้อง
        relevant_tickers = BOND_CLASSIFICATIONS.get(selected_sector, BOND_CLASSIFICATIONS['All']).get('tickers', [])
        
        # 2. สร้าง Options (กรองตัวที่เลือกแล้วออก)
        options = []
        for ticker in relevant_tickers:
            if ticker not in selected_tickers:
                # ใช้ BOND_YIELD_MAP ในการแสดงชื่อเต็ม
                label = BOND_YIELD_MAP.get(ticker, ticker)
                options.append({'label': label, 'value': ticker})
        
        return options
    # --- [END MODIFIED] ---

    # --- [MODIFIED] 8.3 Benchmark options (เปลี่ยน Input ID) ---
    @app.callback(
        Output('bonds-benchmark-select-dropdown', 'options'),
        Input('bonds-yield-sector-dropdown', 'value'), # <<< Input Dummy เพื่อให้ Trigger เมื่อโหลดหน้า
    )
    def update_bond_benchmark_options(dummy_input):
        # Benchmark Dropdown ยังคงแสดงตัวเลือกเดิมทั้งหมด
        options = [{'label': name, 'value': ticker} for ticker, name in BOND_BENCHMARK_MAP.items()]
        return options
    # --- [END MODIFIED] ---

    @app.callback(
        Output('bonds-summary-display', 'children'),
        Output('bonds-benchmark-summary-display', 'children'),
        Input('bonds-user-selections-store', 'data')
    )
    def update_summary_display(data):
        if not data: data = {'tickers': [], 'indices': []}
        tickers, indices = data.get('tickers', []), data.get('indices', [])
        full_map = {**BOND_YIELD_MAP, **BOND_BENCHMARK_MAP}
        
        if not tickers: ticker_content = html.Div([html.Span("No Yields selected.", className="text-white-50 fst-italic")])
        else:
            ticker_content = [html.Label("Selected Yields:", className="text-white small fw-bold")] + [
                dbc.Badge([full_map.get(t, t), html.I(className="bi bi-x-circle-fill ms-2", style={'cursor': 'pointer'}, id={'type': 'bonds-remove-ticker-btn', 'index': t})], color="dark", className="m-1 p-2 border", style={"backgroundColor": "#334155", "borderColor": "#475569", "color": "#f8fafc"}) for t in tickers
            ]
        
        if not indices: index_content = html.Div([html.Span("No Benchmarks selected.", className="text-white-50 fst-italic")])
        else:
            index_content = [html.Label("Selected Benchmarks:", className="text-white small fw-bold")] + [
                dbc.Badge([full_map.get(t, t), html.I(className="bi bi-x-circle-fill ms-2", style={'cursor': 'pointer'}, id={'type': 'bonds-remove-ticker-btn', 'index': t})], color="dark", className="m-1 p-2 border", style={"backgroundColor": "#334155", "borderColor": "#475569", "color": "#f8fafc"})
                for t in indices
            ]
        return ticker_content, index_content


    # --- 8.4 Render Graph Content (MODIFIED with NORMALIZATION) ---
    @app.callback(
        Output('bonds-analysis-pane-content', 'children'),
        Input('bonds-analysis-tabs', 'active_tab'),
        Input('bonds-user-selections-store', 'data'),
        prevent_initial_call=True
    )
    def render_bond_graph_content(active_tab, data): 
        
        all_symbols, tickers, indices = get_user_symbols(data)
        
        if not all_symbols:
            return dbc.Alert("Please select at least one Treasury Yield for analysis.", color="info", className="mt-3 text-center")

        df_all = fetch_daily_prices(all_symbols)
        if df_all.empty:
            return dbc.Alert("No data available for selected instruments.", color="warning", className="mt-3 text-center")
        
        # กำหนดชื่อแกน Y ให้เป็นค่าเริ่มต้นเสมอ (Yield (%) / Price)
        y_axis_title = 'Yield (%) / Price'
            
        # --- Combine Maps for Display Names ---
        full_map = {**BOND_YIELD_MAP, **BOND_BENCHMARK_MAP}
        df_all['display_name'] = df_all['ticker'].map(full_map).fillna(df_all['ticker'])
        
        # --- Tab: HISTORICAL YIELDS (tab-yield-history) ---
        if active_tab == "tab-yield-history":
            # [MODIFIED LOGIC] Normalize to Cumulative Percentage Change
            # 1. Pivot ข้อมูลให้อยู่ในรูปแบบ Wide Format (Date x Ticker) เพื่อให้คำนวณง่าย
            df_pivot = df_all.pivot(index='date', columns='ticker', values='close')
            
            # 2. คำนวณ % Change สะสม: (ราคาปัจจุบัน / ราคา ณ วันแรก) - 1
            # ใช้ iloc[0] คือข้อมูล ณ วันแรกสุดของช่วงเวลานั้นเป็นฐาน (Base = 0%)
            df_normalized = (df_pivot / df_pivot.iloc[0]) - 1
            
            # 3. แปลงกลับเป็น Long Format เพื่อนำไปพลอตกราฟด้วย Plotly Express
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
                template='plotly_dark' # [MODIFIED] Added Dark Template
            )
            # จัดรูปแบบแกน Y เป็นเปอร์เซ็นต์ (เช่น 50%, -10%)
            fig.update_layout(yaxis_tickformat=".2%", yaxis_title='Cumulative Change (%)', legend_title='Instrument', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            return dcc.Graph(figure=fig)

        # --- Tab: YIELD CURVE (tab-yield-curve) ---
        elif active_tab == "tab-yield-curve":
            treasury_yields = {
                '^IRX': 0.08, '^STIX': 0.25, '^GTII': 0.5, # T-Bills
                '^TWS': 2, '^FVX': 5, '^TNX': 10, '^TYX': 30
            }
            
            active_yield_tickers = [t for t in tickers if t in treasury_yields]
            if not active_yield_tickers:
                 return dbc.Alert("Please select US Treasury Yields (^TWS, ^TNX, ^TYX, etc.) to plot the Yield Curve.", color="info", className="mt-3 text-center")

            latest_date = df_all['date'].max()
            compare_dates = {
                "Today": latest_date,
                "Last Week": latest_date - timedelta(weeks=1),
                "Last Month": latest_date - timedelta(days=30),
                "Last Year": latest_date - timedelta(days=365),
            }
            
            df_curve_data = []
            for name, comp_date in compare_dates.items():
                # Filter by active yields and copy to avoid SettingWithCopyWarning
                df_closest_date = df_all[df_all['ticker'].isin(active_yield_tickers)].copy()
                
                # Find the closest date for each period, per ticker
                df_closest_date['diff'] = (df_closest_date['date'] - comp_date).abs()
                
                # Group by ticker and take the row with minimum 'diff' (closest date)
                df_curve = df_closest_date.loc[df_closest_date.groupby('ticker')['diff'].idxmin()].copy()
                
                df_curve['Maturity_Years'] = df_curve['ticker'].map(treasury_yields)
                df_curve['Period'] = name
                df_curve_data.append(df_curve)

            df_curve_final = pd.concat(df_curve_data)
            
            fig = px.line(
                df_curve_final,
                x='Maturity_Years',
                y='close',
                color='Period',
                markers=True,
                line_shape='spline',
                title=f"US Treasury Yield Curve Comparison ({y_axis_title})",
                template='plotly_dark' # [MODIFIED] Added Dark Template
            )
            fig.update_layout(
                xaxis_title='Maturity (Years)',
                yaxis_title=y_axis_title,
                xaxis={'tickmode': 'linear', 'tick0': 0, 'dtick': 5},
                hovermode="x unified",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)' # [MODIFIED] Transparent
            )
            return dcc.Graph(figure=fig)


        # --- Tab: YIELD SPREAD (tab-yield-spread) ---
        elif active_tab == "tab-yield-spread":
            if len(all_symbols) < 2:
                return dbc.Alert("Please select at least two instruments (Yields/Benchmanks) to calculate a Spread.", color="info", className="mt-3 text-center")
            
            benchmark = all_symbols[0] 
            series_to_compare = all_symbols[1:] 
            
            if not series_to_compare:
                 return dbc.Alert(f"Please select at least two instruments. Benchmark is set to {full_map.get(benchmark, benchmark)}.", color="info", className="mt-3 text-center")

            df_pivot = df_all.pivot(index='date', columns='ticker', values='close').dropna(how='all').reset_index()
            
            spread_data = []
            
            if benchmark not in df_pivot.columns:
                 return dbc.Alert(f"Benchmark '{full_map.get(benchmark, benchmark)}' data missing from the fetched results.", color="warning", className="mt-3 text-center")
            
            for series in series_to_compare:
                if series in df_pivot.columns:
                    df_series = df_pivot.dropna(subset=[benchmark, series]).copy()
                    
                    # Spread is in bps (100 * (Y_Series - Y_Benchmark))
                    df_series['Spread'] = (df_series[series] - df_series[benchmark]) * 100 
                    df_series['Spread Name'] = f"{full_map.get(series, series)} - {full_map.get(benchmark, benchmark)}"
                    df_series['Ticker'] = series
                    spread_data.append(df_series[['date', 'Spread', 'Spread Name', 'Ticker']])
                else:
                    return dbc.Alert(f"Series '{full_map.get(series, series)}' data missing from the fetched results.", color="warning", className="mt-3 text-center")

            df_final_spread = pd.concat(spread_data)

            fig = px.line(
                df_final_spread,
                x='date',
                y='Spread',
                color='Spread Name',
                title=f"Historical Yield/Price Spreads vs. {full_map.get(benchmark, benchmark)}",
                labels={'Spread': 'Spread Value (bps)', 'date': 'Date'},
                template='plotly_dark' # [MODIFIED] Added Dark Template
            )
            
            fig.update_layout(yaxis_title='Spread Value (bps)', hovermode="x unified", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            return dcc.Graph(figure=fig)

        # --- Tab: YIELD VOLATILITY (tab-yield-volatility) ---
        elif active_tab == "tab-yield-volatility":
            move_ticker = '^MOVE'
            df_move = df_all[df_all['ticker'] == move_ticker].copy()
            
            if df_move.empty:
                return dbc.Alert(f"The MOVE Index ({move_ticker}) is not currently available in the selected data set.", color="warning")

            fig = px.line(
                df_move, 
                x='date', 
                y='close', 
                title="MOVE Index (US Bond Market Volatility)",
                labels={'close': 'MOVE Index Value', 'date': 'Date'},
                template='plotly_dark' # [MODIFIED] Added Dark Template
            )
            fig.update_layout(yaxis_title='MOVE Index Value', legend_title='Instrument', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            return dcc.Graph(figure=fig)

        return html.P(f"Content for {active_tab} is not implemented yet.")


    # --- 8.5 Render Table Content (MODIFIED) ---
    @app.callback(
        Output('bonds-table-pane-content', 'children'),
        Output('bonds-sort-by-dropdown', 'options'),
        Output('bonds-sort-by-dropdown', 'value'),
        Input('bonds-table-tabs', 'active_tab'),
        Input('bonds-user-selections-store', 'data'),
        Input('bonds-sort-by-dropdown', 'value'),
        prevent_initial_call=True
    )
    def render_bond_table_content(active_tab, data, sort_by):
        
        all_symbols, tickers, indices = get_user_symbols(data)
        full_map = {**BOND_YIELD_MAP, **BOND_BENCHMARK_MAP}

        # --- Helper for Detailed Metric Tabs ---
        # Only Treasury and Corporate ETF yields are eligible for mock detail analysis
        eligible_tickers = [t for t in tickers if t in ['^IRX', '^TWS', '^TNX', '^FVX', '^TYX', 'LQD', 'HYG', 'TIP']]
        
        # --- Tab: CREDIT & STATUS (tab-bond-credit) ---
        if active_tab == "tab-bond-credit": 
            df_details_or_alert = _get_detail_table_data(eligible_tickers)
            
            if isinstance(df_details_or_alert, dbc.Alert):
                 return df_details_or_alert, [], None

            df_details = df_details_or_alert
            # UPDATED: Added PAR, Coupon Rate (%), YTM (%)
            cols_to_show = ['Ticker', 'Credit Rating (S&P)', 'PAR', 'Coupon Rate (%)', 'YTM (%)'] 
            return _render_detailed_table(df_details, cols_to_show, sort_by) 


        # --- Tab: DURATION & RISK (tab-bond-risk) ---
        elif active_tab == "tab-bond-risk":
            df_details_or_alert = _get_detail_table_data(eligible_tickers)
            
            if isinstance(df_details_or_alert, dbc.Alert):
                 return df_details_or_alert, [], None

            df_details = df_details_or_alert
            cols_to_show = ['Ticker', 'Duration (Modified)', 'Convexity', 'Valuation Spread (%)']
            return _render_detailed_table(df_details, cols_to_show, sort_by)


        # --- Tab: YIELD & DIRTY PRICE (tab-bond-pricing) ---
        elif active_tab == "tab-bond-pricing":
            df_details_or_alert = _get_detail_table_data(eligible_tickers)
            
            if isinstance(df_details_or_alert, dbc.Alert):
                 return df_details_or_alert, [], None

            df_details = df_details_or_alert
            # UPDATED: Removed moved metrics, kept Dirty Price ($) + Valuation Spread (%)
            cols_to_show = ['Ticker', 'Clean Price ($)', 'Accrued Interest ($)', 'Dirty Price ($)', 'Valuation Spread (%)']
            return _render_detailed_table(df_details, cols_to_show, sort_by)


        return html.P(f"Table content not available for {active_tab}."), [], None


    # --- Modal Callbacks (Unchanged Logic, Corrected IDs) ---
    @app.callback(
        Output("bonds-definitions-modal", "is_open"),
        Output("bonds-definitions-modal-content", "children"),
        [
            Input("bonds-open-definitions-modal-btn-graphs", "n_clicks"),
            Input("bonds-open-definitions-modal-btn-tables", "n_clicks"),
            Input("bonds-close-definitions-modal", "n_clicks"),
            Input("bonds-analysis-tabs", "active_tab"),
            Input("bonds-table-tabs", "active_tab"),
        ],
        [State("bonds-definitions-modal", "is_open")],
        prevent_initial_call=False
    )
    def toggle_definitions_modal(n_graphs, n_tables, n_close, graph_tab, table_tab, is_open):
        ctx = dash.callback_context 
        if not ctx.triggered: return is_open, html.P("Select a tab for definitions.")
        prop_id = ctx.triggered[0]['prop_id']

        def render_definitions(tab_id):
            definition = BOND_METRIC_DEFINITIONS.get(tab_id, {})
            if not definition: return html.P("Definition not found for this tab.")
            content = [
                html.H4(definition.get('title')),
                html.P(definition.get('description')),
                html.Ul([html.Li(html.B(f"{m['metric']}: "), html.Span(m['definition'])) for m in definition.get('metrics', [])])
            ]
            return content

        if 'bonds-open-definitions-modal-btn' in prop_id:
            active_tab = graph_tab if 'graphs' in prop_id else table_tab
            return True, render_definitions(active_tab)
        elif prop_id == "bonds-close-definitions-modal.n_clicks":
            return False, html.P("Select a tab for definitions.")
        elif 'active_tab' in prop_id:
            if is_open:
                content = render_definitions(graph_tab) if 'analysis-tabs' in prop_id else render_definitions(table_tab)
                return is_open, content
        
        return is_open, html.P("Select a tab for definitions.")

    # Visibility Callbacks (Permanent Fix)
    @app.callback(Output('bonds-open-dcf-modal-btn', 'style'), Input('bonds-analysis-tabs', 'active_tab'))
    def toggle_dcf_gear_button_visibility(active_tab): return {'display': 'none'} 

    @app.callback(Output('bonds-open-forecast-modal-btn', 'style'), Input('bonds-table-tabs', 'active_tab'))
    def toggle_forecast_gear_button_visibility(active_tab): return {'display': 'none'}
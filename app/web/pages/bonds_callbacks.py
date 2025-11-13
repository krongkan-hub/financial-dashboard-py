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
from datetime import date, datetime, timedelta # [MODIFIED: Add timedelta]
from app.constants import BOND_YIELD_MAP, BOND_BENCHMARK_MAP 
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
            base_yield = 4.0 + (5 - 4) * (t in ['^IRX', '^STIX', '^GTII']) # T-Bills are often higher
            if t == '^TYX': base_yield -= 0.5 # Long end often lower
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
    # [NEW] Calculate 1 Week/YTD Change (Mock)
    df['weekly_change'] = df.groupby('ticker')['close'].diff(periods=5) * 10000 
    df['ytd_change'] = df.groupby('ticker')['close'] - df.groupby('ticker')['close'].transform(lambda x: x.iloc[0]) # Mock YTD difference
    return df

# [NEW MOCK] Mock function for advanced individual bond metrics
def mock_individual_bond_metrics(ticker):
    # This is a complex function and should return a dict/DataFrame for the selected ticker.
    # We mock it to include all required fields: Rating, Coupon, YTM, Duration, Pricing Status
    
    # Use ticker name to create some variance
    np.random.seed(hash(ticker) % 4294967295) # Simple seed based on string

    rating = np.random.choice(['AAA', 'AA+', 'A-', 'BBB', 'BB+', 'B-'])
    coupon_rate = np.random.uniform(2.0, 7.0)
    ytm = np.random.uniform(4.0, 6.5)
    
    duration = np.random.uniform(5.0, 15.0)
    convexity = np.random.uniform(0.5, 2.5)
    
    current_price = np.random.uniform(90.0, 110.0)
    par_value = 100.0
    
    if current_price > 105: status = "Premium Bond"
    elif current_price < 95: status = "Discount Bond"
    else: status = "Par Bond"

    accrued_interest = np.random.uniform(0.1, 1.5)
    clean_price = current_price - accrued_interest
    
    # Intrinsic Value Mock: Simple 5% valuation buffer
    intrinsic_value = current_price * (1 + np.random.uniform(-0.02, 0.05))

    return {
        'Ticker': ticker,
        'Maturity': '10-Year' if ticker == '^TNX' else '5-Year' if ticker == '^FVX' else '30-Year',
        'Credit Rating (S&P)': rating,
        'Coupon Rate (%)': coupon_rate,
        'YTM (%)': ytm,
        'Status': status,
        'Duration (Modified)': duration,
        'Convexity': convexity,
        'Clean Price ($)': clean_price,
        'Accrued Interest ($)': accrued_interest,
        'Dirty Price ($)': current_price,
        'Intrinsic Value ($)': intrinsic_value,
        'Valuation Spread (%)': (intrinsic_value / current_price - 1) * 100
    }

# --- Utility Functions (Adapted for Bonds) ---
# ... (get_user_symbols เดิม) ...
def get_user_symbols(data):
    """Helper to extract active symbols (Yields and Benchmarks)."""
    tickers = data.get('tickers', [])
    indices = data.get('indices', [])
    all_symbols = list(set(tickers + indices))
    return all_symbols, tickers, indices

# [NEW HELPER]
def apply_thb_hedging(df, thb_rate_proxy=0.025, usd_rate_proxy=0.055):
    """
    Applies the THB Hedged Yield adjustment to the 'close' column (in percentage points).
    THB Hedged Yield = USD Yield + (THB Rate - USD Rate)
    """
    if df.empty: return df
    
    df_adj = df.copy()
    
    # Assuming the USD Yield ('close') is in percent, convert all rates to decimal first
    # Rate Differential (RD) = THB Rate - USD Rate
    rate_differential = thb_rate_proxy - usd_rate_proxy
    
    # We apply the RD to the yield in percentage points
    df_adj['close_hedged'] = df_adj['close'] + (rate_differential * 100)
    return df_adj

# --- Main Registration Function ---
def register_bonds_callbacks(app: Dash, BOND_METRIC_DEFINITIONS):

    # --- 8.1 Load Data Store (bonds-user-selections-store) ---
    # ... (Callback เดิม) ...
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
            'tickers': ['^TNX', '^TWS', '^TYX'], # [MODIFIED] Set default to Yield Curve components
            'indices': ['^GSPC'],
        }
        
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
    # ... (Callbacks Update Dropdowns and Summary Display เดิม) ...

    @app.callback(
        Output('bonds-yield-select-dropdown', 'options'),
        Input('bonds-yield-type-dropdown', 'value'), 
    )
    def update_bond_options(selected_yield_type):
        options = [{'label': name, 'value': ticker} for ticker, name in BOND_YIELD_MAP.items()]
        return options

    @app.callback(
        Output('bonds-benchmark-select-dropdown', 'options'),
        Input('bonds-yield-type-dropdown', 'value'), 
    )
    def update_bond_benchmark_options(dummy_input):
        options = [{'label': name, 'value': ticker} for ticker, name in BOND_BENCHMARK_MAP.items()]
        return options

    @app.callback(
        Output('bonds-summary-display', 'children'),
        Output('bonds-benchmark-summary-display', 'children'),
        Input('bonds-user-selections-store', 'data')
    )
    def update_summary_display(data):
        if not data: data = {'tickers': [], 'indices': []}
        tickers, indices = data.get('tickers', []), data.get('indices', [])
        full_map = {**BOND_YIELD_MAP, **BOND_BENCHMARK_MAP}
        
        if not tickers: ticker_content = html.Div([html.Span("No Yields selected.", className="text-muted fst-italic")])
        else:
            ticker_content = [html.Label("Selected Yields:", className="text-muted small")] + [
                dbc.Badge([full_map.get(t, t), html.I(className="bi bi-x-circle-fill ms-2", style={'cursor': 'pointer'}, id={'type': 'bonds-remove-ticker-btn', 'index': t})], color="light", className="m-1 p-2 text-dark border") for t in tickers
            ]
        
        if not indices: index_content = html.Div([html.Span("No Benchmarks selected.", className="text-muted fst-italic")])
        else:
            index_content = [html.Label("Selected Benchmarks:", className="text-muted small")] + [
                dbc.Badge([full_map.get(t, t), html.I(className="bi bi-x-circle-fill ms-2", style={'cursor': 'pointer'}, id={'type': 'bonds-remove-ticker-btn', 'index': t})], color="light", className="m-1 p-2 text-dark border")
                for t in indices
            ]
        return ticker_content, index_content


    # --- 8.4 Render Graph Content ---
    @app.callback(
        Output('bonds-analysis-pane-content', 'children'),
        Input('bonds-analysis-tabs', 'active_tab'),
        Input('bonds-user-selections-store', 'data'),
        Input('bonds-thb-view-toggle', 'value'), # [NEW INPUT]
        State('bonds-fx-rates-store', 'data'),  # [NEW STATE]
        prevent_initial_call=True
    )
    def render_bond_graph_content(active_tab, data, thb_view_enabled, fx_rates):
        
        all_symbols, tickers, indices = get_user_symbols(data)
        thb_view_active = bool(thb_view_enabled)
        
        if not all_symbols:
            return dbc.Alert("Please select at least one Treasury Yield for analysis.", color="info", className="mt-3 text-center")

        df_all = fetch_daily_prices(all_symbols)
        if df_all.empty:
            return dbc.Alert("No data available for selected instruments.", color="warning", className="mt-3 text-center")
        
        # --- Apply THB Hedging if enabled ---
        if thb_view_active:
            thb_rate = fx_rates.get('THB_RATE_1Y', 0.025)
            usd_rate = fx_rates.get('USD_RATE_1Y', 0.055)
            # Only apply hedging to yields (tickers)
            df_yields = df_all[df_all['ticker'].isin(tickers)].copy()
            df_benchmarks = df_all[df_all['ticker'].isin(indices)].copy()
            
            df_yields = apply_thb_hedging(df_yields, thb_rate, usd_rate)
            df_yields['close'] = df_yields['close_hedged']
            
            df_all = pd.concat([df_yields, df_benchmarks])
            y_axis_title = 'Yield (THB Hedged %) / Price'
        else:
            y_axis_title = 'Yield (%) / Price'
            
        # --- Combine Maps for Display Names ---
        full_map = {**BOND_YIELD_MAP, **BOND_BENCHMARK_MAP}
        df_all['display_name'] = df_all['ticker'].map(full_map).fillna(df_all['ticker'])
        
        # --- Tab: HISTORICAL YIELDS (tab-yield-history) ---
        if active_tab == "tab-yield-history":
            fig = px.line(
                df_all, 
                x='date', 
                y='close', 
                color='display_name',
                title="Historical Yields and Benchmarks",
                labels={'close': y_axis_title, 'date': 'Date'}
            )
            fig.update_layout(yaxis_title=y_axis_title, legend_title='Instrument')
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

            # Mock old dates for comparison lines
            latest_date = df_all['date'].max()
            compare_dates = {
                "Today": latest_date,
                "Last Week": latest_date - timedelta(weeks=1),
                "Last Month": latest_date - timedelta(days=30),
                "Last Year": latest_date - timedelta(days=365),
            }
            
            df_curve_data = []
            for name, comp_date in compare_dates.items():
                # Find the closest available date in the dataset
                df_closest_date = df_all[df_all['ticker'].isin(active_yield_tickers)].iloc[(df_all['date'] - comp_date).abs().argsort()[:len(active_yield_tickers)]].copy()
                
                df_curve = df_closest_date[df_closest_date['ticker'].isin(active_yield_tickers)].copy()
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
                title=f"US Treasury Yield Curve Comparison ({y_axis_title})"
            )
            fig.update_layout(
                xaxis_title='Maturity (Years)',
                yaxis_title=y_axis_title,
                xaxis={'tickmode': 'linear', 'tick0': 0, 'dtick': 5},
                hovermode="x unified"
            )
            return dcc.Graph(figure=fig)


        # --- Tab: YIELD SPREAD (tab-yield-spread) ---
        elif active_tab == "tab-yield-spread":
            # --- [NEW LOGIC: Single Benchmark Rule] ---
            if len(all_symbols) < 2:
                return dbc.Alert("Please select at least two instruments (Yields/Benchmarks) to calculate a Spread.", color="info", className="mt-3 text-center")
            
            # Use the first index as the required single benchmark
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
                    
                    # Calculate Spread (Series - Benchmark)
                    df_series['Spread'] = (df_series[series] - df_series[benchmark]) * 100
                    df_series['Spread Name'] = f"{full_map.get(series, series)} - {full_map.get(benchmark, benchmark)}"
                    df_series['Ticker'] = series
                    spread_data.append(df_series[['date', 'Spread', 'Spread Name', 'Ticker']])
                else:
                    return dbc.Alert(f"Series '{full_map.get(series, series)}' data missing from the fetched results.", color="warning", className="mt-3 text-center")

            df_final_spread = pd.concat(spread_data)

            # Plotting the Spread
            fig = px.line(
                df_final_spread,
                x='date',
                y='Spread',
                color='Spread Name',
                title=f"Historical Yield/Price Spreads vs. {full_map.get(benchmark, benchmark)}",
                labels={'Spread': 'Spread Value (bps)', 'date': 'Date'}
            )
            
            fig.update_layout(yaxis_title='Spread Value (bps)', hovermode="x unified")
            return dcc.Graph(figure=fig)

        # --- Tab: YIELD VOLATILITY (tab-yield-volatility) ---
        elif active_tab == "tab-yield-volatility":
            # The MOVE Index is already mocked in fetch_daily_prices
            move_ticker = '^MOVE'
            df_move = df_all[df_all['ticker'] == move_ticker].copy()
            
            if df_move.empty:
                return dbc.Alert(f"The MOVE Index ({move_ticker}) is not currently available in the selected data set.", color="warning")

            fig = px.line(
                df_move, 
                x='date', 
                y='close', 
                title="MOVE Index (US Bond Market Volatility)",
                labels={'close': 'MOVE Index Value', 'date': 'Date'}
            )
            fig.update_layout(yaxis_title='MOVE Index Value', legend_title='Instrument')
            return dcc.Graph(figure=fig)

        return html.P(f"Content for {active_tab} is not implemented yet.")


    # --- 8.5 Render Table Content ---
    @app.callback(
        Output('bonds-table-pane-content', 'children'),
        Output('bonds-sort-by-dropdown', 'options'),
        Output('bonds-sort-by-dropdown', 'value'),
        Input('bonds-table-tabs', 'active_tab'),
        Input('bonds-user-selections-store', 'data'),
        Input('bonds-sort-by-dropdown', 'value'),
        Input('bonds-thb-view-toggle', 'value'), # [NEW INPUT]
        State('bonds-fx-rates-store', 'data'),  # [NEW STATE]
        prevent_initial_call=True
    )
    def render_bond_table_content(active_tab, data, sort_by, thb_view_enabled, fx_rates):
        
        all_symbols, tickers, indices = get_user_symbols(data)
        thb_view_active = bool(thb_view_enabled)
        full_map = {**BOND_YIELD_MAP, **BOND_BENCHMARK_MAP}

        if not all_symbols:
            return dbc.Alert("Please select at least one instrument for the summary table.", color="info", className="mt-3 text-center"), [], None
        
        # --- Fetch and Adjust Data ---
        df_daily = fetch_daily_prices(all_symbols)
        if df_daily.empty:
            return dbc.Alert("No data available to generate the summary table.", color="warning", className="mt-3 text-center"), [], None

        if thb_view_active:
            thb_rate = fx_rates.get('THB_RATE_1Y', 0.025)
            usd_rate = fx_rates.get('USD_RATE_1Y', 0.055)
            df_daily_adj = apply_thb_hedging(df_daily.copy(), thb_rate, usd_rate)
            df_daily_adj.loc[df_daily_adj['ticker'].isin(tickers), 'close'] = df_daily_adj['close_hedged']
            df_daily = df_daily_adj
            yield_col_name = 'Latest Yield (THB Hedged %)'
        else:
            yield_col_name = 'Latest Yield (%)'

        # --- Tab: RATES SUMMARY (tab-rates-summary) ---
        if active_tab == "tab-rates-summary":
            latest_date = df_daily['date'].max()
            df_latest = df_daily[df_daily['date'] == latest_date].copy()
            
            # --- [NEW] Calculate Daily/Weekly/YTD Change (Mocked) ---
            df_summary = df_latest[['ticker', 'close', 'previous_close', 'weekly_change', 'ytd_change']].rename(columns={'close': 'latest_yield'})
            df_summary['1-Day Change (bps)'] = (df_summary['latest_yield'] - df_summary['previous_close']) * 10000
            df_summary['1-Week Change (bps)'] = df_summary['weekly_change'] # Already in bps
            df_summary['YTD Change (%)'] = df_summary['ytd_change'] # Mock YTD change
            
            # --- [NEW] Calculate Key Spreads ---
            df_spread = df_daily.pivot(index='date', columns='ticker', values='close')
            spreads = []
            
            # Mock 10Y-2Y Spread
            if '^TNX' in df_spread.columns and '^TWS' in df_spread.columns:
                spread_val = (df_spread['^TNX'] - df_spread['^TWS']).iloc[-1] * 100 # Last point in bps
                spreads.append({
                    'Maturity / Benchmark': 'Spread: US 10Y - 2Y (bps)',
                    'latest_yield': spread_val,
                    '1-Day Change (bps)': np.nan, 
                    '1-Week Change (bps)': np.nan, 
                    'YTD Change (%)': np.nan 
                })
                
            # Mock Credit Spread (HYG - 10Y)
            if 'HYG' in df_spread.columns and '^TNX' in df_spread.columns:
                spread_val = (df_spread['HYG'] - df_spread['^TNX']).iloc[-1] * 100 # Last point in bps
                spreads.append({
                    'Maturity / Benchmark': 'Spread: HYG - 10Y (bps)',
                    'latest_yield': spread_val,
                    '1-Day Change (bps)': np.nan, 
                    '1-Week Change (bps)': np.nan, 
                    'YTD Change (%)': np.nan 
                })

            df_summary['Maturity / Benchmark'] = df_summary['ticker'].map(full_map).fillna(df_summary['ticker'])
            df_table = df_summary[[
                'Maturity / Benchmark', 
                'latest_yield', 
                '1-Day Change (bps)', 
                '1-Week Change (bps)',
                'YTD Change (%)'
            ]].copy()
            
            # Concat with spreads
            df_table = pd.concat([df_table, pd.DataFrame(spreads)]).fillna(value={'latest_yield': np.nan})

            # Rename columns for display
            df_table.columns = [
                'Maturity / Benchmark', 
                yield_col_name, 
                '1-Day Change (bps)', 
                '1-Week Change (bps)',
                'YTD Change (%)'
            ]
            
            sort_cols = [c for c in df_table.columns if c not in ['Maturity / Benchmark']]
            sort_options = [{'label': col, 'value': col} for col in sort_cols]
            
            if sort_by and sort_by in df_table.columns:
                df_table = df_table.sort_values(by=sort_by, ascending=False)
            
            def _generate_datatable_columns(df):
                columns = []
                for col in df.columns:
                    col_def = {"name": col, "id": col}
                    if 'Yield' in col or 'Price' in col:
                        col_def.update({'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.fixed)})
                    elif 'bps' in col:
                        col_def.update({'type': 'numeric', 'format': Format(precision=1, scheme=Scheme.fixed)})
                    elif '%' in col:
                         col_def.update({'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.fixed)})
                    columns.append(col_def)
                return columns

            table = dash_table.DataTable(
                id='bonds-rates-summary-table',
                columns=_generate_datatable_columns(df_table),
                data=df_table.to_dict('records'),
                # ... (Style เดิม) ...
                style_header={'border': '0px', 'backgroundColor': 'transparent', 'fontWeight': '600', 'textTransform': 'uppercase', 'textAlign': 'right'},
                style_data={'border': '0px', 'backgroundColor': 'transparent'},
                style_cell={'textAlign': 'right', 'padding': '14px', 'border': '0px', 'borderBottom': '1px solid #f0f0f0'},
                style_header_conditional=[{'if': {'column_id': 'Maturity / Benchmark'}, 'textAlign': 'left'}],
                style_cell_conditional=[{'if': {'column_id': 'Maturity / Benchmark'}, 'textAlign': 'left', 'width': '40%', 'verticalAlign': 'middle'}],
                markdown_options={"html": True}
            )
            
            return [table], sort_options, sort_by

        # --- Tab: INDIVIDUAL METRICS (tab-individual-metrics) ---
        elif active_tab == "tab-individual-metrics":
            # Focus only on the primary yield tickers for detail view
            eligible_tickers = [t for t in tickers if t in ['^IRX', '^TWS', '^TNX', '^FVX', '^TYX', 'LQD', 'HYG', 'TIP']]
            
            if not eligible_tickers:
                 return dbc.Alert("Please select at least one core Treasury or Corporate ETF Yield for in-depth analysis.", color="info", className="mt-3 text-center"), [], None

            # Mock detailed metrics for all eligible tickers
            df_details = pd.DataFrame([mock_individual_bond_metrics(t) for t in eligible_tickers])

            # Select and format columns
            cols_to_show = [
                'Ticker', 'Credit Rating (S&P)', 'Status', 'Coupon Rate (%)', 
                'YTM (%)', 'Duration (Modified)', 'Convexity', 
                'Clean Price ($)', 'Accrued Interest ($)', 'Dirty Price ($)',
                'Valuation Spread (%)'
            ]
            
            df_table = df_details[cols_to_show].copy()
            sort_options = [{'label': col, 'value': col} for col in cols_to_show if col not in ['Ticker', 'Status', 'Credit Rating (S&P)']]
            
            if sort_by and sort_by in df_table.columns:
                df_table = df_table.sort_values(by=sort_by, ascending=False)
            
            def _generate_datatable_columns_detail(df):
                columns = []
                for col in df.columns:
                    col_def = {"name": col, "id": col}
                    if 'Price' in col or 'Interest' in col:
                        col_def.update({'type': 'numeric', 'format': Format(precision=3, scheme=Scheme.fixed)})
                    elif '%' in col or 'YTM' in col or 'Coupon' in col or 'Duration' in col:
                        col_def.update({'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.fixed)})
                    columns.append(col_def)
                return columns

            table = dash_table.DataTable(
                id='bonds-individual-metrics-table',
                columns=_generate_datatable_columns_detail(df_table),
                data=df_table.to_dict('records'),
                style_header={'border': '0px', 'backgroundColor': 'transparent', 'fontWeight': '600', 'textTransform': 'uppercase', 'textAlign': 'right'},
                style_data={'border': '0px', 'backgroundColor': 'transparent'},
                style_cell={'textAlign': 'right', 'padding': '14px', 'border': '0px', 'borderBottom': '1px solid #f0f0f0'},
                style_header_conditional=[{'if': {'column_id': 'Ticker'}, 'textAlign': 'left'}],
                style_cell_conditional=[{'if': {'column_id': 'Ticker'}, 'textAlign': 'left', 'width': '10%', 'verticalAlign': 'middle'}],
                markdown_options={"html": True}
            )
            
            return [table], sort_options, sort_by

        return html.P(f"Table content not available for {active_tab}."), [], None


    # --- Modal Callbacks (Unchanged Logic, Corrected IDs) ---
    # ... (Toggle Definitions Modal และ Toggle Forecast Modal เดิม) ...
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
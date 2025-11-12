import dash
from dash import Dash, dcc, html, dash_table, callback_context
from dash.dependencies import Input, Output, State, ALL, MATCH
from dash.dash_table.Format import Format, Scheme
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, datetime
from app.constants import BOND_YIELD_MAP, BOND_BENCHMARK_MAP 
from app.web.pages.bonds import BOND_METRIC_DEFINITIONS

# Placeholder for database querying (MOCK DATA)
def fetch_daily_prices(tickers):
    """Mocks fetching daily price/yield data for given tickers."""
    
    # Mock Data (Replace this with real DB query)
    end_date = date.today()
    start_date = datetime(2020, 1, 1).date()
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    data = []
    for t in tickers:
        df_temp = pd.DataFrame({'date': date_range})
        # Generate mock data: Treasury Yields are usually low, ETFs are higher
        base_yield = 4 if t.startswith('^') else 100
        df_temp['close'] = base_yield + (np.random.randn(len(date_range)) * (0.5 if t.startswith('^') else 5))
        if t == '^TNX': df_temp['close'] += 1 
        if t == '^TYX': df_temp['close'] += 1.5 
        if t == '^GSPC': df_temp['close'] *= 20 # S&P 500 is much higher than yields
        df_temp['ticker'] = t
        data.append(df_temp)
    
    if not data:
        return pd.DataFrame(columns=['date', 'ticker', 'close'])
        
    df = pd.concat(data)
    df['previous_close'] = df.groupby('ticker')['close'].shift(1)
    return df

# --- Utility Functions (Adapted for Bonds) ---
def get_user_symbols(data):
    """Helper to extract active symbols (Yields and Benchmarks)."""
    tickers = data.get('tickers', [])
    indices = data.get('indices', [])
    all_symbols = list(set(tickers + indices))
    return all_symbols, tickers, indices

# --- Main Registration Function ---
def register_bonds_callbacks(app: Dash, BOND_METRIC_DEFINITIONS):

    # --- 8.1 Load Data Store (bonds-user-selections-store) ---
    @app.callback(
        Output('bonds-user-selections-store', 'data'),
        Input('bonds-add-yield-button', 'n_clicks'),
        Input('bonds-add-benchmark-button', 'n_clicks'),
        Input({'type': 'bonds-remove-ticker-btn', 'index': ALL}, 'n_clicks'),
        # [MODIFIED: Add URL input to ensure data loads on page navigation]
        Input('url', 'pathname'), 
        State('bonds-user-selections-store', 'data'),
        State('bonds-yield-select-dropdown', 'value'),
        State('bonds-benchmark-select-dropdown', 'value'),
        prevent_initial_call=False
    )
    def load_bonds_data_to_store(add_ticker_clicks, add_index_clicks, remove_clicks, url_pathname, current_data, new_tickers_list, new_indices_list):
        """Initializes the store and manages adding/removing yields and benchmarks."""
        
        # --- Default Selections ---
        # [FIXED: Ensure initial_data is correctly structured and is returned on initial load]
        initial_data = {
            'tickers': ['^TNX'],
            'indices': ['^GSPC'],
        }
        
        if not current_data:
            current_data = initial_data

        ctx = dash.callback_context
        if not ctx.triggered:
            # On initial load (no triggers), ensure default is set.
            return initial_data 

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        tickers = set(current_data.get('tickers', []))
        indices = set(current_data.get('indices', []))

        # Helper to ensure input is iterable
        def ensure_list(value):
            if value is None: return []
            return value if isinstance(value, list) else [value]

        # Add Yield
        if button_id == 'bonds-add-yield-button' and new_tickers_list:
            for t in ensure_list(new_tickers_list):
                tickers.add(t)
        
        # Add Benchmark
        elif button_id == 'bonds-add-benchmark-button' and new_indices_list:
            for i in ensure_list(new_indices_list):
                indices.add(i)

        # Remove Ticker/Index
        elif 'bonds-remove-ticker-btn' in button_id:
            import json
            triggered_id_str = ctx.triggered[0]['prop_id'].split('.')[0]
            try:
                removed_ticker = json.loads(triggered_id_str)['index']
                if removed_ticker in tickers:
                    tickers.remove(removed_ticker)
                elif removed_ticker in indices:
                    indices.remove(removed_ticker)
            except (json.JSONDecodeError, KeyError):
                pass 

        # Ensure defaults remain if all are removed
        if not tickers and not indices:
             current_data = initial_data
        else:
            current_data['tickers'] = sorted(list(tickers))
            current_data['indices'] = sorted(list(indices))

        return current_data


    # --- 8.2 Update Yield Dropdown Options ---
    @app.callback(
        Output('bonds-yield-select-dropdown', 'options'),
        Input('bonds-yield-type-dropdown', 'value'), 
    )
    def update_bond_options(selected_yield_type):
        """Populates the Yield selection dropdown using BOND_YIELD_MAP."""
        options = [{'label': name, 'value': ticker} for ticker, name in BOND_YIELD_MAP.items()]
        return options

    # --- 8.3 Update Benchmark Dropdown Options ---
    @app.callback(
        Output('bonds-benchmark-select-dropdown', 'options'),
        Input('bonds-yield-type-dropdown', 'value'), # Dummy Input to trigger on load
    )
    def update_bond_benchmark_options(dummy_input):
        """Populates the Benchmark selection dropdown using BOND_BENCHMARK_MAP."""
        options = [{'label': name, 'value': ticker} for ticker, name in BOND_BENCHMARK_MAP.items()]
        return options

    # --- Render Summary Display (FIXED: ใช้สไตล์เดียวกับ Stocks) ---
    @app.callback(
        Output('bonds-summary-display', 'children'),
        Output('bonds-benchmark-summary-display', 'children'),
        Input('bonds-user-selections-store', 'data')
    )
    def update_summary_display(data):
        """Renders selected Yields and Benchmarks as badges, mimicking the Stocks page style."""

        # [FIXED] Ensure data is not None
        if not data:
            data = {'tickers': [], 'indices': []}
            
        tickers = data.get('tickers', [])
        indices = data.get('indices', [])
        
        # Combine maps for lookup
        full_map = {**BOND_YIELD_MAP, **BOND_BENCHMARK_MAP}
        
        # --- YIELDS (STOCKS) ---
        if not tickers: 
             ticker_content = html.Div([html.Span("No Yields selected.", className="text-muted fst-italic")])
        else:
            ticker_content = [html.Label("Selected Yields:", className="text-muted small")] + [
                dbc.Badge([
                    # [MODIFIED]: ใช้ BOND_YIELD_MAP สำหรับแสดงชื่อเต็มของ Yields
                    BOND_YIELD_MAP.get(t, t),
                    html.I(className="bi bi-x-circle-fill ms-2", style={'cursor': 'pointer'}, id={'type': 'bonds-remove-ticker-btn', 'index': t})
                ], color="light", className="m-1 p-2 text-dark border") for t in tickers
            ]
        
        # --- BENCHMARKS (INDICES) ---
        if not indices:
            index_content = html.Div([html.Span("No Benchmarks selected.", className="text-muted fst-italic")])
        else:
            index_content = [html.Label("Selected Benchmarks:", className="text-muted small")] + [
                dbc.Badge([
                    full_map.get(t, t), # Use map for display name
                    html.I(className="bi bi-x-circle-fill ms-2", style={'cursor': 'pointer'}, id={'type': 'bonds-remove-ticker-btn', 'index': t})
                ], color="light", className="m-1 p-2 text-dark border")
                for t in indices
            ]
        
        return ticker_content, index_content


    # --- 8.4 Render Graph Content ---
    @app.callback(
        Output('bonds-analysis-pane-content', 'children'),
        Input('bonds-analysis-tabs', 'active_tab'),
        Input('bonds-user-selections-store', 'data'),
        prevent_initial_call=True
    )
    def render_bond_graph_content(active_tab, data):
        """Generates the appropriate graph based on the active tab and selected yields/benchmarks."""
        
        all_symbols, tickers, indices = get_user_symbols(data)
        
        if not all_symbols:
            return dbc.Alert("Please select at least one Treasury Yield for analysis.", color="info", className="mt-3 text-center")

        df_all = fetch_daily_prices(all_symbols)
        if df_all.empty:
            return dbc.Alert("No data available for selected instruments.", color="warning", className="mt-3 text-center")
        
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
                labels={'close': 'Yield (%) / Price', 'date': 'Date'}
            )
            fig.update_layout(
                yaxis_title='Yield (%) / Price', 
                legend_title='Instrument'
            )
            return dcc.Graph(figure=fig)

        # --- Tab: YIELD CURVE (tab-yield-curve) ---
        elif active_tab == "tab-yield-curve":
            
            # 1. Identify active Yield Tickers that represent different maturities
            treasury_yields = {
                '^FVX': 5, 
                '^TNX': 10, 
                '^TYX': 30
            }
            
            active_yield_tickers = [t for t in tickers if t in treasury_yields]
            if not active_yield_tickers:
                 return dbc.Alert("Please select US Treasury Yields (^FVX, ^TNX, ^TYX) to plot the Yield Curve.", color="info", className="mt-3 text-center")

            # 2. Get the latest date and filter data
            latest_date = df_all['date'].max()
            df_latest = df_all[df_all['date'] == latest_date]
            
            df_curve = df_latest[df_latest['ticker'].isin(active_yield_tickers)].copy()
            df_curve['Maturity_Years'] = df_curve['ticker'].map(treasury_yields)
            
            if df_curve.empty:
                return dbc.Alert(f"No latest data found for Treasury Yields on {latest_date.strftime('%Y-%m-%d')}.", color="warning", className="mt-3 text-center")

            # 3. Plotting the Yield Curve
            fig = px.line(
                df_curve,
                x='Maturity_Years',
                y='close',
                markers=True,
                line_shape='spline',
                title=f"US Treasury Yield Curve on {latest_date.strftime('%Y-%m-%d')}"
            )
            fig.update_layout(
                xaxis_title='Maturity (Years)',
                yaxis_title='Yield (%)',
                xaxis={'tickmode': 'linear'},
                hovermode="x unified"
            )
            return dcc.Graph(figure=fig)


        # --- Tab: YIELD SPREAD (tab-yield-spread) ---
        elif active_tab == "tab-yield-spread":
            if len(all_symbols) < 2:
                return dbc.Alert("Please select at least two instruments (Yields or Benchmarks) to calculate a Spread.", color="info", className="mt-3 text-center")
            
            # Use the first two selected for simplicity in this version.
            ticker_a, ticker_b = all_symbols[0], all_symbols[1]
            
            df_filtered = df_all[df_all['ticker'].isin([ticker_a, ticker_b])].pivot(
                index='date', columns='ticker', values='close'
            ).dropna().reset_index()
            
            if df_filtered.empty or ticker_a not in df_filtered.columns or ticker_b not in df_filtered.columns:
                return dbc.Alert(f"Insufficient time-series data to calculate spread between {full_map.get(ticker_a, ticker_a)} and {full_map.get(ticker_b, ticker_b)}.", color="warning", className="mt-3 text-center")

            # Calculate Spread
            df_filtered['Spread'] = df_filtered[ticker_a] - df_filtered[ticker_b]
            
            spread_name = f"{full_map.get(ticker_a, ticker_a)} minus {full_map.get(ticker_b, ticker_b)}"

            # Plotting the Spread
            fig = px.line(
                df_filtered,
                x='date',
                y='Spread',
                title=f"Historical Yield/Price Spread: {spread_name}",
                labels={'Spread': 'Spread Value', 'date': 'Date'}
            )
            
            fig.update_layout(yaxis_title='Spread Value', hovermode="x unified")
            return dcc.Graph(figure=fig)

        return html.P(f"Content for {active_tab} is not implemented yet.")


    # --- 8.5 Render Table Content (FIXED: ใช้ Dash DataTable Style ที่เหมือน Stocks) ---
    @app.callback(
        Output('bonds-table-pane-content', 'children'),
        Input('bonds-table-tabs', 'active_tab'),
        Input('bonds-user-selections-store', 'data'),
        Input('bonds-sort-by-dropdown', 'value'),
        prevent_initial_call=True
    )
    def render_bond_table_content(active_tab, data, sort_by):
        """Generates the RATES SUMMARY table with matching styles."""
        
        all_symbols, tickers, indices = get_user_symbols(data)
        
        if not all_symbols:
            return dbc.Alert("Please select at least one instrument for the summary table.", color="info", className="mt-3 text-center")
        
        if active_tab != "tab-rates-summary":
            return html.P(f"Table content not available for {active_tab}.")

        # --- Combine Maps for Display Names ---
        full_map = {**BOND_YIELD_MAP, **BOND_BENCHMARK_MAP}

        # 1. Fetch Daily Prices (Mocked)
        df_daily = fetch_daily_prices(all_symbols)
        if df_daily.empty:
            return dbc.Alert("No data available to generate the summary table.", color="warning", className="mt-3 text-center")
        
        # 2. Get Latest and Previous day data
        latest_date = df_daily['date'].max()
        df_latest = df_daily[df_daily['date'] == latest_date].copy()
        
        # 3. Create Summary Table and Calculate Change
        # Note: df_latest needs to be renamed here
        df_summary = df_latest[['ticker', 'close', 'previous_close']].copy().rename(columns={'close': 'latest_yield'})
        
        # Calculate 1-Day Change in Basis Points (bps)
        df_summary['change_bps'] = (df_summary['latest_yield'] - df_summary['previous_close']) * 10000
        
        # Map Ticker to Name
        df_summary['Maturity / Benchmark'] = df_summary['ticker'].map(full_map).fillna(df_summary['ticker'])
        
        df_table = df_summary[[
            'Maturity / Benchmark', 
            'latest_yield', 
            'change_bps'
        ]].copy()
        
        # Rename columns for display
        df_table.columns = [
            'Maturity / Benchmark', 
            'Latest Yield (%)', 
            '1-Day Change (bps)'
        ]

        # 4. Sorting
        sort_column_map = {'latest_yield': 'Latest Yield (%)', 'change_bps': '1-Day Change (bps)'}
        sort_col_display = sort_column_map.get(sort_by, 'Latest Yield (%)')
        
        # Assuming higher yield/change is better by default for this simple sort
        df_table = df_table.sort_values(by=sort_col_display, ascending=False)
            
        # 5. Generate Dash DataTable (FIXED: ใช้ Style ที่เหมือน Stocks)
        def _generate_datatable_columns(df):
            columns = []
            for col in df.columns:
                col_def = {"name": col, "id": col}
                if 'Yield' in col or 'Change' in col:
                    # Format Yield/Change columns
                    col_def.update({
                        'type': 'numeric',
                        'format': Format(precision=2, scheme=Scheme.fixed)
                    })
                columns.append(col_def)
            return columns
            
        table = dash_table.DataTable(
            id='bonds-rates-summary-table',
            columns=_generate_datatable_columns(df_table),
            data=df_table.to_dict('records'),
            row_selectable=False, 
            cell_selectable=False,
            # [FIX] ใช้ Styles ที่เหมือน stocks ใน callbacks.py
            style_header={'border': '0px', 'backgroundColor': 'transparent', 'fontWeight': '600', 'textTransform': 'uppercase', 'textAlign': 'right'},
            style_data={'border': '0px', 'backgroundColor': 'transparent'},
            style_cell={'textAlign': 'right', 'padding': '14px', 'verticalAlign': 'middle'},
            style_header_conditional=[{'if': {'column_id': 'Maturity / Benchmark'}, 'textAlign': 'left'}],
            style_cell_conditional=[{'if': {'column_id': 'Maturity / Benchmark'}, 'textAlign': 'left', 'width': '40%'}],
            markdown_options={"html": True}
        )
        
        return [table]


    # --- Modal Callbacks (Unchanged Logic, Corrected IDs) ---

    # Callback to show/hide Definitions Modal
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
        if not ctx.triggered:
            return is_open, html.P("Select a tab for definitions.")

        prop_id = ctx.triggered[0]['prop_id']

        if 'bonds-open-definitions-modal-btn' in prop_id:
            active_tab = graph_tab if 'graphs' in prop_id else table_tab
            content = render_definitions(active_tab)
            return True, content
        elif prop_id == "bonds-close-definitions-modal.n_clicks":
            return False, html.P("Select a tab for definitions.")
        elif 'active_tab' in prop_id:
            if is_open:
                content = render_definitions(graph_tab) if 'analysis-tabs' in prop_id else render_definitions(table_tab)
                return is_open, content
        
        return is_open, html.P("Select a tab for definitions.")
    
    def render_definitions(tab_id):
        """Generates the content for the definitions modal."""
        definition = BOND_METRIC_DEFINITIONS.get(tab_id, {})
        
        if not definition:
            return html.P("Definition not found for this tab.")

        content = [
            html.H4(definition.get('title')),
            html.P(definition.get('description')),
            html.Ul([
                html.Li(html.B(f"{m['metric']}: "), html.Span(m['definition']))
                for m in definition.get('metrics', [])
            ])
        ]
        return content


    # Callback to show/hide Forecast/DCF Modal (Disabled/Simplified)
    @app.callback(
        Output("bonds-forecast-assumptions-modal", "is_open"),
        Input("bonds-open-forecast-modal-btn", "n_clicks"),
        Input("bonds-close-forecast-assumptions-modal", "n_clicks"),
        State("bonds-forecast-assumptions-modal", "is_open"),
        prevent_initial_call=True
    )
    def toggle_forecast_modal(n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open

    # Visibility Callbacks (Permanent Fix)
    @app.callback(Output('bonds-open-dcf-modal-btn', 'style'), Input('bonds-analysis-tabs', 'active_tab'))
    def toggle_dcf_gear_button_visibility(active_tab): 
        # Button is only visible on tab-dcf for Stocks, but here we hide it permanently 
        # (as it is not used for bonds).
        return {'display': 'none'} 

    @app.callback(Output('bonds-open-forecast-modal-btn', 'style'), Input('bonds-table-tabs', 'active_tab'))
    def toggle_forecast_gear_button_visibility(active_tab): 
        # Button is only visible on tab-forecast for Stocks, but here we hide it permanently 
        # (as it is not used for bonds).
        return {'display': 'none'}
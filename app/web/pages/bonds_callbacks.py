# app/web/pages/bonds_callbacks.py

from dash import Dash
from dash.dependencies import Input, Output, State, ALL, MATCH
import dash_html_components as html
import dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime
from app.data.database import FactDailyPrices  # Assumption: Database access object
from app.constants import BOND_YIELD_MAP, BOND_BENCHMARK_MAP 
from app.web.pages.bonds import BOND_METRIC_DEFINITIONS # Import the new definitions

# --- 7.1 Changed function name ---
def register_bonds_callbacks(app: Dash, BOND_METRIC_DEFINITIONS):

    # Placeholder for database querying (replace with actual database integration)
    def fetch_daily_prices(tickers):
        """Mocks fetching daily price/yield data for given tickers."""
        # This function must be replaced by actual DB query returning a DataFrame:
        # Columns: 'date', 'ticker', 'close'
        
        # Mock Data (Replace this with real DB query)
        end_date = date.today()
        start_date = datetime(2020, 1, 1).date()
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        data = []
        for t in tickers:
            df_temp = pd.DataFrame({'date': date_range})
            # Generate mock data: Treasury Yields are usually low, ETFs are higher
            base_yield = 4 if t.startswith('^') else 100
            df_temp['close'] = base_yield + (pd.np.random.randn(len(date_range)) * (0.5 if t.startswith('^') else 5))
            if t == '^TNX': df_temp['close'] += 1 
            if t == '^TYX': df_temp['close'] += 1.5 
            df_temp['ticker'] = t
            data.append(df_temp)
        
        if not data:
            return pd.DataFrame(columns=['date', 'ticker', 'close'])
            
        df = pd.concat(data)
        df['previous_close'] = df.groupby('ticker')['close'].shift(1)
        return df

    # --- Utility Functions (Adapted for Bonds) ---
    def get_user_tickers(data):
        """Helper to extract active tickers and indices/benchmarks."""
        tickers = data.get('tickers', [])
        indices = data.get('indices', [])
        all_symbols = list(set(tickers + indices))
        return all_symbols, tickers, indices

    # --- 8.1 Load Data Store (bonds-user-selections-store) ---
    @app.callback(
        Output('bonds-user-selections-store', 'data'),
        Input('bonds-add-yield-button', 'n_clicks'),
        Input('bonds-add-benchmark-button', 'n_clicks'),
        Input({'type': 'bonds-remove-ticker-btn', 'index': ALL}, 'n_clicks'),
        State('bonds-user-selections-store', 'data'),
        State('bonds-yield-select-dropdown', 'value'),
        State('bonds-benchmark-select-dropdown', 'value'),
        prevent_initial_call=False
    )
    def load_bonds_data_to_store(add_ticker_clicks, add_index_clicks, remove_clicks, current_data, new_ticker, new_index):
        """Initializes the store and manages adding/removing yields and benchmarks."""
        
        # --- Default Selections (8.1 Logic Change) ---
        if not current_data or not current_data.get('tickers'):
            current_data = {
                'tickers': ['^TNX'],  # Default 10-Year Treasury Yield
                'indices': ['LQD', '^GSPC'], # Default Benchmark/ETF
                'peers': [] # No peers by default
            }

        ctx = dash.callback_context
        if not ctx.triggered:
            # Initial load or no action
            return current_data

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        tickers = set(current_data.get('tickers', []))
        indices = set(current_data.get('indices', []))

        # Add Yield
        if button_id == 'bonds-add-yield-button' and new_ticker:
            tickers.add(new_ticker)
        
        # Add Benchmark
        elif button_id == 'bonds-add-benchmark-button' and new_index:
            indices.add(new_index)

        # Remove Ticker/Index
        elif 'bonds-remove-ticker-btn' in button_id:
            # The 'index' in the button ID is the ticker to remove
            removed_ticker = dash.callback_context.triggered[0]['prop_id'].split('"index":"')[1].split('"}')[0]
            if removed_ticker in tickers:
                tickers.remove(removed_ticker)
            elif removed_ticker in indices:
                indices.remove(removed_ticker)

        # Ensure defaults remain if all are removed (optional, but good practice)
        if not tickers and not indices:
             current_data = {'tickers': ['^TNX'], 'indices': ['^GSPC'], 'peers': []}
        else:
            current_data['tickers'] = sorted(list(tickers))
            current_data['indices'] = sorted(list(indices))

        return current_data


    # --- 8.2 Update Yield Dropdown Options ---
    @app.callback(
        Output('bonds-yield-select-dropdown', 'options'),
        Input('bonds-yield-type-dropdown', 'value'), # Input is present but ignored (sector not needed for fixed options)
    )
    def update_bond_options(selected_yield_type):
        """Populates the Yield selection dropdown using BOND_YIELD_MAP."""
        # Logic Change (8.2): Directly use the constant map instead of DB query
        options = [{'label': name, 'value': ticker} for ticker, name in BOND_YIELD_MAP.items()]
        return options

    # --- 8.3 Update Benchmark Dropdown Options ---
    @app.callback(
        Output('bonds-benchmark-select-dropdown', 'options'),
        Input('bonds-yield-select-dropdown', 'value'), # Input is present but ignored (dummy for trigger)
    )
    def update_bond_benchmark_options(dummy_input):
        """Populates the Benchmark selection dropdown using BOND_BENCHMARK_MAP."""
        # Logic Change (8.3): Directly use the constant map instead of DB query
        options = [{'label': name, 'value': ticker} for ticker, name in BOND_BENCHMARK_MAP.items()]
        return options

    # --- Render Summary Display (Same Logic, New IDs) ---
    @app.callback(
        Output('bonds-summary-display', 'children'),
        Output('bonds-benchmark-summary-display', 'children'),
        Input('bonds-user-selections-store', 'data')
    )
    def update_summary_display(data):
        # Uses BOND_YIELD_MAP and BOND_BENCHMARK_MAP for name resolution

        def create_ticker_divs(tickers, mapping):
            if not tickers:
                return "No items selected."
            
            items = []
            for t in tickers:
                display_name = mapping.get(t) or t 
                items.append(
                    html.Span(
                        [
                            html.Span(display_name, style={'marginRight': '5px'}),
                            html.Button(
                                'x', 
                                id={'type': 'bonds-remove-ticker-btn', 'index': t}, 
                                n_clicks=0, 
                                className="close-button" # CSS needed for this button
                            )
                        ],
                        className="badge bg-primary text-white me-2 mb-1 p-2 rounded-pill",
                        style={'display': 'inline-block', 'fontSize': '0.8rem'}
                    )
                )
            return items

        tickers = data.get('tickers', [])
        indices = data.get('indices', [])
        
        # Combine maps for lookup
        full_map = {**BOND_YIELD_MAP, **BOND_BENCHMARK_MAP}
        
        ticker_divs = create_ticker_divs(tickers, full_map)
        index_divs = create_ticker_divs(indices, full_map)
        
        return ticker_divs, index_divs

    # --- 8.4 Render Graph Content ---
    @app.callback(
        Output('bonds-analysis-pane-content', 'children'),
        Input('bonds-analysis-tabs', 'active_tab'),
        Input('bonds-user-selections-store', 'data'),
        prevent_initial_call=True
    )
    def render_bond_graph_content(active_tab, data):
        """Generates the appropriate graph based on the active tab and selected yields/benchmarks."""
        
        all_symbols, tickers, indices = get_user_tickers(data)
        
        if not all_symbols:
            return html.P("Please select at least one Treasury Yield for analysis.")

        df_all = fetch_daily_prices(all_symbols)
        if df_all.empty:
            return html.P("No data available for selected instruments.")
        
        # --- Combine Maps for Display Names ---
        full_map = {**BOND_YIELD_MAP, **BOND_BENCHMARK_MAP}
        df_all['display_name'] = df_all['ticker'].map(full_map).fillna(df_all['ticker'])

        # --- Tab: HISTORICAL YIELDS (tab-yield-history) ---
        if active_tab == "tab-yield-history":
            # Logic: Simple time series plot (similar to Stocks performance tab)
            fig = px.line(
                df_all, 
                x='date', 
                y='close', 
                color='display_name',
                title="Historical Yields and Benchmarks",
                labels={'close': 'Yield / Close Price', 'date': 'Date'}
            )
            fig.update_layout(
                yaxis_title='Yield (%) / Price', 
                legend_title='Instrument'
            )
            return dcc.Graph(figure=fig)

        # --- Tab: YIELD CURVE (tab-yield-curve) ---
        elif active_tab == "tab-yield-curve":
            # --- Logic Change (8.4): Plot Yield Curve ---
            
            # 1. Identify active Yield Tickers that represent different maturities
            treasury_yields = {
                '^FVX': 5, 
                '^TNX': 10, 
                '^TYX': 30
            }
            
            active_yield_tickers = [t for t in tickers if t in treasury_yields]
            if not active_yield_tickers:
                 return html.P("Please select US Treasury Yields (^FVX, ^TNX, ^TYX) to plot the Yield Curve.")

            # 2. Get the latest date and filter data
            latest_date = df_all['date'].max()
            df_latest = df_all[df_all['date'] == latest_date]
            
            df_curve = df_latest[df_latest['ticker'].isin(active_yield_tickers)].copy()
            df_curve['Maturity_Years'] = df_curve['ticker'].map(treasury_yields)
            
            if df_curve.empty:
                return html.P(f"No latest data found for Treasury Yields on {latest_date}.")

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
            # --- Logic Change (8.4): Plot Yield Spread (requires two selected instruments) ---

            if len(all_symbols) < 2:
                return html.P("Please select at least two instruments (Yields or Benchmarks) to calculate a Spread.")
            
            # Use the first two selected for simplicity in this version.
            # In a real app, dedicated dropdowns would select the two for spread.
            ticker_a, ticker_b = all_symbols[0], all_symbols[1]
            
            df_filtered = df_all[df_all['ticker'].isin([ticker_a, ticker_b])].pivot(
                index='date', columns='ticker', values='close'
            ).dropna().reset_index()
            
            if df_filtered.empty or ticker_a not in df_filtered.columns or ticker_b not in df_filtered.columns:
                return html.P(f"Insufficient time-series data to calculate spread between {full_map[ticker_a]} and {full_map[ticker_b]}.")

            # Calculate Spread
            df_filtered['Spread'] = df_filtered[ticker_a] - df_filtered[ticker_b]
            
            spread_name = f"{full_map[ticker_a]} minus {full_map[ticker_b]}"

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


    # --- 8.5 Render Table Content ---
    @app.callback(
        Output('bonds-table-pane-content', 'children'),
        Input('bonds-table-tabs', 'active_tab'), # Should only be 'tab-rates-summary'
        Input('bonds-user-selections-store', 'data'),
        Input('bonds-sort-by-dropdown', 'value'),
        prevent_initial_call=True
    )
    def render_bond_table_content(active_tab, data, sort_by):
        """Generates the RATES SUMMARY table."""
        
        all_symbols, tickers, indices = get_user_tickers(data)
        
        if not all_symbols:
            return html.P("Please select at least one instrument for the summary table.")
        
        if active_tab != "tab-rates-summary":
            return html.P(f"Table content not available for {active_tab}.")

        # --- Combine Maps for Display Names ---
        full_map = {**BOND_YIELD_MAP, **BOND_BENCHMARK_MAP}

        # 1. Fetch Daily Prices
        df_daily = fetch_daily_prices(all_symbols)
        if df_daily.empty:
            return html.P("No data available to generate the summary table.")
        
        # 2. Get Latest and Previous day data
        # Note: df_daily already has 'previous_close' from fetch_daily_prices
        latest_date = df_daily['date'].max()
        df_latest = df_daily[df_daily['date'] == latest_date].copy()
        
        # 3. Create Summary Table and Calculate Change (8.5 Logic Change)
        df_summary = df_latest[['ticker', 'close', 'previous_close']].copy()
        df_summary = df_summary.rename(columns={'close': 'latest_yield'})
        
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
        if sort_by == 'latest_yield':
            df_table = df_table.sort_values(by='Latest Yield (%)', ascending=False)
        elif sort_by == 'change_bps':
            df_table = df_table.sort_values(by='1-Day Change (bps)', ascending=False)
            
        # 5. Generate Dash DataTable
        def _generate_datatable_columns(df):
            """Generates columns definition for dash_table."""
            columns = []
            for col in df.columns:
                if 'Yield' in col or 'Change' in col:
                    # Format Yield/Change columns
                    columns.append({
                        'id': col, 
                        'name': col, 
                        'type': 'numeric',
                        'format': dash_table.Format.Format(precision=2, scheme=dash_table.Format.Scheme.fixed)
                    })
                else:
                    columns.append({'id': col, 'name': col})
            return columns
            
        table = dash_table.DataTable(
            id='bonds-rates-summary-table',
            columns=_generate_datatable_columns(df_table),
            data=df_table.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
        )
        
        return [
            html.P(f"Data as of: {latest_date.strftime('%Y-%m-%d')}"),
            table
        ]


    # --- Modal Callbacks (Same Logic, New IDs) ---

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

        # Determine the active tab based on which button or tab was clicked
        if prop_id in ['bonds-open-definitions-modal-btn-graphs.n_clicks', 'bonds-open-definitions-modal-btn-tables.n_clicks']:
            # Open the modal and set content based on the *currently* active tab
            active_tab = graph_tab if 'graphs' in prop_id else table_tab
            content = render_definitions(active_tab)
            return True, content
        elif 'n_clicks' in prop_id:
            # If any n_clicks button (except close) is triggered, toggle the modal
            return not is_open, html.P("Select a tab for definitions.")
        elif 'active_tab' in prop_id:
            # Update content if the tab changes while the modal is open
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
        Input("bonds-open-dcf-modal-btn", "n_clicks"),
        Input("bonds-close-forecast-assumptions-modal", "n_clicks"),
        State("bonds-forecast-assumptions-modal", "is_open"),
        prevent_initial_call=True
    )
    def toggle_forecast_modal(n1, n2, is_open):
        # Always return False if triggered by n1 (Open button is disabled in layout)
        if n1:
            return not is_open
        if n2:
            return not is_open
        return is_open
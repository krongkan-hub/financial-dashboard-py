# callbacks.py (Final Version - With Default Tickers on Load)

import dash
from dash import dcc, html, callback_context, dash_table
from dash.dependencies import Input, Output, State, ALL
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import yfinance as yf
import numpy as np
import json
import logging
from flask_login import current_user

# Import core app objects from app.py
from app import app, db, server, User, UserSelection, UserAssumptions

# Import layout components from layout.py
from layout import build_layout, create_navbar

# Import helpers and other page layouts
from data_handler import (
    calculate_drawdown, get_competitor_data, get_scatter_data,
    calculate_exit_multiple_valuation,
    calculate_monte_carlo_dcf
)
from pages import deep_dive
from constants import (
    INDEX_TICKER_TO_NAME, SECTOR_TO_INDEX_MAPPING, COLOR_DISCRETE_MAP, SECTORS,
    TOP_5_DEFAULT_TICKERS, ALL_TICKERS_SORTED_BY_MC  # <-- [MODIFIED] Import Tickers เริ่มต้น
)
from auth import create_register_layout

# ==================================================================
# Table Helpers & Config
# ==================================================================
TABS_CONFIG = {
    "tab-performance": { "tab_name": "Performance" },
    "tab-drawdown": { "tab_name": "Drawdown" },
    "tab-scatter": { "tab_name": "Valuation vs. Quality" },
    "tab-dcf": { "tab_name": "Margin of Safety (DCF)" },
    "tab-valuation": { "columns": ["Ticker", "Market Cap", "Company Size", "Price", "P/E", "P/B", "EV/EBITDA"], "higher_is_better": {"P/E": False, "P/B": False, "EV/EBITDA": False}, "tab_name": "Valuation" },
    "tab-growth": { "columns": ["Ticker", "Revenue Growth (YoY)", "Revenue CAGR (3Y)", "Net Income Growth (YoY)"], "higher_is_better": {k: True for k in ["Revenue Growth (YoY)", "Revenue CAGR (3Y)", "Net Income Growth (YoY)"]}, "tab_name": "Growth" },
    "tab-fundamentals": { "columns": ["Ticker", "Operating Margin", "ROE", "D/E Ratio", "Cash Conversion"], "higher_is_better": {"Operating Margin": True, "ROE": True, "D/E Ratio": False, "Cash Conversion": True}, "tab_name": "Fundamentals" },
    "tab-forecast": {
        "columns": ["Ticker", "Target Price", "Target Upside", "IRR %", "Volatility Level", "Valuation Model", "Stock Profile"],
        "higher_is_better": {"Target Upside": True, "IRR %": True},
        "tab_name": "Target"
    }
}

def _format_market_cap(n):
    if pd.isna(n): return '-'
    return f'${n/1e9:,.2f}B' if abs(n) >= 1e9 else f'${n/1e6:,.2f}M'

def _prepare_display_dataframe(df_raw):
    df_display = df_raw.copy()
    def create_ticker_cell(row):
        logo_url, ticker = row.get('logo_url'), row['Ticker']
        logo_html = f'<img src="{logo_url}" style="height: 22px; width: 22px; margin-right: 8px; border-radius: 4px;" onerror="this.style.display=\'none\'">' if logo_url else ''
        return f'''<a href="/deepdive/{ticker}" style="text-decoration: none; color: inherit; font-weight: 600; display: flex; align-items: center;">{logo_html}<span>{ticker}</span></a>'''

    df_display['Ticker'] = df_display.apply(create_ticker_cell, axis=1)
    if "Market Cap" in df_display.columns:
        df_display["Market Cap"] = df_raw["Market Cap"].apply(_format_market_cap)
    scoring_cols = ['Company Size', 'Volatility Level', 'Valuation Model', 'Stock Profile']
    for col in scoring_cols:
        if col in df_display.columns:
            df_display[col] = df_raw[col].apply(lambda x: x if pd.notna(x) else "")
    return df_display

def _generate_datatable_columns(tab_config):
    columns = []
    for col in tab_config["columns"]:
        col_def = {"name": col.replace(" (3Y)", ""), "id": col}
        if col == 'Ticker': col_def.update({"type": "text", "presentation": "markdown"})
        elif col in ['Company Size', 'Volatility Level', 'Valuation Model', 'Stock Profile', 'Market Cap']: col_def.update({"type": "text"})
        elif col in ['Target Upside', 'IRR %']: col_def.update({"type": "numeric", "format": Format(precision=2, scheme=Scheme.percentage)})
        elif col == 'Target Price': col_def.update({"type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)})
        elif any(kw in col for kw in ['Growth', 'Margin', 'ROE', 'Conversion']): col_def.update({"type": "numeric", "format": Format(precision=2, scheme=Scheme.percentage)})
        else: col_def.update({"type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)})
        columns.append(col_def)
    return columns

def _generate_datatable_style_conditionals(tab_config):
    style_data_conditional, style_cell_conditional = [], []
    displayed_columns = tab_config["columns"]
    other_cols = [c for c in displayed_columns if c != 'Ticker']
    ticker_width_percent = 15
    style_cell_conditional.append({'if': {'column_id': 'Ticker'}, 'width': f'{ticker_width_percent}%'})
    if other_cols:
        other_col_width_percent = (100 - ticker_width_percent) / len(other_cols)
        for col in other_cols:
            style_cell_conditional.append({'if': {'column_id': col}, 'width': f'{other_col_width_percent}%'})
    style_cell_conditional.append({'if': {'column_id': 'Ticker'}, 'textAlign': 'left'})
    return style_data_conditional, style_cell_conditional

def apply_custom_scoring(df):
    if df.empty: return df
    bins = [0, 1e10, 10e10, 100e10, float('inf')]
    labels = ["Small Cap", "Mid Cap", "Large Cap", "Mega Cap"]
    df['Company Size'] = pd.cut(df['Market Cap'], bins=bins, labels=labels, right=False)
    conditions_volatility = [df['Beta'].isnull(), df['Beta'] < 0.5, df['Beta'] <= 2, df['Beta'] > 2]
    choices_volatility = ["N/A", "Core", "Growth", "Hyper Growth"]
    df['Volatility Level'] = np.select(conditions_volatility, choices_volatility, default='N/A')
    conditions_valuation = [df['EV/EBITDA'].isnull(), df['EV/EBITDA'] < 10, df['EV/EBITDA'] <= 25, df['EV/EBITDA'] > 25]
    choices_valuation = ["N/A", "Cheap", "Fair Value", "Expensive"]
    df['Valuation Model'] = np.select(conditions_valuation, choices_valuation, default='N/A')
    def categorize_stock(row):
        vol, val = row['Volatility Level'], row['Valuation Model']
        if vol == 'Hyper Growth' and val == 'Fair Value': return "Promising Growth"
        if vol == 'Core' and val == 'Cheap': return "Hidden Value Gem"
        if vol == 'Hyper Growth' and val == 'Expensive': return "High-Flyer"
        if vol == 'Core' and val == 'Fair Value': return "Stable Compounder"
        if vol == 'Growth' and val == 'Fair Value': return "Core Holding"
        if vol == 'Growth' and val == 'Cheap': return "Value Opportunity"
        return "Needs Review"
    df['Stock Profile'] = df.apply(categorize_stock, axis=1)
    return df

# ==================================================================
# Callback Registration
# ==================================================================
def register_callbacks(app, METRIC_DEFINITIONS):

    @app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
    def display_page(pathname):
        if pathname.startswith('/deepdive/'):
            ticker = pathname.split('/')[-1].upper()
            return html.Div([create_navbar(), deep_dive.create_deep_dive_layout(ticker)])
        elif pathname == '/register':
            return create_register_layout()
        return build_layout()

    @app.callback(Output('navbar-container', 'children'), Input('url', 'pathname'))
    def update_navbar(pathname):
        if pathname != '/register' and not pathname.startswith('/deepdive/'):
             return create_navbar()
        return None

    # --- [MODIFIED CALLBACK] ---
    # Loads user data OR default tickers if user is new/logged out
    @app.callback(
        Output('user-selections-store', 'data'),
        Output('forecast-assumptions-store', 'data'),
        Output('dcf-assumptions-store', 'data'),
        Input('url', 'pathname')
    )
    def load_user_data_to_store(pathname):
        # Define defaults first
        default_dcf_data = {
            'simulations': 10000, 'growth_min': 3, 'growth_mode': 5, 'growth_max': 8,
            'perpetual_min': 1.5, 'perpetual_mode': 2.5, 'perpetual_max': 3.0,
            'wacc_min': 7.0, 'wacc_mode': 8.0, 'wacc_max': 10.0
        }
        default_forecast_data = {'years': 5, 'growth': 10, 'pe': 20}
        
        # [NEW] Default selections for non-users or new users
        default_selections_data = {
            'tickers': TOP_5_DEFAULT_TICKERS, 
            'indices': ['^GSPC']
        }
        
        if pathname == '/' and current_user.is_authenticated:
            with server.app_context():
                # Load user's saved data
                stocks = UserSelection.query.filter_by(user_id=current_user.id, symbol_type='stock').all()
                indices = UserSelection.query.filter_by(user_id=current_user.id, symbol_type='index').all()
                
                # [NEW] Check if the logged-in user has any saved data
                if not stocks and not indices:
                    # If they are logged in but have 0 saved items, give them the default
                    selections_data = default_selections_data
                else:
                    # Otherwise, load their saved data
                    selections_data = {'tickers': [s.symbol for s in stocks], 'indices': [i.symbol for i in indices]}

                # Load saved assumptions (or use default)
                assumptions = UserAssumptions.query.filter_by(user_id=current_user.id).first()
                forecast_data = {'years': assumptions.forecast_years, 'growth': assumptions.eps_growth, 'pe': assumptions.terminal_pe} if assumptions else default_forecast_data
                
                return selections_data, forecast_data, default_dcf_data
        
        # [NEW] If user is not authenticated, return the default set
        return default_selections_data, default_forecast_data, default_dcf_data
    # --- [END OF MODIFIED CALLBACK] ---

    def save_selections_to_db(user_id, symbols, symbol_type):
        with server.app_context():
            UserSelection.query.filter_by(user_id=user_id, symbol_type=symbol_type).delete(synchronize_session=False)
            for symbol in symbols:
                db.session.add(UserSelection(user_id=user_id, symbol_type=symbol_type, symbol=symbol))
            db.session.commit()

    @app.callback(Output('user-selections-store', 'data', allow_duplicate=True), Input('add-ticker-button', 'n_clicks'), [State('ticker-select-dropdown', 'value'), State('user-selections-store', 'data')], prevent_initial_call=True)
    def add_ticker_to_store(n_clicks, selected_tickers, store_data):
        store_data = store_data or {'tickers': [], 'indices': []}
        if selected_tickers:
            for ticker in selected_tickers:
                if ticker not in store_data['tickers']:
                    store_data['tickers'].append(ticker)
            if current_user.is_authenticated:
                save_selections_to_db(current_user.id, store_data['tickers'], 'stock')
        return store_data

    @app.callback(Output('user-selections-store', 'data', allow_duplicate=True), Input({'type': 'remove-stock', 'index': ALL}, 'n_clicks'), State('user-selections-store', 'data'), prevent_initial_call=True)
    def remove_ticker_from_store(n_clicks, store_data):
        if not any(n_clicks): return dash.no_update
        store_data = store_data or {'tickers': [], 'indices': []}
        triggered_id = json.loads(callback_context.triggered[0]['prop_id'].split('.')[0])['index']
        if triggered_id in store_data['tickers']:
            store_data['tickers'].remove(triggered_id)
            if current_user.is_authenticated: save_selections_to_db(current_user.id, store_data['tickers'], 'stock')
        return store_data

    @app.callback(Output('user-selections-store', 'data', allow_duplicate=True), Input('add-index-button', 'n_clicks'), [State('index-select-dropdown', 'value'), State('user-selections-store', 'data')], prevent_initial_call=True)
    def add_index_to_store(n_clicks, selected_indices, store_data):
        store_data = store_data or {'tickers': [], 'indices': []}
        if selected_indices:
            for index in selected_indices:
                if index not in store_data['indices']:
                    store_data['indices'].append(index)
            if current_user.is_authenticated:
                save_selections_to_db(current_user.id, store_data['indices'], 'index')
        return store_data

    @app.callback(Output('user-selections-store', 'data', allow_duplicate=True), Input({'type': 'remove-index', 'index': ALL}, 'n_clicks'), State('user-selections-store', 'data'), prevent_initial_call=True)
    def remove_index_from_store(n_clicks, store_data):
        if not any(n_clicks): return dash.no_update
        store_data = store_data or {'tickers': [], 'indices': []}
        triggered_id = json.loads(callback_context.triggered[0]['prop_id'].split('.')[0])['index']
        if triggered_id in store_data['indices']:
            store_data['indices'].remove(triggered_id)
            if current_user.is_authenticated: save_selections_to_db(current_user.id, store_data['indices'], 'index')
        return store_data

    @app.callback(Output('ticker-summary-display', 'children'), Input('user-selections-store', 'data'))
    def update_ticker_summary_display(store_data):
        tickers = store_data.get('tickers', []) if store_data else []
        if not tickers:
            return html.Div([html.Span("No stocks selected.", className="text-muted fst-italic")])
        return [html.Label("Selected Stocks:", className="text-muted small")] + [dbc.Badge([t, html.I(className="bi bi-x-circle-fill ms-2", style={'cursor': 'pointer'}, id={'type': 'remove-stock', 'index': t})], color="light", className="m-1 p-2 text-dark border") for t in tickers]

    @app.callback(Output('index-summary-display', 'children'), Input('user-selections-store', 'data'))
    def update_index_summary_display(store_data):
        indices = store_data.get('indices', []) if store_data else []
        if not indices:
            return html.Span("No indices selected.", className="text-muted fst-italic")
        return [html.Label("Selected Indices:", className="text-muted small")] + [dbc.Badge([INDEX_TICKER_TO_NAME.get(i, i), html.I(className="bi bi-x-circle-fill ms-2", style={'cursor': 'pointer'}, id={'type': 'remove-index', 'index': i})], color="light", className="m-1 p-2 text-dark border") for i in indices]

    # --- [REVERTED CALLBACK] ---
    # This callback now uses the sorted lists from constants.py
    # (No expensive Market Cap call needed here anymore)
    @app.callback(Output('ticker-select-dropdown', 'options'), [Input('sector-dropdown', 'value'), Input('user-selections-store', 'data')])
    def update_ticker_options(selected_sector, store_data):
        if not selected_sector: return []
        selected_tickers = store_data.get('tickers', []) if store_data else []
        
        if selected_sector == 'All':
            # --- [FIX] ใช้ลิสต์ที่เรียงตาม Market Cap จาก constants.py ---
            tickers_to_display = ALL_TICKERS_SORTED_BY_MC
        else:
            # This list is now pre-sorted by Market Cap from constants.py
            tickers_to_display = SECTORS.get(selected_sector, []) 
            
        return [{'label': t, 'value': t} for t in tickers_to_display if t not in selected_tickers]
    # --- [END OF REVERTED CALLBACK] ---


    @app.callback(Output('index-select-dropdown', 'options'), Input('user-selections-store', 'data'))
    def update_index_options(store_data):
        if not store_data or not store_data.get('tickers'): return []
        selected_tickers, selected_indices = store_data.get('tickers', []), store_data.get('indices', [])
        active_sectors = {s for t in selected_tickers for s, stocks in SECTORS.items() if t in stocks}
        relevant_indices = {idx for sec in active_sectors for idx in SECTOR_TO_INDEX_MAPPING.get(sec, [])} | {'^GSPC', '^NDX'}
        return [{'label': INDEX_TICKER_TO_NAME.get(i, i), 'value': i} for i in sorted(list(relevant_indices)) if i not in selected_indices]
        
    @app.callback(
        Output('forecast-assumptions-modal', 'is_open'),
        Input('open-forecast-modal-btn', 'n_clicks'),
        State('forecast-assumptions-modal', 'is_open'),
        prevent_initial_call=True
    )
    def toggle_forecast_modal(n_clicks, is_open):
        if n_clicks: return not is_open
        return is_open
    
    @app.callback(
        Output('open-forecast-modal-btn', 'style'),
        Input('table-tabs', 'active_tab')
    )
    def toggle_gear_button_visibility(active_tab):
        return {'display': 'inline-block'} if active_tab == 'tab-forecast' else {'display': 'none'}

    @app.callback(
        Output('forecast-assumptions-store', 'data', allow_duplicate=True),
        Output('forecast-assumptions-modal', 'is_open', allow_duplicate=True),
        Input('apply-forecast-changes-btn', 'n_clicks'),
        [State('modal-forecast-years-input', 'value'),
         State('modal-forecast-eps-growth-input', 'value'),
         State('modal-forecast-terminal-pe-input', 'value')],
        prevent_initial_call=True
    )
    def save_forecast_assumptions(n_clicks, years, growth, pe):
        if n_clicks:
            new_data = {'years': years, 'growth': growth, 'pe': pe}
            if current_user.is_authenticated:
                with server.app_context():
                    assumptions = UserAssumptions.query.filter_by(user_id=current_user.id).first()
                    if not assumptions:
                        assumptions = UserAssumptions(user_id=current_user.id)
                        db.session.add(assumptions)
                    assumptions.forecast_years, assumptions.eps_growth, assumptions.terminal_pe = years, growth, pe
                    db.session.commit()
            return new_data, False
        return dash.no_update, dash.no_update

    @app.callback(
        Output("definitions-modal", "is_open"),
        Output("definitions-modal-title", "children"),
        Output("definitions-modal-body", "children"),
        [Input("open-definitions-modal-btn-graphs", "n_clicks"),
         Input("open-definitions-modal-btn-tables", "n_clicks"),
         Input("close-definitions-modal-btn", "n_clicks")],
        [State("definitions-modal", "is_open"),
         State("analysis-tabs", "active_tab"),
         State("table-tabs", "active_tab")],
        prevent_initial_call=True
    )
    def toggle_definitions_modal(graphs_clicks, tables_clicks, close_clicks, is_open, analysis_tab, table_tab):
        ctx = callback_context
        if not ctx.triggered or ctx.triggered_id == "close-definitions-modal-btn":
            return False, dash.no_update, dash.no_update
        
        tab_id = analysis_tab if ctx.triggered_id == "open-definitions-modal-btn-graphs" else table_tab
        tab_config = TABS_CONFIG.get(tab_id, {})
        tab_name = tab_config.get('tab_name', 'Metric')
        title = f"{tab_name.upper()} DEFINITION"
        
        body_content = []
        if ctx.triggered_id == "open-definitions-modal-btn-graphs":
             body_content = METRIC_DEFINITIONS.get(tab_id, html.P("No definition available."))
        else:
            columns_in_tab = tab_config.get('columns', [])
            for col in columns_in_tab:
                if col in METRIC_DEFINITIONS:
                    body_content.append(METRIC_DEFINITIONS[col])
                    body_content.append(html.Hr())
            if body_content: body_content.pop()
            else: body_content = [html.P("No definitions for this tab.")]
            
        return True, title, body_content
        
    @app.callback(
        Output('open-dcf-modal-btn', 'style'),
        Input('analysis-tabs', 'active_tab')
    )
    def toggle_dcf_gear_button_visibility(active_tab):
        return {'display': 'inline-block'} if active_tab == 'tab-dcf' else {'display': 'none'}

    @app.callback(
        Output('dcf-assumptions-modal', 'is_open'),
        Input('open-dcf-modal-btn', 'n_clicks'),
        State('dcf-assumptions-modal', 'is_open'),
        prevent_initial_call=True
    )
    def toggle_dcf_modal(n_clicks, is_open):
        if n_clicks: return not is_open
        return is_open

    @app.callback(
        Output('dcf-assumptions-store', 'data', allow_duplicate=True),
        Output('dcf-assumptions-modal', 'is_open', allow_duplicate=True),
        Input('apply-dcf-changes-btn', 'n_clicks'),
        [State('mc-dcf-simulations-input', 'value'),
         State('mc-dcf-growth-min', 'value'), State('mc-dcf-growth-mode', 'value'), State('mc-dcf-growth-max', 'value'),
         State('mc-dcf-perpetual-min', 'value'), State('mc-dcf-perpetual-mode', 'value'), State('mc-dcf-perpetual-max', 'value'),
         State('mc-dcf-wacc-min', 'value'), State('mc-dcf-wacc-mode', 'value'), State('mc-dcf-wacc-max', 'value')],
        prevent_initial_call=True
    )
    def save_dcf_assumptions(n_clicks, sims, g_min, g_mode, g_max, p_min, p_mode, p_max, w_min, w_mode, w_max):
        if n_clicks:
            new_data = {
                'simulations': sims, 'growth_min': g_min, 'growth_mode': g_mode, 'growth_max': g_max,
                'perpetual_min': p_min, 'perpetual_mode': p_mode, 'perpetual_max': p_max,
                'wacc_min': w_min, 'wacc_mode': w_mode, 'wacc_max': w_max,
            }
            return new_data, False
        return dash.no_update, dash.no_update

    @app.callback(
        [Output('mc-dcf-simulations-input', 'value'),
         Output('mc-dcf-growth-min', 'value'), Output('mc-dcf-growth-mode', 'value'), Output('mc-dcf-growth-max', 'value'),
         Output('mc-dcf-perpetual-min', 'value'), Output('mc-dcf-perpetual-mode', 'value'), Output('mc-dcf-perpetual-max', 'value'),
         Output('mc-dcf-wacc-min', 'value'), Output('mc-dcf-wacc-mode', 'value'), Output('mc-dcf-wacc-max', 'value')],
        Input('dcf-assumptions-store', 'data')
    )
    def sync_dcf_modal_inputs(dcf_data):
        if not dcf_data:
            return 10000, 3, 5, 8, 1.5, 2.5, 3.0, 7.0, 8.0, 10.0
        return (dcf_data.get('simulations', 10000), dcf_data.get('growth_min', 3), dcf_data.get('growth_mode', 5), dcf_data.get('growth_max', 8),
                dcf_data.get('perpetual_min', 1.5), dcf_data.get('perpetual_mode', 2.5), dcf_data.get('perpetual_max', 3.0),
                dcf_data.get('wacc_min', 7.0), dcf_data.get('wacc_mode', 8.0), dcf_data.get('wacc_max', 10.0))

    @app.callback(
        [Output('modal-forecast-years-input', 'value'), Output('modal-forecast-eps-growth-input', 'value'), Output('modal-forecast-terminal-pe-input', 'value')],
        Input('forecast-assumptions-store', 'data')
    )
    def sync_forecast_modal_inputs(forecast_data):
        if not forecast_data: return 5, 10, 20
        return forecast_data.get('years', 5), forecast_data.get('growth', 10), forecast_data.get('pe', 20)

    @app.callback(
        Output('analysis-pane-content', 'children'),
        [Input('analysis-tabs', 'active_tab'),
         Input('user-selections-store', 'data'),
         Input('dcf-assumptions-store', 'data')]
    )
    def render_graph_content(active_tab, store_data, dcf_data):
        store_data = store_data or {'tickers': [], 'indices': []}
        tickers = tuple(store_data.get('tickers', []))
        indices = tuple(store_data.get('indices', []))

        if active_tab == "tab-performance":
            all_symbols = tuple(set(tickers + indices))
            if not all_symbols: return dbc.Alert("Please select items to display the chart", color="info", className="mt-3 text-center")
            try:
                raw_data = yf.download(list(all_symbols), period="ytd", auto_adjust=True, progress=False)['Close']
                if isinstance(raw_data, pd.Series): raw_data = raw_data.to_frame(name=all_symbols[0])
                ytd_data = raw_data.dropna(axis=1, how='all').ffill()
                if ytd_data.empty or len(ytd_data) < 2: raise ValueError("Not enough data.")
                ytd_perf = (ytd_data / ytd_data.iloc[0]) - 1
                ytd_perf = ytd_perf.rename(columns=INDEX_TICKER_TO_NAME)
                fig = px.line(ytd_perf, title='YTD Performance Comparison', color_discrete_map=COLOR_DISCRETE_MAP)
                fig.update_layout(yaxis_tickformat=".2%", legend_title_text='Symbol')
                return dbc.Card(dbc.CardBody(dcc.Graph(figure=fig)))
            except Exception as e: return dbc.Alert(f"An error occurred while rendering 'YTD Performance': {e}", color="danger")

        if active_tab == "tab-drawdown":
            all_symbols = tuple(set(tickers + indices))
            if not all_symbols: return dbc.Alert("Please select items to display the chart", color="info", className="mt-3 text-center")
            try:
                drawdown_data = calculate_drawdown(all_symbols, period="1y").rename(columns=INDEX_TICKER_TO_NAME)
                if drawdown_data.empty: raise ValueError("Could not calculate drawdown data.")
                fig = px.line(drawdown_data, title='1-Year Drawdown Comparison', color_discrete_map=COLOR_DISCRETE_MAP)
                fig.update_layout(yaxis_tickformat=".2%", legend_title_text='Symbol')
                return dbc.Card(dbc.CardBody(dcc.Graph(figure=fig)))
            except Exception as e: return dbc.Alert(f"An error occurred while rendering 'Drawdown': {e}", color="danger")

        if active_tab == "tab-scatter":
            if not tickers: return dbc.Alert("Please select stocks to display the chart.", color="info", className="mt-3 text-center")
            try:
                df_scatter = get_scatter_data(tickers)
                if df_scatter.empty: return dbc.Alert("Could not fetch data for some of the selected stocks.", color="warning")
                fig = px.scatter(df_scatter, x="EBITDA Margin", y="EV/EBITDA", text="Ticker", title="Valuation vs. Quality Analysis")
                fig.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
                fig.update_layout(xaxis_tickformat=".2%", yaxis_title="EV / EBITDA (Valuation)", xaxis_title="EBITDA Margin (Quality)")
                x_avg, y_avg = df_scatter["EBITDA Margin"].mean(), df_scatter["EV/EBITDA"].mean()
                fig.add_vline(x=x_avg, line_width=1, line_dash="dash", line_color="grey")
                fig.add_hline(y=y_avg, line_width=1, line_dash="dash", line_color="grey")
                return dbc.Card(dbc.CardBody(dcc.Graph(figure=fig)))
            except Exception as e: return dbc.Alert(f"An error occurred while rendering Scatter Plot: {e}", color="danger")

        if active_tab == "tab-dcf":
            if not tickers: return dbc.Alert("Please select stocks for DCF simulation.", color="info", className="mt-3 text-center")
            if not dcf_data: return dbc.Alert("Please set simulation assumptions using the gear icon.", color="info", className="mt-3 text-center")
            
            all_results = []
            for ticker in tickers:
                result = calculate_monte_carlo_dcf(
                    ticker=ticker,
                    n_simulations=dcf_data.get('simulations', 10000),
                    growth_min=dcf_data.get('growth_min', 3), growth_mode=dcf_data.get('growth_mode', 5), growth_max=dcf_data.get('growth_max', 8),
                    perpetual_min=dcf_data.get('perpetual_min', 1.5), perpetual_mode=dcf_data.get('perpetual_mode', 2.5), perpetual_max=dcf_data.get('perpetual_max', 3.0),
                    wacc_min=dcf_data.get('wacc_min', 7.0), wacc_mode=dcf_data.get('wacc_mode', 8.0), wacc_max=dcf_data.get('wacc_max', 10.0),
                )
                if 'error' not in result:
                    result['Ticker'] = ticker
                    all_results.append(result)

            if not all_results: return dbc.Alert("Could not run simulation for any selected stocks.", color="danger")
            
            # [MODIFIED] Removed height=800
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])

            for res in all_results:
                fig.add_trace(go.Histogram(x=res['simulated_values'], name=res['Ticker'], opacity=0.6, nbinsx=100), row=1, col=1)

            mos_data = [{'Ticker': r['Ticker'], 'current_price': r['current_price'], 'intrinsic_value': r['mean']} for r in all_results]
            df_mos = pd.DataFrame(mos_data)
            
            fig.add_trace(go.Scatter(x=df_mos['current_price'], y=df_mos['Ticker'], mode='markers', marker=dict(color='royalblue', size=10), name='Current Price'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_mos['intrinsic_value'], y=df_mos['Ticker'], mode='markers', marker=dict(color='darkorange', size=10, symbol='diamond'), name='Mean Intrinsic Value'), row=2, col=1)

            for i, row in df_mos.iterrows():
                fig.add_shape(type='line', x0=row['current_price'], y0=row['Ticker'], x1=row['intrinsic_value'], y1=row['Ticker'],
                              line=dict(color='limegreen' if row['intrinsic_value'] > row['current_price'] else 'tomato', width=3), row=2, col=1)

            # [MODIFIED] Removed height=800
            fig.update_layout(title_text='Monte Carlo DCF Analysis', barmode='overlay', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig.update_yaxes(title_text="Frequency", row=1, col=1)
            fig.update_yaxes(title_text="Ticker", row=2, col=1)
            fig.update_xaxes(title_text="Share Price ($)", row=2, col=1)

            return dbc.Card(dbc.CardBody(dcc.Graph(figure=fig)))

        return html.P("This is an empty tab!")

    @app.callback(
        Output('table-pane-content', 'children'),
        Output('sort-by-dropdown', 'options'),
        Output('sort-by-dropdown', 'value'),
        [Input('table-tabs', 'active_tab'),
         Input('user-selections-store', 'data'),
         Input('sort-by-dropdown', 'value'),
         Input('forecast-assumptions-store', 'data')]
    )
    def render_table_content(active_tab, store_data, sort_by_column, forecast_data):
        try:
            if not store_data or not store_data.get('tickers'):
                return dbc.Alert("Please select stocks to view the comparison table.", color="info", className="mt-3 text-center"), [], None

            tickers = tuple(store_data.get('tickers'))
            df_full = get_competitor_data(tickers)

            if df_full.empty:
                return dbc.Alert(f"Could not fetch complete data for the selected tickers: {', '.join(tickers)}", color="warning", className="mt-3 text-center"), [], None

            if active_tab == 'tab-forecast':
                forecast_years, eps_growth, terminal_pe = forecast_data.get('years'), forecast_data.get('growth'), forecast_data.get('pe')
                if all([forecast_years, eps_growth, terminal_pe]):
                    forecast_results = [{'Ticker': ticker, **calculate_exit_multiple_valuation(ticker, forecast_years, eps_growth, terminal_pe)} for ticker in df_full['Ticker']]
                    df_full = pd.merge(df_full, pd.DataFrame(forecast_results), on='Ticker', how='left')

            df_full = apply_custom_scoring(df_full)
            config = TABS_CONFIG[active_tab]

            if sort_by_column and sort_by_column in df_full.columns:
                ascending = not config['higher_is_better'].get(sort_by_column, True)
                df_full.sort_values(by=sort_by_column, ascending=ascending, na_position='last', inplace=True)

            if active_tab == 'tab-forecast' and any(col not in df_full.columns for col in config["columns"]):
                 return dbc.Alert("Could not calculate forecasts. Please check assumptions (gear icon).", color="info"), [], None

            df_display = _prepare_display_dataframe(df_full)
            columns = _generate_datatable_columns(config)
            style_data_conditional, style_cell_conditional = _generate_datatable_style_conditionals(config)
            dropdown_options = [{'label': col, 'value': col} for col in config["columns"] if col not in ['Ticker', 'Company Size', 'Volatility Level', 'Valuation Model', 'Stock Profile']]
            data = df_display[config["columns"]].to_dict('records')
            
            return dash_table.DataTable(
                id='interactive-datatable', data=data, columns=columns,
                style_data_conditional=style_data_conditional, style_cell_conditional=style_cell_conditional,
                row_selectable=False, cell_selectable=False,
                style_header={'border': '0px', 'backgroundColor': 'transparent', 'fontWeight': '600', 'textTransform': 'uppercase', 'textAlign': 'right'},
                style_data={'border': '0px', 'backgroundColor': 'transparent'},
                style_cell={'textAlign': 'right', 'padding': '14px', 'verticalAlign': 'middle'},
                style_header_conditional=[{'if': {'column_id': 'Ticker'}, 'textAlign': 'left'}],
                markdown_options={"html": True}
            ), dropdown_options, sort_by_column

        except Exception as e:
            logging.error(f"Error rendering table content: {e}", exc_info=True)
            return dbc.Alert(f"An unexpected error occurred: {e}", color="danger", className="mt-3"), [], None
# callbacks.py (Final Version with UX updates)

import dash
from dash import dcc, html, callback_context, dash_table
from dash.dependencies import Input, Output, State, ALL
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
import numpy as np
import json
import logging
from flask_login import current_user

# Import core app objects from app.py
from app import app, db, server, User, UserSelection

# Import layout components from layout.py
from layout import build_layout, create_navbar

# Import helpers and other page layouts
from data_handler import (
    calculate_drawdown, get_competitor_data, get_scatter_data,
    calculate_dcf_intrinsic_value,
    calculate_exit_multiple_valuation
)
from pages import deep_dive
from constants import (
    INDEX_TICKER_TO_NAME, SECTOR_TO_INDEX_MAPPING, COLOR_DISCRETE_MAP, SECTORS
)
from auth import create_register_layout

# ==================================================================
# Table Helpers & Config
# ==================================================================
TABS_CONFIG = {
    # Graph Tabs (for definitions modal)
    "tab-performance": { "tab_name": "Performance" },
    "tab-drawdown": { "tab_name": "Drawdown" },
    "tab-scatter": { "tab_name": "Valuation vs. Quality" },
    "tab-dcf": { "tab_name": "Margin of Safety (DCF)" },
    # Table Tabs
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
            df_display[col] = df_raw[col].apply(
                lambda x: x if pd.notna(x) else ""
            )
    return df_display

def _generate_datatable_columns(tab_config):
    columns = []
    for col in tab_config["columns"]:
        col_def = {"name": col.replace(" (3Y)", ""), "id": col}
        if col == 'Ticker':
            col_def.update({"type": "text", "presentation": "markdown"})
        elif col in ['Company Size', 'Volatility Level', 'Valuation Model', 'Stock Profile', 'Market Cap']:
             col_def.update({"type": "text"})
        elif col in ['Target Upside', 'IRR %']:
            col_def.update({"type": "numeric", "format": Format(precision=2, scheme=Scheme.percentage)})
        elif col == 'Target Price':
            col_def.update({"type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)})
        elif any(kw in col for kw in ['Growth', 'Margin', 'ROE', 'Conversion']):
            col_def.update({"type": "numeric", "format": Format(precision=2, scheme=Scheme.percentage)})
        else:
            col_def.update({"type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)})
        columns.append(col_def)
    return columns

def _generate_datatable_style_conditionals(tab_config):
    style_data_conditional = []
    style_cell_conditional = []
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
    if df.empty:
        return df
    bins = [0, 1e9, 10e10, 100e10, float('inf')]
    labels = ["Small Cap", "Mid Cap", "Large Cap", "Mega Cap"]
    df['Company Size'] = pd.cut(df['Market Cap'], bins=bins, labels=labels, right=False)
    conditions_volatility = [df['Beta'].isnull(), df['Beta'] < 0.5, df['Beta'] <= 2, df['Beta'] > 2]
    choices_volatility = ["N/A", "Core", "Growth", "Hyper Growth"]
    df['Volatility Level'] = np.select(conditions_volatility, choices_volatility, default='N/A')
    conditions_valuation = [df['EV/EBITDA'].isnull(), df['EV/EBITDA'] < 10, df['EV/EBITDA'] <= 25, df['EV/EBITDA'] > 25]
    choices_valuation = ["N/A", "Cheap", "Fair Value", "Expensive"]
    df['Valuation Model'] = np.select(conditions_valuation, choices_valuation, default='N/A')
    def categorize_stock(row):
        vol = row['Volatility Level']
        val = row['Valuation Model']
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

    @app.callback(Output('user-selections-store', 'data'), Input('url', 'pathname'))
    def load_user_selections_to_store(pathname):
        if pathname == '/':
            with server.app_context():
                if current_user.is_authenticated:
                    stocks = UserSelection.query.filter_by(user_id=current_user.id, symbol_type='stock').all()
                    indices = UserSelection.query.filter_by(user_id=current_user.id, symbol_type='index').all()
                    return {'tickers': [s.symbol for s in stocks], 'indices': [i.symbol for i in indices]}
        return {'tickers': [], 'indices': []}

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
            return html.Div([
                html.Span("No stocks selected.", className="text-muted fst-italic")
            ])
        return [html.Label("Selected Stocks:", className="text-muted small")] + [dbc.Badge([t, html.I(className="bi bi-x-circle-fill ms-2", style={'cursor': 'pointer'}, id={'type': 'remove-stock', 'index': t})], color="light", className="m-1 p-2 text-dark border") for t in tickers]

    @app.callback(Output('index-summary-display', 'children'), Input('user-selections-store', 'data'))
    def update_index_summary_display(store_data):
        indices = store_data.get('indices', []) if store_data else []
        if not indices:
            return html.Span("No indices selected.", className="text-muted fst-italic")
        return [html.Label("Selected Indices:", className="text-muted small")] + [dbc.Badge([INDEX_TICKER_TO_NAME.get(i, i), html.I(className="bi bi-x-circle-fill ms-2", style={'cursor': 'pointer'}, id={'type': 'remove-index', 'index': i})], color="light", className="m-1 p-2 text-dark border") for i in indices]

    @app.callback(Output('ticker-select-dropdown', 'options'), [Input('sector-dropdown', 'value'), Input('user-selections-store', 'data')])
    def update_ticker_options(selected_sector, store_data):
        if not selected_sector: return []
        selected_tickers = store_data.get('tickers', []) if store_data else []
        tickers_to_display = []
        if selected_sector == 'All':
            all_tickers_list = [t for tickers in SECTORS.values() for t in tickers]
            tickers_to_display = sorted(list(set(all_tickers_list)))
        else:
            tickers_to_display = sorted(SECTORS.get(selected_sector, []))
        return [{'label': t, 'value': t} for t in tickers_to_display if t not in selected_tickers]

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
        if n_clicks:
            return not is_open
        return is_open
    
    @app.callback(
        Output('open-forecast-modal-btn', 'style'),
        Input('table-tabs', 'active_tab')
    )
    def toggle_gear_button_visibility(active_tab):
        if active_tab == 'tab-forecast':
            return {'display': 'inline-block'}
        else:
            return {'display': 'none'}

    @app.callback(
        Output('forecast-assumptions-store', 'data'),
        Output('forecast-assumptions-modal', 'is_open', allow_duplicate=True),
        Input('apply-forecast-changes-btn', 'n_clicks'),
        [State('modal-forecast-years-input', 'value'),
         State('modal-forecast-eps-growth-input', 'value'),
         State('modal-forecast-terminal-pe-input', 'value')],
        prevent_initial_call=True
    )
    def save_forecast_assumptions(n_clicks, years, growth, pe):
        if n_clicks:
            return {'years': years, 'growth': growth, 'pe': pe}, False
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

        triggered_id = ctx.triggered_id
        
        if triggered_id == "open-definitions-modal-btn-graphs":
            tab_name = TABS_CONFIG[analysis_tab].get('tab_name', 'Chart')
            title = f"{tab_name.upper()} DEFINITION"
            body_content = METRIC_DEFINITIONS.get(analysis_tab, html.P("No definition available for this chart."))
            return True, title, body_content

        if triggered_id == "open-definitions-modal-btn-tables":
            tab_name = TABS_CONFIG[table_tab].get('tab_name', 'Table')
            title = f"{tab_name.upper()} METRIC DEFINITIONS"
            columns_in_tab = TABS_CONFIG[table_tab].get('columns', [])
            body_content = []
            for col in columns_in_tab:
                if col in METRIC_DEFINITIONS:
                    body_content.append(METRIC_DEFINITIONS[col])
                    body_content.append(html.Hr())
            if not body_content:
                body_content = [html.P("No specific definitions for this tab.")]
            else:
                 body_content.pop()
            return True, title, body_content
        
        return is_open, dash.no_update, dash.no_update
        
    @app.callback(
        Output('open-dcf-modal-btn', 'style'),
        Input('analysis-tabs', 'active_tab')
    )
    def toggle_dcf_gear_button_visibility(active_tab):
        if active_tab == 'tab-dcf':
            return {'display': 'inline-block'}
        else:
            return {'display': 'none'}

    @app.callback(
        Output('dcf-assumptions-modal', 'is_open'),
        Input('open-dcf-modal-btn', 'n_clicks'),
        State('dcf-assumptions-modal', 'is_open'),
        prevent_initial_call=True
    )
    def toggle_dcf_modal(n_clicks, is_open):
        if n_clicks:
            return not is_open
        return is_open

    @app.callback(
        Output('dcf-assumptions-store', 'data'),
        Output('dcf-assumptions-modal', 'is_open', allow_duplicate=True),
        Input('apply-dcf-changes-btn', 'n_clicks'),
        [State('modal-dcf-forecast-growth-input', 'value'),
         State('modal-dcf-perpetual-growth-input', 'value'),
         State('modal-dcf-wacc-input', 'value')],
        prevent_initial_call=True
    )
    def save_dcf_assumptions(n_clicks, forecast_growth, perpetual_growth, wacc):
        if n_clicks:
            return {'forecast_growth': forecast_growth, 'perpetual_growth': perpetual_growth, 'wacc': wacc}, False
        return dash.no_update, dash.no_update

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
                raw_data = yf.download(all_symbols, period="ytd", auto_adjust=True, progress=False)
                ytd_data = raw_data.get('Close', pd.DataFrame())
                if isinstance(ytd_data, pd.Series): ytd_data = ytd_data.to_frame(name=all_symbols[0])
                ytd_data = ytd_data.dropna(axis=1, how='all').ffill()
                if ytd_data.empty or len(ytd_data) < 2: raise ValueError("Not enough data.")
                ytd_perf = (ytd_data / ytd_data.iloc[0]) - 1; ytd_perf = ytd_perf.rename(columns=INDEX_TICKER_TO_NAME)
                fig = px.line(ytd_perf, title='YTD Performance Comparison', color_discrete_map=COLOR_DISCRETE_MAP)
                fig.update_layout(yaxis_tickformat=".2%", legend_title_text='Symbol'); return dbc.Card(dbc.CardBody(dcc.Graph(figure=fig)))
            except Exception as e: return dbc.Alert(f"An error occurred while rendering 'YTD Performance': {e}", color="danger")

        if active_tab == "tab-drawdown":
            all_symbols = tuple(set(tickers + indices))
            if not all_symbols: return dbc.Alert("Please select items to display the chart", color="info", className="mt-3 text-center")
            try:
                drawdown_data = calculate_drawdown(all_symbols, period="1y")
                if drawdown_data.empty: raise ValueError("Could not calculate drawdown data.")
                drawdown_data = drawdown_data.rename(columns=INDEX_TICKER_TO_NAME)
                fig = px.line(drawdown_data, title='1-Year Drawdown Comparison', color_discrete_map=COLOR_DISCRETE_MAP)
                fig.update_layout(yaxis_tickformat=".2%", legend_title_text='Symbol'); return dbc.Card(dbc.CardBody(dcc.Graph(figure=fig)))
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
            if not tickers: return dbc.Alert("Please select stocks to display the chart.", color="info", className="mt-3 text-center")
            try:
                forecast_growth = dcf_data.get('forecast_growth', 5) / 100.0
                perpetual_growth = dcf_data.get('perpetual_growth', 2.5) / 100.0
                wacc_override = dcf_data.get('wacc', 8.0) / 100.0
                
                dcf_results = [calculate_dcf_intrinsic_value(
                    t,
                    forecast_growth_rate=forecast_growth,
                    perpetual_growth_rate=perpetual_growth,
                    wacc_override=wacc_override
                ) for t in tickers]
                
                successful_results = [res for res in dcf_results if 'error' not in res]
                failed_results = [res for res in dcf_results if 'error' in res]
                output_components = []

                if failed_results:
                    error_messages = {res['Ticker']: res['error'] for res in failed_results}
                    alerts = [dbc.Alert([html.I(className="bi bi-exclamation-triangle-fill me-2"), f"Could not calculate DCF for {ticker}: {reason}"], color="warning", className="mb-1", dismissable=True) for ticker, reason in error_messages.items()]
                    output_components.extend(alerts)

                if successful_results:
                    df_dcf = pd.DataFrame(successful_results)
                    fig = go.Figure()
                    for i, row in df_dcf.iterrows():
                        fig.add_shape(type='line', x0=row['current_price'], y0=row['Ticker'], x1=row['intrinsic_value'], y1=row['Ticker'], line=dict(color='limegreen' if row['intrinsic_value'] > row['current_price'] else 'tomato', width=3))
                    fig.add_trace(go.Scatter(x=df_dcf['current_price'], y=df_dcf['Ticker'], mode='markers', marker=dict(color='royalblue', size=10), name='Current Price'))
                    fig.add_trace(go.Scatter(x=df_dcf['intrinsic_value'], y=df_dcf['Ticker'], mode='markers', marker=dict(color='darkorange', size=10, symbol='diamond'), name='Intrinsic Value (DCF)'))
                    title_text = f'Margin of Safety (DCF) with {forecast_growth:.1%} Growth Forecast'
                    fig.update_layout(title=title_text, xaxis_title='Share Price ($)', yaxis_title='Ticker', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    output_components.append(dbc.Card(dbc.CardBody(dcc.Graph(figure=fig))))
                
                if not output_components:
                    return dbc.Alert("Could not process DCF for any selected stocks.", color="danger")

                return output_components
            except Exception as e: return dbc.Alert(f"An error occurred while rendering DCF chart: {e}", color="danger")
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
                alert = dbc.Alert("Please select stocks to view the comparison table.", color="info", className="mt-3 text-center")
                return alert, [], None

            tickers = tuple(store_data.get('tickers'))
            df_full = get_competitor_data(tickers)

            if df_full.empty:
                alert = dbc.Alert(f"Could not fetch complete data for the selected tickers: {', '.join(tickers)}", color="warning", className="mt-3 text-center")
                return alert, [], None

            if active_tab == 'tab-forecast':
                forecast_years = forecast_data.get('years')
                eps_growth = forecast_data.get('growth')
                terminal_pe = forecast_data.get('pe')
                if all([forecast_years, eps_growth, terminal_pe]):
                    forecast_results = []
                    for ticker in df_full['Ticker']:
                        result = calculate_exit_multiple_valuation(ticker, forecast_years, eps_growth, terminal_pe)
                        result['Ticker'] = ticker
                        forecast_results.append(result)
                    df_forecast = pd.DataFrame(forecast_results)
                    df_full = pd.merge(df_full, df_forecast, on='Ticker', how='left')

            df_full = apply_custom_scoring(df_full)
            config = TABS_CONFIG[active_tab]

            if sort_by_column and sort_by_column in df_full.columns:
                ascending = not config['higher_is_better'].get(sort_by_column, True)
                df_full.sort_values(by=sort_by_column, ascending=ascending, na_position='last', inplace=True)

            missing_cols = [col for col in config["columns"] if col not in df_full.columns]
            if active_tab == 'tab-forecast' and missing_cols:
                 return dbc.Alert(f"Could not calculate {', '.join(missing_cols)}. Please check assumptions using the gear icon.", color="info"), [], None

            df_display = _prepare_display_dataframe(df_full)
            columns = _generate_datatable_columns(config)
            style_data_conditional, style_cell_conditional = _generate_datatable_style_conditionals(config)
            dropdown_options = [{'label': col, 'value': col} for col in config["columns"] if col not in ['Ticker', 'Company Size', 'Volatility Level', 'Valuation Model', 'Stock Profile']]
            data = df_display[config["columns"]].to_dict('records')
            sort_value = sort_by_column if callback_context.triggered_id == 'sort-by-dropdown' else None

            datatable = dash_table.DataTable(
                id='interactive-datatable',
                data=data,
                columns=columns,
                style_data_conditional=style_data_conditional,
                style_cell_conditional=style_cell_conditional,
                row_selectable=False,
                cell_selectable=False,
                style_header={'border': '0px', 'backgroundColor': 'transparent', 'fontWeight': '600', 'textTransform': 'uppercase', 'textAlign': 'right'},
                style_data={'border': '0px', 'backgroundColor': 'transparent'},
                style_cell={'textAlign': 'right', 'padding': '14px', 'verticalAlign': 'middle'},
                style_header_conditional=[{'if': {'column_id': 'Ticker'}, 'textAlign': 'left'}],
                markdown_options={"html": True}
            )
            return datatable, dropdown_options, sort_value

        except Exception as e:
            logging.error(f"Error rendering table content: {e}", exc_info=True)
            alert = dbc.Alert(f"An unexpected error occurred while building the table: {e}", color="danger", className="mt-3")
            return alert, [], None
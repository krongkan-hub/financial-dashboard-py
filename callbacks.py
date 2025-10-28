# callbacks.py (Refactored - Step 4 - FIXED pd.read_sql TypeError + Added Smart Peer Finder)
# (เวอร์ชันสมบูรณ์ - เปลี่ยนไปดึงข้อมูลจาก DB ทั้งหมด)

import dash
from dash import dcc, html, callback_context, dash_table
from dash.dependencies import Input, Output, State, ALL
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
import logging
from flask_login import current_user

# Import core app objects from app.py
from app import app, db, server, User, UserSelection, UserAssumptions
# --- [REFACTOR STEP 4 IMPORTS] ---
# Import models ใหม่ที่เราจะใช้ Query
from app import DimCompany, FactCompanySummary, FactDailyPrices, FactFinancialStatements # <<< [UPDATE: Added FactFinancialStatements]
from sqlalchemy import func, distinct # <<< [MODIFIED] Import SQL functions (added distinct)
from datetime import datetime, timedelta # Import datetime
# --- [END REFACTOR STEP 4 IMPORTS] ---

# Import layout components from layout.py
from layout import build_layout, create_navbar

# Import helpers and other page layouts
from data_handler import (
    # calculate_drawdown, get_scatter_data, (ถูกลบออก)
    # calculate_exit_multiple_valuation, (ถูกลบออก)
    calculate_monte_carlo_dcf # ถูกปรับปรุงให้ใช้ DB ภายใน
)
from pages import deep_dive
from constants import (
    INDEX_TICKER_TO_NAME, SECTOR_TO_INDEX_MAPPING, COLOR_DISCRETE_MAP, SECTORS,
    TOP_5_DEFAULT_TICKERS, ALL_TICKERS_SORTED_BY_MC
)
from auth import create_register_layout

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==================================================================
# Table Helpers & Config (เหมือนเดิม)
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
    # ใช้ค่าจริงจาก DB ที่เป็น float
    return f'${n/1e9:,.2f}B' if abs(n) >= 1e9 else f'${n/1e6:,.2f}M'

def _prepare_display_dataframe(df_raw):
    # ฟังก์ชันนี้รับ DataFrame ที่มีคอลัมน์จาก DB + logo_url แล้ว
    df_display = df_raw.copy()
    def create_ticker_cell(row):
        # ใช้ logo_url จาก df_raw โดยตรง (ซึ่งมาจากการ Join)
        logo_url, ticker = row.get('logo_url'), row['Ticker']
        logo_html = f'<img src="{logo_url}" style="height: 22px; width: 22px; margin-right: 8px; border-radius: 4px;" onerror="this.style.display=\'none\'">' if logo_url and pd.notna(logo_url) else ''
        return f'''<a href="/deepdive/{ticker}" style="text-decoration: none; color: inherit; font-weight: 600; display: flex; align-items: center;">{logo_html}<span>{ticker}</span></a>'''

    df_display['Ticker'] = df_display.apply(create_ticker_cell, axis=1)

    # จัดรูปแบบ Market Cap ใหม่ (เพราะตอนนี้รับ float จาก DB)
    if "Market Cap" in df_display.columns:
        # ใช้ค่า float ดิบ 'market_cap_raw' ที่เราจะสร้างขึ้นมาแทน
        df_display["Market Cap"] = df_raw["market_cap_raw"].apply(_format_market_cap)

    # จัดการ Scoring Columns (เหมือนเดิม)
    scoring_cols = ['Company Size', 'Volatility Level', 'Valuation Model', 'Stock Profile']
    for col in scoring_cols:
        if col in df_display.columns:
            # ใช้ค่าจาก df_raw ที่ apply_custom_scoring คำนวณให้
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
    ticker_width_percent = 15 # อาจปรับตามความเหมาะสม
    style_cell_conditional.append({'if': {'column_id': 'Ticker'}, 'width': f'{ticker_width_percent}%'})
    if other_cols:
        other_col_width_percent = (100 - ticker_width_percent) / len(other_cols)
        for col in other_cols:
            style_cell_conditional.append({'if': {'column_id': col}, 'width': f'{other_col_width_percent}%'})
    style_cell_conditional.append({'if': {'column_id': 'Ticker'}, 'textAlign': 'left'})
    return style_data_conditional, style_cell_conditional


def apply_custom_scoring(df):
    if df.empty: return df

    # ใช้ market_cap (ตัวเล็ก) จาก DB
    bins = [0, 10e9, 100e9, 1000e9, float('inf')] # 10B, 100B, 1T
    labels = ["Small Cap", "Mid Cap", "Large Cap", "Mega Cap"]
    df['Company Size'] = pd.cut(df['market_cap'], bins=bins, labels=labels, right=False)

    # ใช้ beta (ตัวเล็ก) จาก DB
    conditions_volatility = [df['beta'].isnull(), df['beta'] < 0.5, df['beta'] <= 2, df['beta'] > 2]
    choices_volatility = ["N/A", "Core", "Growth", "Hyper Growth"]
    df['Volatility Level'] = np.select(conditions_volatility, choices_volatility, default='N/A')

    # ใช้ ev_ebitda (ตัวเล็ก) จาก DB
    conditions_valuation = [df['ev_ebitda'].isnull(), df['ev_ebitda'] < 10, df['ev_ebitda'] <= 25, df['ev_ebitda'] > 25]
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

    # --- Callbacks อื่นๆ ทั้งหมด (display_page, update_navbar, load_user_data_to_store, ---
    # --- add/remove tickers/indices, update summaries, update dropdown options, ---
    # --- toggle modals, save assumptions, sync modal inputs) ---
    # --- >>> เหมือนเดิม ยกเว้นส่วนที่เกี่ยวกับ Dropdown Options ที่มีการเพิ่มส่วน Peer Finder <<< ---
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

    @app.callback(
        Output('user-selections-store', 'data'),
        Output('forecast-assumptions-store', 'data'),
        Output('dcf-assumptions-store', 'data'),
        Input('url', 'pathname') # Trigger on page load/navigation
    )
    def load_user_data_to_store(pathname):
        default_dcf_data = {
            'simulations': 10000, 'growth_min': 3.0, 'growth_mode': 5.0, 'growth_max': 8.0,
            'perpetual_min': 1.5, 'perpetual_mode': 2.5, 'perpetual_max': 3.0,
            'wacc_min': 7.0, 'wacc_mode': 8.0, 'wacc_max': 10.0
        }
        default_forecast_data = {'years': 5, 'growth': 10, 'pe': 20}
        default_selections_data = {'tickers': TOP_5_DEFAULT_TICKERS, 'indices': ['^GSPC']}

        if pathname != '/register' and current_user.is_authenticated:
            with server.app_context():
                user_id = current_user.id
                stocks = UserSelection.query.filter_by(user_id=user_id, symbol_type='stock').all()
                indices = UserSelection.query.filter_by(user_id=user_id, symbol_type='index').all()
                if not stocks and not indices: selections_data = default_selections_data
                else: selections_data = {'tickers': [s.symbol for s in stocks], 'indices': [i.symbol for i in indices]}
                assumptions = UserAssumptions.query.filter_by(user_id=user_id).first()
                if assumptions:
                    forecast_data = {'years': assumptions.forecast_years, 'growth': assumptions.eps_growth, 'pe': assumptions.terminal_pe}
                    dcf_data = { 'simulations': assumptions.dcf_simulations, 'growth_min': assumptions.dcf_growth_min, 'growth_mode': assumptions.dcf_growth_mode, 'growth_max': assumptions.dcf_growth_max, 'perpetual_min': assumptions.dcf_perpetual_min, 'perpetual_mode': assumptions.dcf_perpetual_mode, 'perpetual_max': assumptions.dcf_perpetual_max, 'wacc_min': assumptions.dcf_wacc_min, 'wacc_mode': assumptions.dcf_wacc_mode, 'wacc_max': assumptions.dcf_wacc_max }
                else: forecast_data, dcf_data = default_forecast_data, default_dcf_data
                return selections_data, forecast_data, dcf_data
        return default_selections_data, default_forecast_data, default_dcf_data

    def save_selections_to_db(user_id, symbols, symbol_type):
        UserSelection.query.filter_by(user_id=user_id, symbol_type=symbol_type).delete(synchronize_session=False)
        for symbol in symbols: db.session.add(UserSelection(user_id=user_id, symbol_type=symbol_type, symbol=symbol))
        db.session.commit()

    @app.callback(Output('user-selections-store', 'data', allow_duplicate=True), Input('add-ticker-button', 'n_clicks'), [State('ticker-select-dropdown', 'value'), State('user-selections-store', 'data')], prevent_initial_call=True)
    def add_ticker_to_store(n_clicks, selected_tickers, store_data):
        store_data = store_data or {'tickers': [], 'indices': []}; updated = False
        if selected_tickers:
            for ticker in selected_tickers:
                if ticker not in store_data['tickers']: store_data['tickers'].append(ticker); updated = True
            if updated and current_user.is_authenticated:
                with server.app_context(): save_selections_to_db(current_user.id, store_data['tickers'], 'stock')
        return store_data

    @app.callback(Output('user-selections-store', 'data', allow_duplicate=True), Input({'type': 'remove-stock', 'index': ALL}, 'n_clicks'), State('user-selections-store', 'data'), prevent_initial_call=True)
    def remove_ticker_from_store(n_clicks, store_data):
        if not any(n_clicks): return dash.no_update
        store_data = store_data or {'tickers': [], 'indices': []}; triggered_id_str = callback_context.triggered[0]['prop_id'].split('.')[0]
        try:
            triggered_id = json.loads(triggered_id_str)['index']
            if triggered_id in store_data['tickers']:
                store_data['tickers'].remove(triggered_id)
                if current_user.is_authenticated:
                    with server.app_context(): save_selections_to_db(current_user.id, store_data['tickers'], 'stock')
                return store_data
        except (json.JSONDecodeError, KeyError): logging.warning(f"Could not parse triggered ID for remove-stock: {triggered_id_str}")
        return dash.no_update

    @app.callback(Output('user-selections-store', 'data', allow_duplicate=True), Input('add-index-button', 'n_clicks'), [State('index-select-dropdown', 'value'), State('user-selections-store', 'data')], prevent_initial_call=True)
    def add_index_to_store(n_clicks, selected_indices, store_data):
        store_data = store_data or {'tickers': [], 'indices': []}; updated = False
        if selected_indices:
            for index in selected_indices:
                if index not in store_data['indices']: store_data['indices'].append(index); updated = True
            if updated and current_user.is_authenticated:
                with server.app_context(): save_selections_to_db(current_user.id, store_data['indices'], 'index')
        return store_data

    @app.callback(Output('user-selections-store', 'data', allow_duplicate=True), Input({'type': 'remove-index', 'index': ALL}, 'n_clicks'), State('user-selections-store', 'data'), prevent_initial_call=True)
    def remove_index_from_store(n_clicks, store_data):
        if not any(n_clicks): return dash.no_update
        store_data = store_data or {'tickers': [], 'indices': []}; triggered_id_str = callback_context.triggered[0]['prop_id'].split('.')[0]
        try:
            triggered_id = json.loads(triggered_id_str)['index']
            if triggered_id in store_data['indices']:
                store_data['indices'].remove(triggered_id)
                if current_user.is_authenticated:
                    with server.app_context(): save_selections_to_db(current_user.id, store_data['indices'], 'index')
                return store_data
        except (json.JSONDecodeError, KeyError): logging.warning(f"Could not parse triggered ID for remove-index: {triggered_id_str}")
        return dash.no_update

    @app.callback(Output('ticker-summary-display', 'children'), Input('user-selections-store', 'data'))
    def update_ticker_summary_display(store_data):
        tickers = store_data.get('tickers', []) if store_data else []
        if not tickers: return html.Div([html.Span("No stocks selected.", className="text-muted fst-italic")])
        return [html.Label("Selected Stocks:", className="text-muted small")] + [dbc.Badge([t, html.I(className="bi bi-x-circle-fill ms-2", style={'cursor': 'pointer'}, id={'type': 'remove-stock', 'index': t})], color="light", className="m-1 p-2 text-dark border") for t in tickers]

    @app.callback(Output('index-summary-display', 'children'), Input('user-selections-store', 'data'))
    def update_index_summary_display(store_data):
        indices = store_data.get('indices', []) if store_data else []
        if not indices: return html.Span("No indices selected.", className="text-muted fst-italic")

        # แก้ไข 'i' เป็น 't'
        return [html.Label("Selected Indices:", className="text-muted small")] + [
            dbc.Badge([
                INDEX_TICKER_TO_NAME.get(t, t), # <-- เปลี่ยน i เป็น t
                html.I(className="bi bi-x-circle-fill ms-2", style={'cursor': 'pointer'}, id={'type': 'remove-index', 'index': t})
            ], color="light", className="m-1 p-2 text-dark border")
            for t in indices
        ]

    @app.callback(Output('ticker-select-dropdown', 'options'), [Input('sector-dropdown', 'value'), Input('user-selections-store', 'data')])
    def update_ticker_options(selected_sector, store_data):
        if not selected_sector: return []
        selected_tickers = store_data.get('tickers', []) if store_data else []
        tickers_to_display = ALL_TICKERS_SORTED_BY_MC if selected_sector == 'All' else SECTORS.get(selected_sector, [])
        return [{'label': t, 'value': t} for t in tickers_to_display if t not in selected_tickers]

    @app.callback(Output('index-select-dropdown', 'options'), Input('user-selections-store', 'data'))
    def update_index_options(store_data):
        if not store_data or not store_data.get('tickers'): return []
        selected_tickers, selected_indices = store_data.get('tickers', []), store_data.get('indices', [])
        ticker_to_sector = {t: s for s, stocks in SECTORS.items() for t in stocks}
        active_sectors = {ticker_to_sector.get(t) for t in selected_tickers if ticker_to_sector.get(t)}
        relevant_indices = {idx for sec in active_sectors for idx in SECTOR_TO_INDEX_MAPPING.get(sec, [])} | {'^GSPC', '^NDX'}
        return [{'label': INDEX_TICKER_TO_NAME.get(i, i), 'value': i} for i in sorted(list(relevant_indices)) if i not in selected_indices]

    @app.callback(Output('forecast-assumptions-modal', 'is_open'), Input('open-forecast-modal-btn', 'n_clicks'), State('forecast-assumptions-modal', 'is_open'), prevent_initial_call=True)
    def toggle_forecast_modal(n_clicks, is_open): return not is_open if n_clicks else is_open

    @app.callback(Output('open-forecast-modal-btn', 'style'), Input('table-tabs', 'active_tab'))
    def toggle_gear_button_visibility(active_tab): return {'display': 'inline-block'} if active_tab == 'tab-forecast' else {'display': 'none'}

    @app.callback(Output('forecast-assumptions-store', 'data', allow_duplicate=True), Output('forecast-assumptions-modal', 'is_open', allow_duplicate=True), Input('apply-forecast-changes-btn', 'n_clicks'), [State('modal-forecast-years-input', 'value'), State('modal-forecast-eps-growth-input', 'value'), State('modal-forecast-terminal-pe-input', 'value')], prevent_initial_call=True)
    def save_forecast_assumptions(n_clicks, years, growth, pe):
        if n_clicks:
            new_data = {'years': years, 'growth': growth, 'pe': pe}
            if current_user.is_authenticated:
                with server.app_context():
                    user_id = current_user.id; assumptions = UserAssumptions.query.filter_by(user_id=user_id).first()
                    if not assumptions: assumptions = UserAssumptions(user_id=user_id); db.session.add(assumptions)
                    assumptions.forecast_years, assumptions.eps_growth, assumptions.terminal_pe = years, growth, pe
                    db.session.commit()
            return new_data, False
        return dash.no_update, dash.no_update

    @app.callback(Output("definitions-modal", "is_open"), Output("definitions-modal-title", "children"), Output("definitions-modal-body", "children"), [Input("open-definitions-modal-btn-graphs", "n_clicks"), Input("open-definitions-modal-btn-tables", "n_clicks"), Input("close-definitions-modal-btn", "n_clicks")], [State("definitions-modal", "is_open"), State("analysis-tabs", "active_tab"), State("table-tabs", "active_tab")], prevent_initial_call=True)
    def toggle_definitions_modal(graphs_clicks, tables_clicks, close_clicks, is_open, analysis_tab, table_tab):
        ctx = callback_context;
        if not ctx.triggered or ctx.triggered_id == "close-definitions-modal-btn": return False, dash.no_update, dash.no_update
        tab_id = analysis_tab if ctx.triggered_id == "open-definitions-modal-btn-graphs" else table_tab
        tab_config = TABS_CONFIG.get(tab_id, {}); tab_name = tab_config.get('tab_name', 'Metric'); title = f"{tab_name.upper()} DEFINITION"; body_content = []
        if ctx.triggered_id == "open-definitions-modal-btn-graphs": body_content = METRIC_DEFINITIONS.get(tab_id, html.P("No definition available."))
        else:
            columns_in_tab = tab_config.get('columns', [])
            for col in columns_in_tab:
                if col in METRIC_DEFINITIONS: body_content.append(METRIC_DEFINITIONS[col]); body_content.append(html.Hr())
            if body_content: body_content.pop()
            else: body_content = [html.P("No definitions for this tab.")]
        return True, title, body_content

    @app.callback(Output('open-dcf-modal-btn', 'style'), Input('analysis-tabs', 'active_tab'))
    def toggle_dcf_gear_button_visibility(active_tab): return {'display': 'inline-block'} if active_tab == 'tab-dcf' else {'display': 'none'}

    @app.callback(Output('dcf-assumptions-modal', 'is_open'), Input('open-dcf-modal-btn', 'n_clicks'), State('dcf-assumptions-modal', 'is_open'), prevent_initial_call=True)
    def toggle_dcf_modal(n_clicks, is_open): return not is_open if n_clicks else is_open

    @app.callback(Output('dcf-assumptions-store', 'data', allow_duplicate=True), Output('dcf-assumptions-modal', 'is_open', allow_duplicate=True), Input('apply-dcf-changes-btn', 'n_clicks'), [State('mc-dcf-simulations-input', 'value'), State('mc-dcf-growth-min', 'value'), State('mc-dcf-growth-mode', 'value'), State('mc-dcf-growth-max', 'value'), State('mc-dcf-perpetual-min', 'value'), State('mc-dcf-perpetual-mode', 'value'), State('mc-dcf-perpetual-max', 'value'), State('mc-dcf-wacc-min', 'value'), State('mc-dcf-wacc-mode', 'value'), State('mc-dcf-wacc-max', 'value')], prevent_initial_call=True)
    def save_dcf_assumptions(n_clicks, sims, g_min, g_mode, g_max, p_min, p_mode, p_max, w_min, w_mode, w_max):
        if n_clicks:
            new_data = {'simulations': sims, 'growth_min': g_min, 'growth_mode': g_mode, 'growth_max': g_max, 'perpetual_min': p_min, 'perpetual_mode': p_mode, 'perpetual_max': p_max, 'wacc_min': w_min, 'wacc_mode': w_mode, 'wacc_max': w_max}
            if current_user.is_authenticated:
                with server.app_context():
                    user_id = current_user.id; assumptions = UserAssumptions.query.filter_by(user_id=user_id).first()
                    if not assumptions: assumptions = UserAssumptions(user_id=user_id); db.session.add(assumptions)
                    assumptions.dcf_simulations, assumptions.dcf_growth_min, assumptions.dcf_growth_mode, assumptions.dcf_growth_max = sims, g_min, g_mode, g_max
                    assumptions.dcf_perpetual_min, assumptions.dcf_perpetual_mode, assumptions.dcf_perpetual_max = p_min, p_mode, p_max
                    assumptions.dcf_wacc_min, assumptions.dcf_wacc_mode, assumptions.dcf_wacc_max = w_min, w_mode, w_max
                    db.session.commit()
            return new_data, False
        return dash.no_update, dash.no_update

    @app.callback([Output('mc-dcf-simulations-input', 'value'), Output('mc-dcf-growth-min', 'value'), Output('mc-dcf-growth-mode', 'value'), Output('mc-dcf-growth-max', 'value'), Output('mc-dcf-perpetual-min', 'value'), Output('mc-dcf-perpetual-mode', 'value'), Output('mc-dcf-perpetual-max', 'value'), Output('mc-dcf-wacc-min', 'value'), Output('mc-dcf-wacc-mode', 'value'), Output('mc-dcf-wacc-max', 'value')], Input('dcf-assumptions-store', 'data'))
    def sync_dcf_modal_inputs(dcf_data):
        if not dcf_data: return 10000, 3.0, 5.0, 8.0, 1.5, 2.5, 3.0, 7.0, 8.0, 10.0
        return (dcf_data.get('simulations', 10000), dcf_data.get('growth_min', 3.0), dcf_data.get('growth_mode', 5.0), dcf_data.get('growth_max', 8.0), dcf_data.get('perpetual_min', 1.5), dcf_data.get('perpetual_mode', 2.5), dcf_data.get('perpetual_max', 3.0), dcf_data.get('wacc_min', 7.0), dcf_data.get('wacc_mode', 8.0), dcf_data.get('wacc_max', 10.0))

    @app.callback([Output('modal-forecast-years-input', 'value'), Output('modal-forecast-eps-growth-input', 'value'), Output('modal-forecast-terminal-pe-input', 'value')], Input('forecast-assumptions-store', 'data'))
    def sync_forecast_modal_inputs(forecast_data):
        if not forecast_data: return 5, 10, 20
        return forecast_data.get('years', 5), forecast_data.get('growth', 10), forecast_data.get('pe', 20)

    # --- [START OF GRAPH REFACTOR - MODIFIED DCF SECTION] ---
    @app.callback(
        Output('analysis-pane-content', 'children'),
        [Input('analysis-tabs', 'active_tab'),
         Input('user-selections-store', 'data'),
         Input('dcf-assumptions-store', 'data'),
         Input('table-pane-content', 'children')] # <--- [เพิ่ม] Trigger เมื่อตารางโหลดเสร็จ
    )
    def render_graph_content(active_tab, store_data, dcf_data, table_content):
        store_data = store_data or {'tickers': [], 'indices': []}
        tickers = tuple(store_data.get('tickers', []))
        indices = tuple(store_data.get('indices', []))
        all_symbols = tuple(set(tickers + indices))

        if not all_symbols:
            return dbc.Alert("Please select items to display the chart", color="info", className="mt-3 text-center")

        try:
            if active_tab == "tab-performance":
                with server.app_context():
                    start_of_year = datetime(datetime.now().year, 1, 1).date()
                    query = db.session.query(FactDailyPrices.date, FactDailyPrices.ticker, FactDailyPrices.close) \
                                      .filter(FactDailyPrices.ticker.in_(all_symbols),
                                              FactDailyPrices.date >= start_of_year)
                    # --- [FIXED] ใช้ db.engine แทน db.session.bind ---
                    raw_data = pd.read_sql(query.statement, db.engine)

                if raw_data.empty: raise ValueError("No price data found in DB for YTD.")

                ytd_data = raw_data.pivot(index='date', columns='ticker', values='close').sort_index().ffill()
                if ytd_data.empty or len(ytd_data) < 2: raise ValueError("Not enough data after pivot.")

                ytd_perf = (ytd_data / ytd_data.iloc[0]) - 1
                ytd_perf = ytd_perf.rename(columns=INDEX_TICKER_TO_NAME)
                fig = px.line(ytd_perf, title='YTD Performance Comparison', color_discrete_map=COLOR_DISCRETE_MAP)
                fig.update_layout(yaxis_tickformat=".2%", legend_title_text='Symbol')
                return dbc.Card(dbc.CardBody(dcc.Graph(figure=fig)))

            if active_tab == "tab-drawdown":
                with server.app_context():
                    one_year_ago = datetime.utcnow().date() - timedelta(days=365)
                    query = db.session.query(FactDailyPrices.date, FactDailyPrices.ticker, FactDailyPrices.close) \
                                      .filter(FactDailyPrices.ticker.in_(all_symbols),
                                              FactDailyPrices.date >= one_year_ago)
                    # --- [FIXED] ใช้ db.engine แทน db.session.bind ---
                    raw_data = pd.read_sql(query.statement, db.engine)

                if raw_data.empty: raise ValueError("No price data found in DB for 1-Year Drawdown.")

                prices = raw_data.pivot(index='date', columns='ticker', values='close').sort_index().ffill()
                if prices.empty: raise ValueError("Not enough data after pivot.")

                rolling_max = prices.cummax()
                drawdown_data = (prices / rolling_max) - 1
                drawdown_data = drawdown_data.rename(columns=INDEX_TICKER_TO_NAME)

                fig = px.line(drawdown_data, title='1-Year Drawdown Comparison', color_discrete_map=COLOR_DISCRETE_MAP)
                fig.update_layout(yaxis_tickformat=".2%", legend_title_text='Symbol')
                return dbc.Card(dbc.CardBody(dcc.Graph(figure=fig)))

            if active_tab == "tab-scatter":
                if not tickers: return dbc.Alert("Please select stocks to display the chart.", color="info", className="mt-3 text-center")

                with server.app_context():
                    # Query ข้อมูลล่าสุดจาก FactCompanySummary
                    latest_date_sq = db.session.query(
                        FactCompanySummary.ticker,
                        func.max(FactCompanySummary.date_updated).label('max_date')
                    ).filter(FactCompanySummary.ticker.in_(tickers)).group_by(FactCompanySummary.ticker).subquery()

                    query = db.session.query(
                        FactCompanySummary.ticker,
                        FactCompanySummary.ev_ebitda,
                        FactCompanySummary.ebitda_margin
                    ).join(
                        latest_date_sq,
                        (FactCompanySummary.ticker == latest_date_sq.c.ticker) &
                        (FactCompanySummary.date_updated == latest_date_sq.c.max_date)
                    )
                    # --- [FIXED] ใช้ db.engine แทน db.session.bind ---
                    df_scatter = pd.read_sql(query.statement, db.engine)
                    df_scatter.rename(columns={'ticker': 'Ticker', 'ev_ebitda': 'EV/EBITDA', 'ebitda_margin': 'EBITDA Margin'}, inplace=True)

                if df_scatter.empty: return dbc.Alert("Could not fetch scatter data from DB.", color="warning")

                df_scatter = df_scatter.dropna()
                if df_scatter.empty: return dbc.Alert("No valid scatter data points to plot.", color="warning")

                fig = px.scatter(df_scatter, x="EBITDA Margin", y="EV/EBITDA", text="Ticker", title="Valuation vs. Quality Analysis")
                fig.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
                fig.update_layout(xaxis_tickformat=".2%", yaxis_title="EV / EBITDA (Valuation)", xaxis_title="EBITDA Margin (Quality)")
                x_avg, y_avg = df_scatter["EBITDA Margin"].mean(), df_scatter["EV/EBITDA"].mean()
                fig.add_vline(x=x_avg, line_width=1, line_dash="dash", line_color="grey"); fig.add_hline(y=y_avg, line_width=1, line_dash="dash", line_color="grey")
                return dbc.Card(dbc.CardBody(dcc.Graph(figure=fig)))

            if active_tab == "tab-dcf":
                # (ส่วนนี้ถูกปรับปรุงให้ดึงข้อมูลฐานจาก DB ก่อนนำไปคำนวณ Monte Carlo Simulation)
                if not tickers: return dbc.Alert("Please select stocks for DCF simulation.", color="info", className="mt-3 text-center")
                if not dcf_data: return dbc.Alert("Please set simulation assumptions using the gear icon.", color="info", className="mt-3 text-center")

                all_results = []
                for ticker in tickers:
                    # ฟังก์ชันนี้ถูกปรับปรุงให้ดึงข้อมูล TTM จาก DB ภายใน
                    result = calculate_monte_carlo_dcf(
                        ticker=ticker,
                        n_simulations=dcf_data.get('simulations', 10000),
                        growth_min=dcf_data.get('growth_min', 3.0),
                        growth_mode=dcf_data.get('growth_mode', 5.0),
                        growth_max=dcf_data.get('growth_max', 8.0),
                        perpetual_min=dcf_data.get('perpetual_min', 1.5),
                        perpetual_mode=dcf_data.get('perpetual_mode', 2.5),
                        perpetual_max=dcf_data.get('perpetual_max', 3.0),
                        wacc_min=dcf_data.get('wacc_min', 7.0),
                        wacc_mode=dcf_data.get('wacc_mode', 8.0),
                        wacc_max=dcf_data.get('wacc_max', 10.0)
                    )

                    if 'error' not in result:
                        result['Ticker'] = ticker
                        all_results.append(result)
                    else:
                         # แสดง Error ที่ละเอียดขึ้น
                         logging.warning(f"DCF simulation failed for {ticker}: {result['error']}")
                         # แสดง Alert ที่เฉพาะเจาะจงมากขึ้น แต่ไม่หยุดการทำงานทั้งหมด
                         # return dbc.Alert(f"DCF simulation for {ticker} failed: {result['error']}", color="danger", className="mt-3")
                         # แทนที่จะ return เราอาจจะเก็บ error message ไว้แสดงทีหลัง หรือแค่ log
                         # ในที่นี้เลือกที่จะ log และดำเนินการต่อ

                if not all_results:
                    # ถ้า *ทุกตัว* ล้มเหลว ค่อยแสดง Alert
                    return dbc.Alert("Could not run simulation for any selected stocks. Check logs for details (e.g., missing financial data).", color="danger")

                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
                for res in all_results: fig.add_trace(go.Histogram(x=res['simulated_values'], name=res['Ticker'], opacity=0.6, nbinsx=100), row=1, col=1)
                mos_data = [{'Ticker': r['Ticker'], 'current_price': r['current_price'], 'intrinsic_value': r['mean']} for r in all_results]; df_mos = pd.DataFrame(mos_data)
                fig.add_trace(go.Scatter(x=df_mos['current_price'], y=df_mos['Ticker'], mode='markers', marker=dict(color='royalblue', size=10), name='Current Price'), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_mos['intrinsic_value'], y=df_mos['Ticker'], mode='markers', marker=dict(color='darkorange', size=10, symbol='diamond'), name='Mean Intrinsic Value'), row=2, col=1)
                for i, row in df_mos.iterrows(): fig.add_shape(type='line', x0=row['current_price'], y0=row['Ticker'], x1=row['intrinsic_value'], y1=row['Ticker'], line=dict(color='limegreen' if row['intrinsic_value'] > row['current_price'] else 'tomato', width=3), row=2, col=1)
                fig.update_layout(title_text='Monte Carlo DCF Analysis', barmode='overlay', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                fig.update_yaxes(title_text="Frequency", row=1, col=1); fig.update_yaxes(title_text="Ticker", row=2, col=1); fig.update_xaxes(title_text="Share Price ($)", row=2, col=1)
                return dbc.Card(dbc.CardBody(dcc.Graph(figure=fig)))

            return html.P("This is an empty tab!")

        except Exception as e:
            tab_name = TABS_CONFIG.get(active_tab, {}).get('tab_name', 'Graph')
            logging.error(f"Error rendering graph content for tab {tab_name} ({active_tab}): {e}", exc_info=True)
            return dbc.Alert(f"An error occurred while rendering '{tab_name}': {type(e).__name__} - {e}", color="danger")
    # --- [END OF GRAPH REFACTOR - MODIFIED DCF SECTION] ---


    # --- [START OF TABLE REFACTOR - เหมือนเดิม] ---
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

            tickers = tuple(store_data.get('tickers')) # Tickers ที่ User เลือก

            # --- [Query from Warehouse with Join] ---
            df_full = pd.DataFrame()
            with server.app_context():
                latest_date_sq = db.session.query(
                    FactCompanySummary.ticker,
                    func.max(FactCompanySummary.date_updated).label('max_date')
                ).group_by(FactCompanySummary.ticker).subquery()

                # Query ข้อมูลทั้งหมดที่ FactCompanySummary มี
                query = db.session.query(
                    FactCompanySummary,
                    DimCompany.logo_url
                ).join(
                    latest_date_sq,
                    (FactCompanySummary.ticker == latest_date_sq.c.ticker) &
                    (FactCompanySummary.date_updated == latest_date_sq.c.max_date)
                ).join(
                    DimCompany,
                    FactCompanySummary.ticker == DimCompany.ticker
                ).filter(
                    FactCompanySummary.ticker.in_(tickers)
                )

                results = query.all()

                if results:
                    data_list = []
                    for summary_obj, logo_url_val in results:
                        # ดึงข้อมูลทั้งหมดจาก object
                        data_dict = {c.name: getattr(summary_obj, c.name, None) for c in summary_obj.__table__.columns}
                        data_dict['logo_url'] = logo_url_val
                        # เพิ่มคอลัมน์ดิบสำหรับ Market Cap เพื่อใช้ format ทีหลัง
                        data_dict['market_cap_raw'] = data_dict.get('market_cap')
                        data_list.append(data_dict)
                    df_full = pd.DataFrame(data_list)
                else:
                     # Fallback: ถ้า Query.all() ไม่ได้ผลลัพธ์
                     # ลองใช้ pd.read_sql กับ query.statement โดยตรง
                     # (เพื่อรองรับการใช้งานที่ db.session.query อาจไม่ได้ return tuple ที่คาดหวัง)
                     try:
                        # --- [FIXED] ใช้ db.engine แทน db.session.bind ---
                        df_full = pd.read_sql(query.statement, db.engine)
                        # ต้องตั้งชื่อคอลัมน์เองตามที่ Query เลือกไว้
                        # (FactCompanySummary.* + DimCompany.logo_url)
                        summary_cols = [c.name for c in FactCompanySummary.__table__.columns]
                        new_cols = summary_cols + ['logo_url']
                        if not df_full.empty:
                            df_full.columns = new_cols
                            df_full['market_cap_raw'] = df_full['market_cap']
                     except Exception as sql_err:
                         logging.warning(f"Failed direct read_sql fallback: {sql_err}")


            # --- [END QUERY] ---

            if df_full.empty:
                return dbc.Alert(f"No summary data found in the warehouse for: {', '.join(tickers)}. Please wait for the next ETL run or check ETL logs.", color="warning", className="mt-3 text-center"), [], None

            # --- Apply Scoring (ใช้ข้อมูลจาก DB ที่เป็น snake_case) ---
            df_full = apply_custom_scoring(df_full) # เพิ่ม 'Company Size', etc.

            # --- Rename columns from snake_case (DB) to display format ---
            column_mapping = {
                'ticker': 'Ticker', 'price': 'Price',
                'market_cap': 'Market Cap', # จะถูก format ทีหลัง
                'beta': 'Beta', 'pe_ratio': 'P/E', 'pb_ratio': 'P/B', 'ev_ebitda': 'EV/EBITDA',
                'revenue_growth_yoy': 'Revenue Growth (YoY)', 'revenue_cagr_3y': 'Revenue CAGR (3Y)',
                'net_income_growth_yoy': 'Net Income Growth (YoY)',
                'roe': 'ROE', 'de_ratio': 'D/E Ratio',
                'operating_margin': 'Operating Margin', 'cash_conversion': 'Cash Conversion',
                'logo_url': 'logo_url', # เก็บไว้ใช้ใน _prepare_display_dataframe
                'trailing_eps': 'Trailing EPS', # เพิ่มคอลัมน์ใหม่
                'ebitda_margin': 'EBITDA Margin', # เพิ่มคอลัมน์ใหม่
                'peer_cluster_id': 'peer_cluster_id' # <<< [เพิ่ม] เก็บ cluster ID ไว้ด้วย
            }
            df_full.rename(columns={k: v for k, v in column_mapping.items() if k in df_full.columns}, inplace=True)


            # --- [Handle Forecast Tab (คำนวณจากข้อมูลใน DB)] ---
            if active_tab == 'tab-forecast':
                forecast_years, eps_growth, terminal_pe = forecast_data.get('years'), forecast_data.get('growth'), forecast_data.get('pe')
                if all(v is not None for v in [forecast_years, eps_growth, terminal_pe]):

                    df_full['Trailing EPS'] = pd.to_numeric(df_full.get('Trailing EPS'), errors='coerce')
                    df_full['Price'] = pd.to_numeric(df_full.get('Price'), errors='coerce')

                    eps_growth_decimal = eps_growth / 100.0

                    def calc_target(row):
                        if pd.isna(row['Trailing EPS']) or row['Trailing EPS'] <= 0 or pd.isna(row['Price']) or row['Price'] <= 0:
                            return pd.NA, pd.NA, pd.NA

                        try:
                            future_eps = row['Trailing EPS'] * ((1 + eps_growth_decimal) ** forecast_years)
                            target_price = future_eps * terminal_pe
                            target_upside = (target_price / row['Price']) - 1
                            irr = ((target_price / row['Price']) ** (1 / forecast_years)) - 1
                            return target_price, target_upside, irr
                        except Exception:
                            return pd.NA, pd.NA, pd.NA

                    df_full[['Target Price', 'Target Upside', 'IRR %']] = df_full.apply(calc_target, axis=1, result_type='expand')
                else:
                    logging.warning("Forecast assumptions incomplete, skipping forecast calculation.")
                    for col in ["Target Price", "Target Upside", "IRR %"]:
                         if col not in df_full.columns: df_full[col] = pd.NA

            # --- Get Tab Config & Sort ---
            config = TABS_CONFIG[active_tab]
            if sort_by_column and sort_by_column in df_full.columns:
                ascending = not config['higher_is_better'].get(sort_by_column, True)

                # ใช้คอลัมน์ดิบ (ถ้ามี) สำหรับการเรียงลำดับที่เป็นตัวเลข
                # (แปลง 'Target Upside' -> 'target_upside')
                sort_col_raw = sort_by_column.lower().replace(' ', '_').replace('/', '_').replace('-', '_').replace('(', '').replace(')', '')
                sort_col_to_use = sort_col_raw if sort_col_raw in df_full.columns else sort_by_column

                try:
                    df_full[sort_col_to_use] = pd.to_numeric(df_full[sort_col_to_use], errors='coerce')
                    df_full.sort_values(by=sort_col_to_use, ascending=ascending, na_position='last', inplace=True)
                except Exception as sort_err:
                    logging.warning(f"Could not sort numerically by {sort_col_to_use} (derived from {sort_by_column}): {sort_err}. Trying string sort.")
                    df_full.sort_values(by=sort_by_column, ascending=ascending, na_position='last', key=lambda col: col.astype(str), inplace=True)

            # --- Prepare Display DataFrame (Format Ticker, Market Cap) ---
            df_display = _prepare_display_dataframe(df_full) # ใช้ market_cap_raw ที่เพิ่มไว้

            # --- Generate Table Components ---
            columns_for_tab = _generate_datatable_columns(config)
            style_data_conditional, style_cell_conditional = _generate_datatable_style_conditionals(config)
            dropdown_options = [{'label': col, 'value': col} for col in config["columns"] if col not in ['Ticker', 'Company Size', 'Volatility Level', 'Valuation Model', 'Stock Profile', 'Market Cap']]

            # --- Select data and columns for the specific tab ---
            columns_to_show_ids = [c['id'] for c in columns_for_tab]
            final_columns_present = [col for col in columns_to_show_ids if col in df_display.columns]
            missing_cols = set(columns_to_show_ids) - set(final_columns_present)
            if missing_cols:
                 logging.warning(f"Tab '{active_tab}': Final display missing columns: {missing_cols}")
                 columns_for_tab = [c for c in columns_for_tab if c['id'] in final_columns_present]
                 if not columns_for_tab:
                     return dbc.Alert(f"No data columns available to display for tab '{active_tab}'.", color="danger"), [], None

            data = df_display[final_columns_present].to_dict('records')

            return dash_table.DataTable(
                id='interactive-datatable', data=data, columns=columns_for_tab,
                style_data_conditional=style_data_conditional, style_cell_conditional=style_cell_conditional,
                row_selectable=False, cell_selectable=False,
                style_header={'border': '0px', 'backgroundColor': 'transparent', 'fontWeight': '600', 'textTransform': 'uppercase', 'textAlign': 'right'},
                style_data={'border': '0px', 'backgroundColor': 'transparent'},
                style_cell={'textAlign': 'right', 'padding': '14px', 'verticalAlign': 'middle'},
                style_header_conditional=[{'if': {'column_id': 'Ticker'}, 'textAlign': 'left'}],
                markdown_options={"html": True}
            ), dropdown_options, sort_by_column

        except Exception as e:
            tab_name = TABS_CONFIG.get(active_tab, {}).get('tab_name', 'Table')
            logging.error(f"Error rendering table content for tab {tab_name} ({active_tab}): {e}", exc_info=True)
            # แสดง Error ที่เจาะจงมากขึ้น ถ้าเป็นไปได้
            error_msg = f"An unexpected error occurred while loading table data: {type(e).__name__} - {e}"
            return dbc.Alert(error_msg, color="danger", className="mt-3"), [], None
    # --- [END OF TABLE REFACTOR] ---


    # ==================================================================
    # --- [START] NEW CALLBACKS FOR SMART PEER FINDER ---
    # ==================================================================

    @app.callback(
        Output('peer-reference-stock-dropdown', 'options'),
        Input('user-selections-store', 'data')
    )
    def update_peer_reference_options(store_data):
        """Populate the reference stock dropdown with currently selected tickers."""
        tickers = store_data.get('tickers', []) if store_data else []
        if not tickers:
            return [{'label': 'Select stocks first', 'value': '', 'disabled': True}]
        return [{'label': t, 'value': t} for t in sorted(tickers)]

    @app.callback(
        Output('peer-select-dropdown', 'options'),
        Output('peer-select-dropdown', 'value'), # Clear selection when reference changes
        Output('peer-finder-status', 'children'),
        Input('peer-reference-stock-dropdown', 'value'),
        State('user-selections-store', 'data')
    )
    def update_peer_select_options(reference_ticker, store_data):
        """Find peers for the selected reference stock and update the peer selection dropdown."""
        if not reference_ticker:
            return [], [], "" # No reference selected, clear options, value, and status

        current_tickers = set(store_data.get('tickers', [])) if store_data else set()
        peer_options = []
        status_message = ""

        try:
            with server.app_context():
                latest_date = db.session.query(func.max(FactCompanySummary.date_updated)).scalar()
                if not latest_date:
                    status_message = "Error: Could not find latest data date."
                    return [], [], status_message

                # 1. Find the cluster ID of the reference ticker
                ref_cluster_id_result = db.session.query(FactCompanySummary.peer_cluster_id) \
                                                  .filter(FactCompanySummary.date_updated == latest_date,
                                                          FactCompanySummary.ticker == reference_ticker) \
                                                  .first()

                if ref_cluster_id_result and ref_cluster_id_result[0] is not None:
                    ref_cluster_id = ref_cluster_id_result[0]

                    # 2. Find all tickers in the same cluster (excluding the reference and already selected)
                    query = db.session.query(FactCompanySummary.ticker) \
                                      .filter(FactCompanySummary.date_updated == latest_date,
                                              FactCompanySummary.peer_cluster_id == ref_cluster_id,
                                              FactCompanySummary.ticker != reference_ticker) \
                                      .order_by(FactCompanySummary.ticker)
                    all_peers_in_cluster = [item[0] for item in query.all()]

                    # 3. Filter out peers already in the user's selection
                    peers_to_show = [p for p in all_peers_in_cluster if p not in current_tickers]

                    if peers_to_show:
                        peer_options = [{'label': p, 'value': p} for p in peers_to_show]
                        status_message = f"Found {len(peers_to_show)} potential peers for {reference_ticker}."
                    else:
                        status_message = f"No *new* peers found for {reference_ticker} in Cluster {ref_cluster_id}."
                else:
                    status_message = f"Peer cluster data not available for {reference_ticker}."

        except Exception as e:
            logging.error(f"Error finding peers for {reference_ticker}: {e}", exc_info=True)
            status_message = f"Error finding peers for {reference_ticker}."

        return peer_options, [], status_message # Return options, clear value, set status

    @app.callback(
        Output('user-selections-store', 'data', allow_duplicate=True),
        Output('peer-reference-stock-dropdown', 'value', allow_duplicate=True), # Clear reference dropdown
        Output('peer-select-dropdown', 'options', allow_duplicate=True),      # Clear peer options
        Output('peer-select-dropdown', 'value', allow_duplicate=True),        # Clear peer selection
        Output('peer-finder-status', 'children', allow_duplicate=True),        # Clear status
        Input('add-peer-button', 'n_clicks'),
        State('peer-select-dropdown', 'value'), # Peers selected by the user
        State('user-selections-store', 'data'),
        prevent_initial_call=True
    )
    def add_selected_peers_to_store(n_clicks, selected_peers, store_data):
        """Adds the selected peers from the dropdown to the main user selection store."""
        if not n_clicks or not selected_peers:
            # If button not clicked or no peers selected, do nothing
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        store_data = store_data or {'tickers': [], 'indices': []}
        current_tickers = store_data.get('tickers', [])
        updated = False

        for peer in selected_peers:
            if peer not in current_tickers:
                current_tickers.append(peer)
                updated = True

        if updated:
            store_data['tickers'] = current_tickers # Update the list in the store data
            if current_user.is_authenticated:
                with server.app_context():
                    # Save the updated list (including newly added peers) to DB
                    save_selections_to_db(current_user.id, current_tickers, 'stock')
            # Return updated store data and clear the peer finder UI elements
            return store_data, None, [], [], ""
        else:
            # No new peers were actually added (e.g., they were already there somehow)
            # Just clear the peer finder UI
            return dash.no_update, None, [], [], ""

    # ==================================================================
    # --- [END] NEW CALLBACKS FOR SMART PEER FINDER ---
    # ==================================================================

# <<< End of register_callbacks function definition
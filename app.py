# app.py (Final Version with Scrollbar and Formula Fixes)

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
import os

from flask import Flask, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, logout_user, current_user

# Import components from other files
from auth import create_login_modal, create_register_layout, register_auth_callbacks
from data_handler import (
    calculate_drawdown, get_competitor_data, get_scatter_data,
    calculate_dcf_intrinsic_value,
    calculate_exit_multiple_valuation
)
from pages import deep_dive
from constants import (
    SECTORS, INDEX_TICKER_TO_NAME, SECTOR_TO_INDEX_MAPPING, COLOR_DISCRETE_MAP
)
from config import Config

# ==================================================================
# 1. App Initialization
# ==================================================================
server = Flask(__name__)
server.config.from_object(Config)

instance_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'instance')
if not os.path.exists(instance_path):
    os.makedirs(instance_path)

db = SQLAlchemy(server)
login_manager = LoginManager()
login_manager.init_app(server)
app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.LUX, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    assets_folder='assets'
)

# ==================================================================
# 2. Database Models & User Loader
# ==================================================================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class UserSelection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    symbol_type = db.Column(db.String(10), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    with server.app_context():
        return db.session.get(User, int(user_id))

@server.route('/logout')
def logout():
    logout_user()
    return redirect('/', code=302)
    
# ==================================================================
# 3. Layout Definition
# ==================================================================

# --- Dictionary for Metric Definitions ---
METRIC_DEFINITIONS = {
    # Valuation Metrics
    "P/E": dcc.Markdown("""
    **P/E (Price-to-Earnings) Ratio**
    * **Definition:** A valuation ratio that compares a company's current share price to its per-share earnings.
    * **Formula:** $$
        P/E\\ Ratio = \\frac{\\text{Market Price per Share}}{\\text{Earnings per Share (EPS)}}
        $$
    * **Formula Components:**
        * `Market Price per Share`: The current market price of a single share.
        * `Earnings per Share (EPS)`: The company's total profit allocated to each outstanding share of common stock.
    * **Interpretation:**
        * A **high P/E** can suggest that a stock's price is high relative to earnings and is possibly overvalued. It may also indicate that investors are expecting high future growth.
        * A **low P/E** might indicate that a stock is undervalued or that the company is performing well compared to its past trends.
    """, mathjax=True),
    "P/B": dcc.Markdown("""
    **P/B (Price-to-Book) Ratio**
    * **Definition:** Compares a company's market capitalization to its book value. It indicates the value investors place on the company's net assets.
    * **Formula:** $$
        P/B\\ Ratio = \\frac{\\text{Market Price per Share}}{\\text{Book Value per Share}}
        $$
    * **Formula Components:**
        * `Market Price per Share`: The current market price of a single share.
        * `Book Value per Share`: The net asset value of a company divided by the number of shares outstanding, calculated as (Total Assets - Intangible Assets - Liabilities).
    * **Interpretation:**
        * A **P/B ratio under 1.0** is often considered a potential sign of an undervalued stock.
        * It is particularly useful for valuing companies with significant tangible assets (e.g., banks, industrials) and less useful for service-based companies.
    """, mathjax=True),
    "EV/EBITDA": dcc.Markdown("""
    **EV/EBITDA (Enterprise Value-to-EBITDA) Ratio**
    * **Definition:** A ratio used to compare the total value of a company to its cash earnings less non-cash expenses. It's often considered more comprehensive than P/E as it accounts for debt.
    * **Formula:** $$
        EV/EBITDA = \\frac{\\text{Enterprise Value}}{\\text{EBITDA}}
        $$
    * **Formula Components:**
        * `Enterprise Value (EV)`: Market Capitalization + Total Debt - Cash and Cash Equivalents.
        * `EBITDA`: Earnings Before Interest, Taxes, Depreciation, and Amortization.
    * **Interpretation:**
        * A **low EV/EBITDA ratio** suggests that a company might be undervalued.
        * This metric is useful for comparing companies across different industries, especially those with differences in capital structure and tax rates.
    """, mathjax=True),

    # Growth Metrics
    "Revenue Growth (YoY)": dcc.Markdown("""
    **Revenue Growth (Year-over-Year)**
    * **Definition:** Measures the percentage increase in a company's revenue over the most recent twelve-month period compared to the prior twelve-month period.
    * **Formula:** $$
        YoY\\ Growth = \\left( \\frac{\\text{Current Period Revenue}}{\\text{Prior Period Revenue}} - 1 \\right) x 100
        $$
    * **Formula Components:**
        * `Current Period Revenue`: Revenue from the most recent 12-month period.
        * `Prior Period Revenue`: Revenue from the 12-month period before the current one.
    * **Interpretation:** Indicates the pace at which a company's sales are growing. Consistently high growth is a positive sign, but it's important to understand the drivers behind it.
    """, mathjax=True),
    "Revenue CAGR (3Y)": dcc.Markdown("""
    **Revenue CAGR (3-Year)**
    * **Definition:** The Compound Annual Growth Rate of revenue over a three-year period. It provides a smoothed, annualized growth rate that irons out volatility.
    * **Formula:** $$
        CAGR = \\left( \\left( \\frac{\\text{Ending Value}}{\\text{Beginning Value}} \\right)^{\\frac{1}{\\text{No. of Years}}} - 1 \\right) x 100
        $$
    * **Formula Components:**
        * `Ending Value`: Revenue from the final year in the period.
        * `Beginning Value`: Revenue from the starting year in the period.
        * `No. of Years`: The total number of years in the period (e.g., 3).
    * **Interpretation:** A more stable indicator of long-term growth trends compared to a single-year growth rate.
    """, mathjax=True),
    "Net Income Growth (YoY)": dcc.Markdown("""
    **Net Income Growth (Year-over-Year)**
    * **Definition:** Measures the percentage increase in a company's net profit (after all expenses and taxes) over the past year.
    * **Formula:** $$
        YoY\\ Growth = \\left( \\frac{\\text{Current Period Net Income}}{\\text{Prior Period Net Income}} - 1 \\right) x 100
        $$
    * **Formula Components:**
        * `Current Period Net Income`: Net Income from the most recent 12-month period.
        * `Prior Period Net Income`: Net Income from the 12-month period before the current one.
    * **Interpretation:** Shows how effectively a company is translating revenue growth into actual profit for shareholders. It's a critical measure of profitability improvement.
    """, mathjax=True),

    # Fundamentals Metrics
    "Operating Margin": dcc.Markdown("""
    **Operating Margin**
    * **Definition:** Measures how much profit a company makes on a dollar of sales, after paying for variable costs of production but before paying interest or tax.
    * **Formula:** $$
        Operating\\ Margin = \\frac{\\text{Operating Income}}{\\text{Revenue}} x 100
        $$
    * **Formula Components:**
        * `Operating Income`: The profit realized from a business's own, core operations.
        * `Revenue`: The total amount of income generated by the sale of goods or services.
    * **Interpretation:** A higher operating margin indicates greater efficiency in the company's core business operations. It reflects the profitability of the business before the effects of financing and taxes.
    """, mathjax=True),
    "ROE": dcc.Markdown("""
    **ROE (Return on Equity)**
    * **Definition:** A measure of financial performance calculated by dividing net income by shareholders' equity.
    * **Formula:** $$
        ROE = \\frac{\\text{Net Income}}{\\text{Average Shareholder's Equity}} x 100
        $$
    * **Formula Components:**
        * `Net Income`: The company's profit after all expenses, including taxes and interest, have been deducted.
        * `Average Shareholder's Equity`: The average value of shareholder's equity over a period (usually the beginning and ending equity divided by 2).
    * **Interpretation:** ROE is considered a gauge of a corporation's profitability and how efficiently it generates profits. A consistently high ROE can be a sign of a strong competitive advantage (a "moat").
    """, mathjax=True),
    "D/E Ratio": dcc.Markdown("""
    **D/E (Debt-to-Equity) Ratio**
    * **Definition:** A ratio used to evaluate a company's financial leverage. It is a measure of the degree to which a company is financing its operations through debt versus wholly-owned funds.
    * **Formula:** $$
        D/E\\ Ratio = \\frac{\\text{Total Debt}}{\\text{Total Shareholder's Equity}}
        $$
    * **Formula Components:**
        * `Total Debt`: The sum of all short-term and long-term liabilities.
        * `Total Shareholder's Equity`: The corporation's owners' residual claim on assets after debts have been paid.
    * **Interpretation:**
        * A **high D/E ratio** generally means that a company has been aggressive in financing its growth with debt. This can result in volatile earnings because of the additional interest expense.
        * A **low D/E ratio** may indicate a more financially stable, conservative company. What is considered "high" or "low" varies by industry.
    """, mathjax=True),
    "Cash Conversion": dcc.Markdown("""
    **Cash Conversion**
    * **Definition:** Measures how efficiently a company converts its net income into operating cash flow.
    * **Formula:** $$
        Cash\\ Conversion = \\frac{\\text{Operating Cash Flow}}{\\text{Net Income}}
        $$
    * **Formula Components:**
        * `Operating Cash Flow (CFO)`: The cash generated from normal business operations.
        * `Net Income`: The company's profit after all expenses.
    * **Interpretation:**
        * A ratio **greater than 1.0** is generally considered strong, as it indicates the company is generating more cash than it reports in accounting profit.
        * A ratio **consistently below 1.0** could be a red flag, suggesting that reported earnings are not being backed by actual cash.
    """, mathjax=True),

    # Target/Forecast Metrics
    "Target Price": dcc.Markdown("""
    **Target Price**
    * **Definition:** The projected future price of a stock based on the assumptions entered in the 'Forecast Assumptions' modal.
    * **Formula:** $$
        Target\\ Price = (\\text{Current EPS} x (1 + \\text{EPS Growth})^{\\text{Years}}) x \\text{Terminal P/E}
        $$
    * **Formula Components:**
        * `Current EPS`: The company's earnings per share for the last twelve months.
        * `EPS Growth`: Your assumed annual growth rate for EPS.
        * `Years`: The number of years in your forecast period.
        * `Terminal P/E`: Your assumed P/E ratio for the company at the end of the forecast period.
    * **Interpretation:** This is an estimated future value, not a guarantee. Its accuracy is entirely dependent on the validity of the growth and valuation multiple assumptions.
    """, mathjax=True),
    "Target Upside": dcc.Markdown("""
    **Target Upside**
    * **Definition:** The potential percentage return an investor could achieve if the stock reaches its calculated Target Price from its current price.
    * **Formula:** $$
        Target\\ Upside = \\left( \\frac{\\text{Target Price}}{\\text{Current Price}} - 1 \\right) x 100
        $$
    * **Formula Components:**
        * `Target Price`: The estimated future stock price from your forecast.
        * `Current Price`: The current market stock price.
    * **Interpretation:** It quantifies the potential reward based on your forecast. A higher upside suggests a more attractive investment, assuming the forecast is accurate.
    """, mathjax=True),
    "IRR %": dcc.Markdown("""
    **IRR (Internal Rate of Return) %**
    * **Definition:** The projected compound annual growth rate (CAGR) of an investment if it moves from the current price to the target price over the forecast period.
    * **Formula:** $$
        IRR = \\left( \\left( \\frac{\\text{Target Price}}{\\text{Current Price}} \\right)^{\\frac{1}{\\text{Forecast Years}}} - 1 \\right) x 100
        $$
    * **Formula Components:**
        * `Target Price`: The estimated future stock price from your forecast.
        * `Current Price`: The current market stock price.
        * `Forecast Years`: The number of years in your forecast period.
    * **Interpretation:** IRR represents the annualized rate of return for this specific investment scenario. It's useful for comparing the potential returns of different investment opportunities over different time horizons.
    """, mathjax=True),
}


def create_navbar():
    if current_user.is_authenticated:
        login_button = dbc.Button("Logout", href="/logout", color="secondary", external_link=True)
    else:
        login_button = dbc.Button("Login", id="open-login-modal-button", color="primary")
    return dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    dbc.Row(
                        [
                            dbc.Col(html.Img(src="/assets/logo.png", height="45px")),
                            dbc.Col(dbc.NavbarBrand("FINANCIAL ANALYSIS DASHBOARD", className="ms-2 fw-bold")),
                        ],
                        align="center",
                        className="g-0",
                    ),
                    href="/",
                    style={"textDecoration": "none"},
                ),
                dbc.Stack(login_button, direction="horizontal", className="ms-auto")
            ],
            fluid=True
        ),
        color="dark",
        dark=True,
        className="py-2 fixed-top"
    )

def create_forecast_modal():
    """Creates the pop-up modal for forecast assumptions."""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("FORECAST ASSUMPTIONS")),
            dbc.ModalBody([
                dbc.Label("Forecast Period (Years):"),
                dbc.Input(id="modal-forecast-years-input", type="number", value=5, min=1, max=10, step=1, className="mb-3"),
                dbc.Label("Annual EPS Growth (%):"),
                dbc.Input(id="modal-forecast-eps-growth-input", type="number", value=10, step=1, className="mb-3"),
                dbc.Label("Terminal P/E Ratio:"),
                dbc.Input(id="modal-forecast-terminal-pe-input", type="number", value=20, min=1, step=1),
            ]),
            dbc.ModalFooter(
                dbc.Button("Apply Changes", id="apply-forecast-changes-btn", color="primary")
            ),
        ],
        id="forecast-assumptions-modal",
        is_open=False,
    )

def create_definitions_modal():
    """Creates the pop-up modal for context-aware metric definitions."""
    return dbc.Modal(
        [
            dbc.ModalHeader(id="definitions-modal-title"),
            dbc.ModalBody(id="definitions-modal-body"),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-definitions-modal-btn", className="ms-auto")
            ),
        ],
        id="definitions-modal",
        is_open=False,
        size="lg",
        scrollable=False,
    )

def build_layout():
    return html.Div([
        dcc.Store(id='user-selections-store', storage_type='session'),
        dcc.Store(id='forecast-assumptions-store', storage_type='session', data={'years': 5, 'growth': 10, 'pe': 20}),
        html.Div(id="navbar-container"),
        dbc.Container([
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.Label("Add Stocks to Analysis", className="fw-bold"),
                    dcc.Dropdown(
                        id='sector-dropdown',
                        options=[{'label': 'All Sectors', 'value': 'All'}] + [{'label': k, 'value': k} for k in SECTORS.keys()],
                        value='All',
                        clearable=False
                    ),
                    dcc.Dropdown(id='ticker-select-dropdown', className="mt-2", placeholder="Select one or more tickers...", multi=True),
                    dbc.Button([html.I(className="bi bi-plus-circle-fill me-2"), "Add Stock(s)"], id="add-ticker-button", n_clicks=0, className="mt-2 w-100"),
                    html.Hr(),
                    html.Label("Add Benchmarks to Compare", className="fw-bold"),
                    dcc.Dropdown(id='index-select-dropdown', placeholder="Select one or more indices...", multi=True),
                    dbc.Button([html.I(className="bi bi-plus-circle-fill me-2"), "Add Benchmark(s)"], id="add-index-button", n_clicks=0, className="mt-2 w-100"),
                    html.Hr(className="my-4"),
                    html.Div(id='ticker-summary-display'),
                    html.Div(id='index-summary-display', className="mt-2"),
                ])), width=12, md=3, className="sidebar-fixed"),
                dbc.Col([
                    html.Div(className="custom-tabs-container", children=[
                        dbc.Tabs(id="analysis-tabs", active_tab="tab-performance", children=[
                            dbc.Tab(label="PERFORMANCE (YTD)", tab_id="tab-performance"),
                            dbc.Tab(label="DRAWDOWN (RISK)", tab_id="tab-drawdown"),
                            dbc.Tab(label="VALUATION VS. QUALITY", tab_id="tab-scatter"),
                            dbc.Tab(label="MARGIN OF SAFETY (DCF)", tab_id="tab-dcf"),
                        ])
                    ]),
                    dcc.Loading(html.Div(id='analysis-pane-content', className="mt-3")),
                    html.Hr(className="my-5"),
                    
                    dbc.Row([
                        dbc.Col(
                            html.Div(className="custom-tabs-container", children=[
                                dbc.Tabs(id="table-tabs", active_tab="tab-valuation", children=[
                                    dbc.Tab(label="VALUATION", tab_id="tab-valuation"),
                                    dbc.Tab(label="GROWTH", tab_id="tab-growth"),
                                    dbc.Tab(label="FUNDAMENTALS", tab_id="tab-fundamentals"),
                                    dbc.Tab(label="TARGET", tab_id="tab-forecast"), # <-- RENAMED
                                ])
                            ]),
                        ),
                        dbc.Col(
                            dbc.Stack(
                                [
                                    dbc.Button(html.I(className="bi bi-info-circle-fill"), id="open-definitions-modal-btn", color="secondary", outline=True),
                                    dbc.Button(html.I(className="bi bi-gear-fill"), id="open-forecast-modal-btn", color="secondary", outline=True),
                                    dcc.Dropdown(id='sort-by-dropdown', placeholder="Sort by", style={'width': '180px'})
                                ],
                                direction="horizontal",
                                gap=2
                            ),
                            width="auto"
                        )
                    ], justify="between", align="center", className="mt-3 g-2"),

                    dcc.Loading(html.Div(id="table-pane-content", className="mt-2"))
                ], width=12, md=9, className="content-offset"),
            ], className="g-4")
        ], fluid=True, className="p-4 main-content-container"),
        create_login_modal(),
        create_forecast_modal(),
        create_definitions_modal()
    ])

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# ==================================================================
# 4. Routing and User Selection Callbacks
# ==================================================================
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
            html.Span("No stocks selected.", className="text-muted fst-italic"),
            html.Small("Use the dropdowns above to start your analysis.", className="d-block text-muted mt-1")
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

# ==================================================================
# 4.5. Modal Management Callbacks
# ==================================================================
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
    [Input("open-definitions-modal-btn", "n_clicks"), Input("close-definitions-modal-btn", "n_clicks")],
    [State("definitions-modal", "is_open"), State("table-tabs", "active_tab")],
    prevent_initial_call=True
)
def toggle_definitions_modal(open_clicks, close_clicks, is_open, active_tab):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update

    triggered_id = ctx.triggered_id
    if triggered_id == "open-definitions-modal-btn":
        tab_name = TABS_CONFIG[active_tab].get('tab_name', active_tab.replace('tab-', '').title())
        title = f"{tab_name.upper()} METRIC DEFINITIONS"
        
        columns_in_tab = TABS_CONFIG[active_tab]['columns']
        
        body_content = []
        for col in columns_in_tab:
            if col in METRIC_DEFINITIONS:
                body_content.append(METRIC_DEFINITIONS[col])
                body_content.append(html.Hr())

        if not body_content:
            body_content = [html.P("No specific definitions for this tab.")]
        else:
             body_content.pop() # Remove last Hr

        return True, title, body_content

    if triggered_id == "close-definitions-modal-btn":
        return False, dash.no_update, dash.no_update

    return is_open, dash.no_update, dash.no_update

# ==================================================================
# 5. Callbacks for Main Dashboard Panes
# ==================================================================
@app.callback(Output('analysis-pane-content', 'children'), [Input('analysis-tabs', 'active_tab'), Input('user-selections-store', 'data')])
def render_graph_content(active_tab, store_data):
    store_data = store_data or {'tickers': [], 'indices': []}
    tickers = tuple(store_data.get('tickers', []))
    indices = tuple(store_data.get('indices', []))

    if active_tab == "tab-performance":
        all_symbols = tuple(set(tickers + indices))
        if not all_symbols: return dbc.Card(dbc.CardBody(html.P("Please select items to display the chart", className="text-center text-muted")))
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
        if not all_symbols: return dbc.Card(dbc.CardBody(html.P("Please select items to display the chart", className="text-center text-muted")))
        try:
            drawdown_data = calculate_drawdown(all_symbols, period="1y")
            if drawdown_data.empty: raise ValueError("Could not calculate drawdown data.")
            drawdown_data = drawdown_data.rename(columns=INDEX_TICKER_TO_NAME)
            fig = px.line(drawdown_data, title='1-Year Drawdown Comparison', color_discrete_map=COLOR_DISCRETE_MAP)
            fig.update_layout(yaxis_tickformat=".2%", legend_title_text='Symbol'); return dbc.Card(dbc.CardBody(dcc.Graph(figure=fig)))
        except Exception as e: return dbc.Alert(f"An error occurred while rendering 'Drawdown': {e}", color="danger")

    if active_tab == "tab-scatter":
        if not tickers: return dbc.Card(dbc.CardBody(html.P("Please select stocks to display the chart.", className="text-center text-muted")))
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
        if not tickers: return dbc.Card(dbc.CardBody(html.P("Please select stocks to display the chart.", className="text-center text-muted")))
        try:
            growth_rate = 0.05
            dcf_results = [calculate_dcf_intrinsic_value(t, growth_rate) for t in tickers]
            successful_results = [res for res in dcf_results if 'error' not in res]
            failed_results = [res for res in dcf_results if 'error' in res]

            output_components = []
            if failed_results:
                failed_tickers = ', '.join([res['Ticker'] for res in failed_results])
                output_components.append(dbc.Alert([html.I(className="bi bi-exclamation-triangle-fill me-2"), f"Could not calculate DCF for: {failed_tickers} (Missing data)"], color="warning", className="mb-3",dismissable=True))

            if successful_results:
                df_dcf = pd.DataFrame(successful_results)
                fig = go.Figure()
                for i, row in df_dcf.iterrows():
                    fig.add_shape(type='line', x0=row['current_price'], y0=row['Ticker'], x1=row['intrinsic_value'], y1=row['Ticker'], line=dict(color='limegreen' if row['intrinsic_value'] > row['current_price'] else 'tomato', width=3))
                fig.add_trace(go.Scatter(x=df_dcf['current_price'], y=df_dcf['Ticker'], mode='markers', marker=dict(color='royalblue', size=10), name='Current Price'))
                fig.add_trace(go.Scatter(x=df_dcf['intrinsic_value'], y=df_dcf['Ticker'], mode='markers', marker=dict(color='darkorange', size=10, symbol='diamond'), name='Intrinsic Value (DCF)'))
                fig.update_layout(title=f'Margin of Safety (DCF) with {growth_rate:.0%} Growth Forecast', xaxis_title='Share Price ($)', yaxis_title='Ticker', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                output_components.append(dbc.Card(dbc.CardBody(dcc.Graph(figure=fig))))

            if not output_components:
                return dbc.Alert("Could not process DCF for any selected stocks.", color="danger")

            return output_components
        except Exception as e: return dbc.Alert(f"An error occurred while rendering DCF chart: {e}", color="danger")

    return html.P("This is an empty tab!")

# ==================================================================
# 6. SCORING SYSTEM AND TABLE HELPERS
# ==================================================================
def apply_custom_scoring(df):
    """Applies custom scoring logic to the dataframe."""
    if df.empty:
        return df

    # 1. Company Size Model (based on Market Cap)
    bins = [0, 1e9, 10e10, 100e10, float('inf')]
    labels = ["Small Cap", "Mid Cap", "Large Cap", "Mega Cap"]
    df['Company Size'] = pd.cut(df['Market Cap'], bins=bins, labels=labels, right=False)

    # 2. Volatility Level Model (based on Beta)
    conditions_volatility = [df['Beta'].isnull(), df['Beta'] < 0.5, df['Beta'] <= 2, df['Beta'] > 2]
    choices_volatility = ["N/A", "Core", "Growth", "Hyper Growth"]
    df['Volatility Level'] = np.select(conditions_volatility, choices_volatility, default='N/A')

    # 3. Valuation Model (based on EV/EBITDA)
    conditions_valuation = [df['EV/EBITDA'].isnull(), df['EV/EBITDA'] < 10, df['EV/EBITDA'] <= 25, df['EV/EBITDA'] > 25]
    choices_valuation = ["N/A", "Cheap", "Fair Value", "Expensive"]
    df['Valuation Model'] = np.select(conditions_valuation, choices_valuation, default='N/A')

    # 4. Stock Profile (Categorization)
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

# --- Table Configuration and Styling ---
TABS_CONFIG = {
    "tab-valuation": { "columns": ["Ticker", "Market Cap", "Company Size", "Price", "P/E", "P/B", "EV/EBITDA"], "higher_is_better": {"P/E": False, "P/B": False, "EV/EBITDA": False}, "tab_name": "Valuation" },
    "tab-growth": { "columns": ["Ticker", "Revenue Growth (YoY)", "Revenue CAGR (3Y)", "Net Income Growth (YoY)"], "higher_is_better": {k: True for k in ["Revenue Growth (YoY)", "Revenue CAGR (3Y)", "Net Income Growth (YoY)"]}, "tab_name": "Growth" },
    "tab-fundamentals": { "columns": ["Ticker", "Operating Margin", "ROE", "D/E Ratio", "Cash Conversion"], "higher_is_better": {"Operating Margin": True, "ROE": True, "D/E Ratio": False, "Cash Conversion": True}, "tab_name": "Fundamentals" },
    "tab-forecast": {
        "columns": ["Ticker", "Target Price", "Target Upside", "IRR %", "Volatility Level", "Valuation Model", "Stock Profile"],
        "higher_is_better": {"Target Upside": True, "IRR %": True},
        "tab_name": "Target" # <-- RENAMED
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
            style_header={
                'border': '0px',
                'backgroundColor': 'transparent',
                'fontWeight': '600',
                'textTransform': 'uppercase',
                'textAlign': 'right'
            },
            style_data={'border': '0px', 'backgroundColor': 'transparent'},
            style_cell={
                'textAlign': 'right',
                'padding': '14px',
                'verticalAlign': 'middle'
            },
            style_header_conditional=[
                {'if': {'column_id': 'Ticker'}, 'textAlign': 'left'}
            ],
            markdown_options={"html": True}
        )
        return datatable, dropdown_options, sort_value

    except Exception as e:
        logging.error(f"Error rendering table content: {e}", exc_info=True)
        alert = dbc.Alert(f"An unexpected error occurred while building the table: {e}", color="danger", className="mt-3")
        return alert, [], None

# ==================================================================
# 7. Register Auth Callbacks & Run App
# ==================================================================
register_auth_callbacks(app, db, User)

if __name__ == '__main__':
    with server.app_context():
        db.create_all()
    app.run(debug=True)
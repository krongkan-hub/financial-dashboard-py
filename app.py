# app.py (Fully Corrected version with working Deep Dive Tabs)

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

from flask import Flask, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, logout_user, current_user

# Import components from other files
from auth import create_login_modal, create_register_layout, register_auth_callbacks
from data_handler import (
    calculate_drawdown, get_competitor_data, get_scatter_data,
    calculate_dcf_intrinsic_value, get_deep_dive_data
)
from pages import deep_dive

# ==================================================================
# 1. App Initialization & Constants
# ==================================================================
server = Flask(__name__)
server.config['SECRET_KEY'] = 'a-very-secret-key-that-you-should-change'
server.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
server.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(server)
login_manager = LoginManager()
login_manager.init_app(server)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.LUX, dbc.icons.BOOTSTRAP], suppress_callback_exceptions=True, assets_folder='assets')

# ... (Constants remain unchanged) ...
SECTORS = {
    'Technology': ['MSFT', 'NVDA', 'AVGO', 'CRM', 'AMD', 'QCOM', 'ORCL', 'ADBE', 'ACN', 'INTC', 'IBM', 'CSCO', 'AMAT', 'UBER', 'NOW', 'INTU', 'LRCX', 'APH', 'ADI', 'PANW', 'MU', 'SNPS', 'CDNS', 'KLAC', 'ROP', 'DELL', 'ANET', 'MCHP', 'FTNT', 'TEL', 'IT', 'CTSH', 'SNOW', 'MSI', 'PAYX', 'ADSK', 'KEYS', 'CRWD', 'GLW', 'HPE', 'HPQ', 'CDW', 'FICO', 'TER', 'STX', 'ANSS', 'TRMB', 'VRSN', 'ZBRA', 'ENET', 'TDY', 'WDC', 'JBL', 'EXC', 'UI', 'NTAP', 'SMCI', 'GPN', 'PTC', 'FFIV', 'FLT', 'GRMN', 'DXC', 'FIS', 'AKAM', 'KEYS', 'BR', 'MDB', 'DDOG', 'NET', 'TEAM', 'ZS', 'OKTA', 'PLTR', 'SHOP', 'SQ', 'ARM', 'WDAY', 'HUBS', 'ZM', 'EPAM', 'APP', 'AFRM', 'IOT', 'S', 'GEN', 'CHKP', 'DT', 'VEEV', 'FTV', 'TOST', 'DOCN', 'UIPATH', 'PATH', 'AI', 'Z', 'GTLB', 'RBLX', 'MSTR'],
    'Consumer Cyclical': ['IREN','AMZN', 'TSLA', 'HD', 'MCD', 'LOW', 'SBUX', 'BKNG', 'TJX', 'MAR', 'CMG', 'GM', 'F', 'ROST', 'HLT', 'YUM', 'ORLY', 'AZO', 'LVS', 'CPRT', 'NKE', 'CCL', 'RCL', 'DRI', 'GRMN', 'ULTA', 'LEN', 'DHI', 'TGT', 'EBAY', 'EXPE', 'POOL', 'WYN', 'PHM', 'KMX', 'BBY', 'WYNN', 'MGM', 'ETSY', 'LULU', 'TPR', 'CZR', 'NVR', 'DPZ', 'RL', 'VFC', 'WHR', 'HAS', 'MAT', 'MHK', 'APTV', 'GPC', 'LKQ', 'BWA', 'KDP', 'MNST', 'HOG', 'PENN', 'DKS', 'FL', 'BBWI', 'RH', 'WYND', 'CHS', 'AAP', 'LEG', 'GPS', 'JWN', 'M', 'KSS', 'TCOM', 'TRIP', 'SNA', 'SWK', 'PAG', 'AN', 'RUSHA', 'SAH', 'ABG', 'CWH', 'GNTX', 'LEA', 'VC', 'SON', 'TPX', 'WGO', 'THO'],
    'Financials': ['BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'AXP', 'SPGI', 'SCHW', 'CB', 'PNC', 'C', 'AON', 'MMC', 'USB', 'ICE', 'TROW', 'COF', 'MET', 'PRU', 'AIG', 'TRV', 'ALL', 'CME', 'DFS', 'PGR', 'WRB', 'MCO', 'PYPL', 'FISV', 'SQ', 'ACGL', 'RE', 'AMP', 'AJG', 'BEN', 'IVZ', 'L', 'NTRS', 'STT', 'TFC', 'ZION', 'FITB', 'KEY', 'HBAN', 'RF', 'CFG', 'CMA', 'MTB', 'BAC', 'GS', 'MS', 'JPM', 'C', 'WFC', 'USB', 'PNC', 'TFC', 'COF', 'BK', 'STT', 'NTRS', 'AMP', 'AXP', 'DFS', 'MA', 'V', 'PYPL', 'FISV', 'GPN', 'FLT', 'SQ', 'AFRM', 'HOOD', 'COIN', 'CBOE', 'NDAQ', 'MKTX', 'TRU', 'EFX', 'FDS', 'AFL', 'CINF', 'HIG', 'LNC', 'UNM'],
    'Energy': ['XOM', 'CVX', 'SHEL', 'COP', 'SLB', 'EOG', 'PXD', 'OXY', 'MPC', 'VLO', 'WMB', 'KMI', 'PSX', 'HAL', 'DVN', 'HES', 'OKE', 'BKR', 'FANG', 'TRGP', 'CTRA', 'MRO', 'APA', 'PBR', 'EQT', 'ENB', 'TRP', 'SU', 'CNQ', 'IMO', 'CVE', 'PBA', 'ET', 'EPD', 'PAA', 'MPLX', 'LNG', 'NOV', 'FTI', 'OII', 'HP', 'RIG', 'VET', 'MUR', 'NFG', 'AR', 'RRC', 'SWN', 'CHK', 'CNX', 'ARCH', 'BTU', 'CEIX', 'METC', 'WFRD', 'DK', 'PBF', 'CVI', 'USAC', 'NGL', 'SUN', 'GEL', 'CQP', 'SHLS', 'ENPH', 'SEDG', 'RUN', 'FSLR', 'SPWR', 'BE', 'PLUG', 'FCEL', 'BLDP'],
    'Healthcare': ['LLY', 'UNH', 'JNJ', 'MRK', 'ABBV', 'TMO', 'PFE', 'DHR', 'ABT', 'AMGN', 'MDT', 'GILD', 'ISRG', 'SYK', 'CVS', 'REGN', 'VRTX', 'CI', 'BSX', 'BDX', 'HCA', 'ELV', 'MCK', 'ZBH', 'HUM', 'IDXX', 'EW', 'BIIB', 'COR', 'LH', 'ALGN', 'A', 'DXCM', 'RMD', 'WST', 'TECH', 'IQV', 'HOLX', 'DGX', 'CNC', 'BAX', 'CAH', 'MOH', 'UHS', 'XRAY', 'INCY', 'SGEN', 'JAZZ', 'BGNE', 'MRNA', 'BNTX', 'NVAX', 'VEEV', 'TDOC', 'CERT', 'DVA', 'PODD', 'TFX', 'COO', 'STE', 'SRPT'],
    'Industrials': ['CAT', 'UPS', 'GE', 'UNP', 'BA', 'RTX', 'HON', 'DE', 'LMT', 'ADP', 'ETN', 'WM', 'CSX', 'NSC', 'PCAR', 'ITW', 'GD', 'TDG', 'EMR', 'ROP', 'PAYX', 'CMI', 'PH', 'VRSK', 'FAST', 'OTIS', 'CARR', 'GWW', 'JCI', 'RSG', 'IR', 'AME', 'LDOS', 'XYL', 'WAB', 'EXPD', 'JBHT', 'ODFL', 'CHRW', 'DAL', 'UAL', 'AAL', 'LUV', 'FDX', 'SWK', 'SNA', 'DOV', 'PNR', 'ROK', 'TT', 'GNRC', 'MMM', 'TXT', 'HII', 'NOC', 'LHX', 'HWM', 'SPR', 'MAS', 'FBHS', 'BLD', 'J', 'KBR', 'FLR', 'PWR', 'FIX', 'URI', 'AIT', 'AVY', 'SIRI'],
    'Consumer Defensive': ['WMT', 'PG', 'COST', 'KO', 'PEP', 'MDLZ', 'MO', 'PM', 'TGT', 'CL', 'EL', 'KMB', 'GIS', 'KR', 'ADM', 'SYY', 'KDP', 'STZ', 'MNST', 'DG', 'DLTR', 'HSY', 'K', 'CPB', 'CAG', 'HRL', 'SJM', 'MKC', 'BF-B', 'CHD', 'TAP', 'TSN', 'CALM', 'LANC', 'SFBS', 'UNFI', 'HFFG', 'THS', 'JJSF', 'BGS', 'FLO', 'PPC', 'LW', 'POST', 'BRBR', 'UTZ', 'PRMW', 'BYND', 'CELH', 'CHGG', 'EDG', 'HELE', 'HRB', 'NWL', 'REV', 'CUTR', 'PRPL', 'TPX', 'SCI', 'CSV', 'MATW'],
    'Communication Services': ['GOOGL', 'META', 'GOOG', 'DIS', 'NFLX', 'CMCSA', 'TMUS', 'VZ', 'T', 'CHTR', 'ATVI', 'EA', 'WBD', 'IPG', 'OMC', 'LYV', 'TTWO', 'PARA', 'FOXA', 'DISCA', 'ROKU', 'MTCH', 'TME', 'SPOT', 'SNAP', 'PINS', 'TWTR', 'BILI', 'Z', 'ZG', 'IAC', 'NWSA', 'NYT', 'GTN', 'SBGI', 'TGNA', 'VIAC', 'FWONA', 'LSXMA', 'SIRI', 'DISCK', 'DISH', 'LUMN', 'FYBR', 'ATEX', 'IDT', 'MAX', 'MGNI', 'PUBM', 'TTD', 'PERI', 'APPS', 'UPLD'],
    'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL', 'PEG', 'WEC', 'EIX', 'ES', 'AWK', 'ED', 'PCG', 'DTE', 'FE', 'AEE', 'ETR', 'PPL', 'CNP', 'SJI', 'LNT', 'CMS', 'NI', 'EVRG', 'PNW', 'NRG', 'AES', 'ALE', 'BKH', 'CPK', 'IDA', 'MGEE', 'OGE', 'OTTR', 'WTRG'],
    'Real Estate': ['PLD', 'AMT', 'EQIX', 'CCI', 'O', 'PSA', 'SPG', 'WELL', 'DLR', 'AVB', 'WY', 'EQR', 'SBAC', 'VICI', 'ARE', 'INVH', 'VTR', 'ESS', 'MAA', 'CPT', 'EXR', 'CBRE', 'UDR', 'IRM', 'KIM', 'REG', 'FRT', 'PEAK', 'BXP', 'HST', 'MAC', 'SLG', 'VNO', 'DEA', 'KRC', 'HIW', 'SRC', 'ADC', 'EPRT', 'NNN', 'WPC', 'GLPI', 'LAMR', 'OUT', 'PCH', 'GEO', 'CXW', 'UNIT', 'LSI', 'RITM', 'STWD', 'ARI', 'ACRE', 'BRT', 'GOOD', 'IRT', 'APTS', 'AIRC', 'STAG', 'TRNO', 'FR', 'WSR', 'CIO', 'DEA'],
    'Basic Materials': ['LIN','RIO','NEM','SHW','AEM','ECL','BHP','APD','B','WPM','VALE','FNV','GFI','AU','DD','KGC','PPG','TECK','IFF','LYB','RPM','AGI','MP','RGLD','CDE','HMY','SQM','ALB','WLK','AVTR','EQX','HL','SBSW','NEU','IAG','OR','BTG','EMN','ESI','AXTA','EGO','NGD','BCPC','SSRM','NG','SXT','CBT','TMC','SSL','ORLA','SAND','HWKN','USAR','PRM','FUL','USAS','AVNT','WDFC','OLN','SA','DRD','MTRN','CGAU','KWR','ASH','SKE','AAUC','NGVT','CC','LAC','IOSP','MTX','IPX','ARMN','VZLA','UAMY','CRML','CNL','NAK','SCL','REX','TMQ','ECVT','GSM','SLI','ODC','NB','CMP','IAUX','LAR','GAU','SGML','NEXA','LWLG','CMCL','IDR','GRPE','KRO','GTI','NFGC']
}
INDEX_TICKER_TO_NAME = { 
    'XLK': 'US Tech Giants (XLK)', 'SOXX': 'Semiconductors (SOXX)', 
    'SKYY': 'Cloud Computing (SKYY)', 'XLF': 'Financials (XLF)', 
    'XLV': 'Healthcare (XLV)', 'XRT': 'US Retail (XRT)', 
    'HERO': 'Gaming (HERO)', 'XLE': 'Energy (XLE)', 
    'XLI': 'Industrials (XLI)', 'XLP': 'Consumer Staples (XLP)',
    'XLC': 'Communication Services (XLC)', 'XLU': 'Utilities (XLU)',
    'XLRE': 'Real Estate (XLRE)', 'XLB': 'Basic Materials (XLB)',
    '^GSPC': 'S&P 500 Index', '^NDX': 'NASDAQ 100 Index' 
}
SECTOR_TO_INDEX_MAPPING = { 
    'Technology': ['XLK', 'SOXX', 'SKYY'], 
    'Financials': ['XLF'], 
    # --- เปลี่ยน 'Health Services' เป็น 'Healthcare' และเพิ่ม Mapping ใหม่ ---
    'Healthcare': ['XLV'], 
    'Consumer Cyclical': ['XRT', 'HERO'], 
    'Energy': ['XLE'],
    'Industrials': ['XLI'],
    'Consumer Defensive': ['XLP'],
    'Communication Services': ['XLC'],
    'Utilities': ['XLU'],
    'Real Estate': ['XLRE'],
    'Basic Materials': ['XLB']
}
all_possible_symbols = list(INDEX_TICKER_TO_NAME.keys())
for sector_tickers in SECTORS.values(): all_possible_symbols.extend(sector_tickers)
all_possible_symbols = sorted(list(set(all_possible_symbols)))
COLOR_DISCRETE_MAP = { symbol: color for symbol, color in zip(all_possible_symbols, px.colors.qualitative.Plotly * (len(all_possible_symbols) // 10 + 1)) }


# ==================================================================
# 2. Database Models & User Loader (Unchanged)
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
    with server.app_context(): return db.session.get(User, int(user_id))
@server.route('/logout')
def logout():
    logout_user()
    return redirect('/', code=302)

# ==================================================================
# 3. Layout Definition (Unchanged)
# ==================================================================
def create_navbar():
    # ... (unchanged) ...
    if current_user.is_authenticated:
        login_button = dbc.Button("Logout", href="/logout", color="secondary", external_link=True)
    else:
        login_button = dbc.Button("Login", id="open-login-modal-button", color="primary")
    return dbc.Navbar(
        dbc.Container(
            [
                # --- START: NEW LOGO CODE ---
                html.A(
                    dbc.Row(
                        [
                            dbc.Col(html.Img(src="/assets/logo.png", height="45px")),
                            dbc.Col(dbc.NavbarBrand("FINANCIAL ANALYSIS DASHBOARD", className="ms-2 fw-bold")),
                        ],
                        align="center",
                        className="g-0", # g-0 to remove gutters between columns
                    ),
                    href="/",
                    style={"textDecoration": "none"}, # Remove underline from link
                ),
                # --- END: NEW LOGO CODE ---
                dbc.Stack(login_button, direction="horizontal", className="ms-auto")
            ],
            fluid=True
        ),
        color="dark",
        dark=True,
        className="py-2 fixed-top"
    )


def build_layout():
    # ... (unchanged) ...
    return html.Div([
        dcc.Store(id='user-selections-store', storage_type='session'),
        html.Div(id="navbar-container"),
        dbc.Container([
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.Label("Add Stocks to Analysis", className="fw-bold"),
                    dcc.Dropdown(
                        id='sector-dropdown', 
                        options=[{'label': 'All Sectors', 'value': 'All'}] + [{'label': k, 'value': k} for k in SECTORS.keys()], 
                        value='All',  # Set 'All' as the default value
                        clearable=False
                    ),
                    dcc.Dropdown(id='ticker-select-dropdown', className="mt-2", placeholder="Select a ticker..."),
                    dbc.Button([html.I(className="bi bi-plus-circle-fill me-2"), "Add Stock"], id="add-ticker-button", n_clicks=0, className="mt-2 w-100"),
                    html.Hr(),
                    html.Label("Add Benchmarks to Compare", className="fw-bold"),
                    dcc.Dropdown(id='index-select-dropdown', placeholder="Select an index..."),
                    dbc.Button([html.I(className="bi bi-plus-circle-fill me-2"), "Add Benchmark"], id="add-index-button", n_clicks=0, className="mt-2 w-100"),
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
                                ])
                            ]), width="auto"
                        ),
                        dbc.Col(dcc.Dropdown(id='sort-by-dropdown', placeholder="Sort by (best value)..."), width=12, md=4, className="ms-auto align-self-center")
                    ], align="center", className="mt-3"),
                    dcc.Loading(html.Div(
                        id="table-pane-content",
                        children=dash_table.DataTable(
                            id='interactive-datatable',
                            row_selectable=False,
                            cell_selectable=False,
                            style_header={'border': '0px', 'backgroundColor': 'transparent'},
                            style_data={'border': '0px', 'backgroundColor': 'transparent'},
                            style_cell={'textAlign': 'right', 'padding': '10px'},
                            style_header_conditional=[{'if': {'column_id': 'Ticker'},'textAlign': 'left'}],
                            markdown_options={"html": True}
                        ),
                        className="mt-2"
                    ))
                ], width=12, md=9, className="content-offset"),
            ], className="g-4")
        ], fluid=True, className="p-4 main-content-container"),
        create_login_modal()
    ])

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# ==================================================================
# 4. Callbacks (Unchanged Section)
# ==================================================================
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname.startswith('/deepdive/'):
        ticker = pathname.split('/')[-1].upper()
        return html.Div([
            html.Div(id="navbar-container"),
            deep_dive.create_deep_dive_layout(ticker)
        ])
    elif pathname == '/register':
        return create_register_layout()
    else:
        return build_layout()

@app.callback(Output('navbar-container', 'children'),
              Input('url', 'pathname'))
def update_navbar(pathname):
    return create_navbar()

@app.callback(Output('user-selections-store', 'data'),
              Input('url', 'pathname'))
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
        UserSelection.query.filter_by(user_id=user_id, symbol_type=symbol_type).delete()
        for symbol in symbols: db.session.add(UserSelection(user_id=user_id, symbol_type=symbol_type, symbol=symbol))
        db.session.commit()

# ... (All other user selection callbacks remain unchanged) ...
@app.callback(Output('user-selections-store', 'data', allow_duplicate=True), Input('add-ticker-button', 'n_clicks'), [State('ticker-select-dropdown', 'value'), State('user-selections-store', 'data')], prevent_initial_call=True)
def add_ticker_to_store(n_clicks, selected_ticker, store_data):
    store_data = store_data or {'tickers': [], 'indices': []}
    if selected_ticker and selected_ticker not in store_data['tickers']:
        store_data['tickers'].append(selected_ticker)
        if current_user.is_authenticated: save_selections_to_db(current_user.id, store_data['tickers'], 'stock')
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
def add_index_to_store(n_clicks, selected_index, store_data):
    store_data = store_data or {'tickers': [], 'indices': []}
    if selected_index and selected_index not in store_data['indices']:
        store_data['indices'].append(selected_index)
        if current_user.is_authenticated: save_selections_to_db(current_user.id, store_data['indices'], 'index')
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
    if not tickers: return html.Span("No stocks selected.", className="text-muted fst-italic")
    return [html.Label("Selected Stocks:", className="text-muted small")] + [dbc.Badge([t, html.I(className="bi bi-x-circle-fill ms-2", style={'cursor': 'pointer'}, id={'type': 'remove-stock', 'index': t})], color="light", className="m-1 p-2 text-dark border") for t in tickers]
@app.callback(Output('index-summary-display', 'children'), Input('user-selections-store', 'data'))
def update_index_summary_display(store_data):
    indices = store_data.get('indices', []) if store_data else []
    if not indices: return html.Span("No indices selected.", className="text-muted fst-italic")
    return [html.Label("Selected Indices:", className="text-muted small")] + [dbc.Badge([INDEX_TICKER_TO_NAME.get(i, i), html.I(className="bi bi-x-circle-fill ms-2", style={'cursor': 'pointer'}, id={'type': 'remove-index', 'index': i})], color="light", className="m-1 p-2 text-dark border") for i in indices]
@app.callback(
    Output('ticker-select-dropdown', 'options'),
    [Input('sector-dropdown', 'value'), Input('user-selections-store', 'data')]
)
def update_ticker_options(selected_sector, store_data):
    if not selected_sector:
        return []

    selected_tickers = store_data.get('tickers', []) if store_data else []
    
    tickers_to_display = []
    
    if selected_sector == 'All':
        # Logic สำหรับ 'All Sectors' (ซึ่งมีการเรียงลำดับอยู่แล้ว)
        all_tickers_list = []
        for sector_tickers in SECTORS.values():
            all_tickers_list.extend(sector_tickers)
        tickers_to_display = sorted(list(set(all_tickers_list)))
    else:
        # --- START: IMPROVEMENT ---
        # Logic สำหรับ Sector ที่เลือก: เพิ่ม sorted() เพื่อเรียงลำดับหุ้น
        tickers_to_display = sorted(SECTORS.get(selected_sector, []))
        # --- END: IMPROVEMENT ---

    # สร้าง options โดยกรองหุ้นที่ถูกเลือกไปแล้วออก
    options = [{'label': t, 'value': t} for t in tickers_to_display if t not in selected_tickers]
    return options
@app.callback(Output('index-select-dropdown', 'options'), Input('user-selections-store', 'data'))
def update_index_options(store_data):
    if not store_data or not store_data.get('tickers'): return []
    selected_tickers, selected_indices = store_data.get('tickers', []), store_data.get('indices', [])
    active_sectors = {s for t in selected_tickers for s, stocks in SECTORS.items() if t in stocks}
    relevant_indices = {idx for sec in active_sectors for idx in SECTOR_TO_INDEX_MAPPING.get(sec, [])} | {'^GSPC', '^NDX'}
    return [{'label': INDEX_TICKER_TO_NAME.get(i, i), 'value': i} for i in sorted(list(relevant_indices)) if i not in selected_indices]


# ==================================================================
# 5. Callbacks for Main Dashboard Graphs & Table (Unchanged)
# ==================================================================
@app.callback(
    Output('analysis-pane-content', 'children'),
    [Input('analysis-tabs', 'active_tab'), Input('user-selections-store', 'data')]
)
def render_graph_content(active_tab, store_data):
    # ... (This function is unchanged) ...
    store_data = store_data or {'tickers': [], 'indices': []}
    tickers = tuple(store_data.get('tickers', []))
    if active_tab == "tab-performance":
        all_symbols = list(set(tickers + tuple(store_data.get('indices', []))))
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
        all_symbols = tuple(set(tickers + tuple(store_data.get('indices', []))))
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
            output_components, dcf_results = [], [calculate_dcf_intrinsic_value(t, growth_rate) for t in tickers]
            successful_results, failed_results = [res for res in dcf_results if 'error' not in res], [res for res in dcf_results if 'error' in res]
            if failed_results:
                failed_tickers = ', '.join([res['Ticker'] for res in failed_results])
                output_components.append(dbc.Alert([html.I(className="bi bi-exclamation-triangle-fill me-2"), f"Could not calculate DCF for: {failed_tickers} (Missing data)"], color="warning", className="mb-3",dismissable=True))
            if successful_results:
                df_dcf = pd.DataFrame(successful_results)
                fig = go.Figure()
                for i, row in df_dcf.iterrows(): fig.add_shape(type='line', x0=row['current_price'], y0=row['Ticker'], x1=row['intrinsic_value'], y1=row['Ticker'], line=dict(color='limegreen' if row['intrinsic_value'] > row['current_price'] else 'tomato', width=3))
                fig.add_trace(go.Scatter(x=df_dcf['current_price'], y=df_dcf['Ticker'], mode='markers', marker=dict(color='royalblue', size=10), name='Current Price'))
                fig.add_trace(go.Scatter(x=df_dcf['intrinsic_value'], y=df_dcf['Ticker'], mode='markers', marker=dict(color='darkorange', size=10, symbol='diamond'), name='Intrinsic Value (DCF)'))
                fig.update_layout(title=f'Margin of Safety (DCF) with {growth_rate:.0%} Growth Forecast', xaxis_title='Share Price ($)', yaxis_title='Ticker', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                output_components.append(dbc.Card(dbc.CardBody(dcc.Graph(figure=fig))))
            elif not successful_results and failed_results: return output_components
            elif not successful_results and not failed_results: return dbc.Alert("Could not process DCF for any selected stocks.", color="danger")
            return output_components
        except Exception as e: return dbc.Alert(f"An error occurred while rendering DCF chart: {e}", color="danger")
    return html.P("This is an empty tab!")

@app.callback(
    Output('interactive-datatable', 'data'), Output('interactive-datatable', 'columns'),
    Output('interactive-datatable', 'style_data_conditional'),
    Output('interactive-datatable', 'style_cell_conditional'),
    Output('interactive-datatable', 'tooltip_header'),
    Output('sort-by-dropdown', 'options'), Output('sort-by-dropdown', 'value'),
    [Input('table-tabs', 'active_tab'), Input('user-selections-store', 'data'), Input('sort-by-dropdown', 'value')]
)
def render_table_content(active_tab, store_data, sort_by_column):
    # ... (This function is unchanged) ...
    if not store_data or not store_data.get('tickers'): return [], [], [], [], {}, [], None
    tickers = tuple(store_data.get('tickers')); df_full = get_competitor_data(tickers)
    if df_full.empty: return [], [], [], [], {}, [], None
    TABS_CONFIG = {
        "tab-valuation": { "columns": ["Ticker", "Market Cap", "Price", "P/E", "P/B", "EV/EBITDA"], "higher_is_better": {"P/E": False, "P/B": False, "EV/EBITDA": False} },
        "tab-growth": { "columns": ["Ticker", "Revenue Growth (YoY)", "Revenue CAGR (3Y)", "Net Income Growth (YoY)"], "higher_is_better": {k: True for k in ["Revenue Growth (YoY)", "Revenue CAGR (3Y)", "Net Income Growth (YoY)"]} },
        "tab-fundamentals": { "columns": ["Ticker", "Operating Margin", "ROE", "D/E Ratio", "Cash Conversion"], "higher_is_better": {"Operating Margin": True, "ROE": True, "D/E Ratio": False, "Cash Conversion": True} }
    }
    TOOLTIP_DEFINITIONS = { "Ticker": "Click for Deep Dive Analysis", "Market Cap": "Total market value", "Price": "Current price", "P/E": "Price-to-Earnings", "P/B": "Price-to-Book", "EV/EBITDA": "Enterprise Value to EBITDA", "Revenue Growth (YoY)": "Year-over-Year Revenue Growth", "Revenue CAGR (3Y)": "3-Year Compound Annual Growth Rate of Revenue", "Net Income Growth (YoY)": "Year-over-Year Earnings Growth", "Operating Margin": "Operating Income / Revenue", "ROE": "Return on Equity", "D/E Ratio": "Total Debt / Equity", "Cash Conversion": "Operating Cash Flow / Net Income" }
    config = TABS_CONFIG[active_tab]
    if sort_by_column:
        ascending = not config['higher_is_better'].get(sort_by_column, True)
        df_full.sort_values(by=sort_by_column, ascending=ascending, na_position='last', inplace=True)
    df_display = df_full.copy()
    df_display['Ticker'] = df_display['Ticker'].apply(
         lambda t: f'<a href="/deepdive/{t}" style="text-decoration: none; color: inherit;">{t}</a>'
    )
    def format_market_cap(n):
        if pd.isna(n): return '-'
        return f'${n/1e9:,.2f}B' if n >= 1e9 else f'${n/1e6:,.2f}M'
    if "Market Cap" in df_display.columns: df_display["Market Cap"] = df_full["Market Cap"].apply(format_market_cap)
    displayed_columns = config["columns"]
    other_cols = [c for c in displayed_columns if c != 'Ticker']
    num_other_cols = len(other_cols)
    ticker_width_percent = 15
    style_cell_conditional = []
    style_cell_conditional.append({'if': {'column_id': 'Ticker'},'width': f'{ticker_width_percent}%','minWidth': f'{ticker_width_percent}%','maxWidth': f'{ticker_width_percent}%'})
    if num_other_cols > 0:
        remaining_width = 100 - ticker_width_percent
        other_col_width_percent = remaining_width / num_other_cols
        for col in other_cols:
            style_cell_conditional.append({'if': {'column_id': col}, 'width': f'{other_col_width_percent}%'})
    columns = []
    for col in config["columns"]:
        col_def = {"name": col.replace(" (3Y)", ""), "id": col}
        if col == 'Ticker': col_def.update({"type": "text", "presentation": "markdown"})
        elif col == "Market Cap": col_def.update({"type": "text"})
        elif any(kw in col for kw in ['Growth', 'Margin', 'ROE']): col_def.update({"type": "numeric", "format": Format(precision=2, scheme=Scheme.percentage)})
        else: col_def.update({"type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)})
        columns.append(col_def)
    style_data_conditional=[{'if': {'column_id': 'Ticker'},'textAlign': 'left'}]
    dropdown_options = [{'label': col, 'value': col} for col in config["columns"] if col != 'Ticker']
    data = df_display[config["columns"]].to_dict('records')
    tooltips = {k: v for k, v in TOOLTIP_DEFINITIONS.items() if k in config["columns"]}
    sort_value = sort_by_column if callback_context.triggered_id == 'sort-by-dropdown' else None
    return data, columns, style_data_conditional, style_cell_conditional, tooltips, dropdown_options, sort_value

# ==================================================================
# 6. Callbacks for Deep Dive Page  <-- [CRITICAL FIX IS HERE]
# ==================================================================

@app.callback(
    Output('deep-dive-tab-content', 'children'),
    Input('deep-dive-main-tabs', 'active_tab'),
    State('deep-dive-ticker-store', 'data')
)
def render_deep_dive_tab_content(active_tab, store_data):
    if not store_data or not store_data.get('ticker'):
        return "" # Return empty if no ticker is available
    
    ticker = store_data['ticker']
    data = get_deep_dive_data(ticker)

    if active_tab == "tab-charts-deep-dive":
        financial_trends = data.get("financial_trends", pd.DataFrame())
        price_history = data.get("price_history", pd.DataFrame())

        # Financial Trends Figure
        fig_trends = go.Figure()
        if not financial_trends.empty:
            fig_trends.add_trace(go.Bar(x=financial_trends.index, y=financial_trends['Revenue'], name='Revenue', marker_color='royalblue'))
            fig_trends.add_trace(go.Scatter(x=financial_trends.index, y=financial_trends['Net Income'], name='Net Income', yaxis='y2', mode='lines+markers', line=dict(color='darkorange')))
        fig_trends.update_layout(
            title=f'Annual Financial Trends',
            yaxis=dict(title='Revenue ($)'),
            yaxis2=dict(title='Net Income ($)', overlaying='y', side='right', showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=50)
        )

        # Price History Figure
        fig_price = go.Figure()
        if not price_history.empty:
            fig_price.add_trace(go.Scatter(x=price_history.index, y=price_history['Close'], mode='lines', name='Close Price', line=dict(color='#636EFA')))
        fig_price.update_layout(title=f'5-Year Stock Price History', xaxis_title='Date', yaxis_title='Price ($)', margin=dict(t=50))
        
        return dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_price), md=6),
            dbc.Col(dcc.Graph(figure=fig_trends), md=6),
        ], className="mt-3")

    elif active_tab == "tab-financials-deep-dive":
        return html.Div([
            dbc.Tabs(
                id="financial-statement-tabs",
                active_tab="tab-income",
                children=[
                    dbc.Tab(label="Income Statement", tab_id="tab-income"),
                    dbc.Tab(label="Balance Sheet", tab_id="tab-balance"),
                    dbc.Tab(label="Cash Flow", tab_id="tab-cashflow"),
                ],
                className="mt-3"
            ),
            dcc.Loading(html.Div(id="financial-statement-content", className="mt-3 table-responsive"))
        ])

    elif active_tab == "tab-valuation-deep-dive":
        return dbc.Row([
            dbc.Col([
                html.H5("DCF Assumptions", className="mt-3"),
                html.Hr(),
                dbc.Label("Your Forecast Growth Rate (%):", html_for="dcf-growth-rate-input"),
                dcc.Input(id='dcf-growth-rate-input', type='number', value=5, step=0.5, className="mb-2 form-control"),
                html.P("This model uses a standard WACC calculation and a perpetual growth rate of 2.5%.", className="text-muted small")
            ], md=4),
            dbc.Col(
                dcc.Loading(dcc.Graph(id='interactive-dcf-chart')),
                md=8
            )
        ], className="mt-3", align="center")
        
    return html.P("Select a tab")


@app.callback(
    Output('financial-statement-content', 'children'),
    Input('financial-statement-tabs', 'active_tab'),
    State('deep-dive-ticker-store', 'data')
)
def render_financial_statement_table(active_tab, store_data):
    if not store_data or not store_data.get('ticker'): return ""
    ticker = store_data['ticker']
    data = get_deep_dive_data(ticker)
    statements = data.get("financial_statements", {})
    df = pd.DataFrame()
    if active_tab == 'tab-income': df = statements.get('income', pd.DataFrame())
    elif active_tab == 'tab-balance': df = statements.get('balance', pd.DataFrame())
    elif active_tab == 'tab-cashflow': df = statements.get('cashflow', pd.DataFrame())
    if df.empty: return dbc.Alert("Financial data not available.", color="warning")
    
    # Format numbers to be more readable (e.g., in millions)
    df_formatted = df.applymap(lambda x: f"{x/1_000_000:,.1f}M" if isinstance(x, (int, float)) else x)
    df_reset = df_formatted.reset_index().rename(columns={'index': 'Metric'})
    
    return dbc.Table.from_dataframe(df_reset, striped=True, bordered=True, hover=True, class_name="small")

@app.callback(
    Output('interactive-dcf-chart', 'figure'),
    Input('dcf-growth-rate-input', 'value'),
    State('deep-dive-ticker-store', 'data')
)
def update_interactive_dcf_chart(growth_rate, store_data):
    if not store_data or not store_data.get('ticker') or growth_rate is None:
        return go.Figure().update_layout(title_text="Enter a growth rate to calculate.")
    ticker = store_data['ticker']
    growth_rate_decimal = float(growth_rate) / 100.0
    result = calculate_dcf_intrinsic_value(ticker, growth_rate_decimal)
    if 'error' in result:
        return go.Figure().update_layout(title=f"Error Calculating DCF for {ticker}", annotations=[dict(text=result['error'], showarrow=False)])
    
    iv = result.get('intrinsic_value', 0)
    price = result.get('current_price', 0)
    
    margin_of_safety = ((iv / price) - 1) if price > 0 else 0
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = price,
        number = {'prefix': "$", 'font': {'size': 40}},
        delta = {'reference': iv, 'relative': False, 'valueformat': '.2f', 
                 'increasing': {'color': "#dc3545"}, 'decreasing': {'color': "#198754"}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {
            'text': f"<b>Margin of Safety: {margin_of_safety:.2%}</b><br><span style='font-size:0.8em;color:gray'>Intrinsic Value (DCF): ${iv:.2f}</span>",
            'font': {'size': 20}
        },
        gauge = {
            'axis': {'range': [min(price, iv) * 0.75, max(price, iv) * 1.25], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "rgba(0,0,0,0)"}, # Make the main bar transparent
            'steps' : [
                {'range': [0, iv], 'color': "rgba(25, 135, 84, 0.2)"},
                {'range': [iv, max(price, iv) * 1.25], 'color': "rgba(220, 53, 69, 0.2)"}
            ],
            'threshold' : {
                'line': {'color': "#636EFA", 'width': 5}, 'thickness': 1, 'value': price
            }
        }
    ))
    fig.update_layout(
        title_text=f"Valuation with {growth_rate:.1f}% Growth",
        height=400,
        margin=dict(t=80, b=40)
    )
    return fig


# ==================================================================
# 7. Register Auth Callbacks & Run App (Unchanged)
# ==================================================================
register_auth_callbacks(app, db, User)

if __name__ == '__main__':
    with server.app_context():
        db.create_all()
    app.run(debug=True)
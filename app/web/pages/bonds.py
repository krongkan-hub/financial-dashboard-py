import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
from app.constants import BOND_YIELD_MAP, BOND_BENCHMARK_MAP, TOP_5_DEFAULT_TICKERS, HISTORICAL_START_DATE
from datetime import date

# --- 4.6 (Optional) Bond Specific Metric Definitions ---
# This dictionary is used by the Definitions Modal on the bonds page.
BOND_METRIC_DEFINITIONS = {
    "tab-yield-history": {
        "title": "HISTORICAL YIELDS",
        "description": "Historical performance data for selected Treasury Yields and Benchmarks.",
        "metrics": [
            {"metric": "Yield (%)", "definition": "The daily closing yield or price return of the selected instrument."},
            {"metric": "Date Range", "definition": f"Data is currently pulled from {HISTORICAL_START_DATE.strftime('%Y-%m-%d')} to the latest available day."},
        ]
    },
    "tab-yield-curve": {
        "title": "YIELD CURVE",
        "description": "Visualizes the current yield curve (yield vs. maturity).",
        "metrics": [
            {"metric": "Yield Curve", "definition": "A line plot showing the interest rates for Treasury securities with different maturity dates. An inverted curve (short-term > long-term) is often a recession indicator."},
        ]
    },
    "tab-yield-spread": {
        "title": "YIELD SPREAD",
        "description": "The difference in yield between two selected instruments (e.g., 10-Year Treasury minus 2-Year Treasury).",
        "metrics": [
            {"metric": "Spread", "definition": "The difference between the yields of two instruments. Widening or narrowing spreads reflect changes in market sentiment and risk perception."},
        ]
    },
    "tab-rates-summary": {
        "title": "RATES SUMMARY",
        "description": "A summary table showing the latest yield, 1-day change, and relevant metadata.",
        "metrics": [
            {"metric": "Latest Yield (%)", "definition": "The closing yield on the last trading day."},
            {"metric": "1-Day Change (bps)", "definition": "The change in yield from the previous day, measured in basis points (bps). 100 bps = 1.00%."},
        ]
    },
}

# --- Shared Components (Used multiple times in the layout) ---

# The Alert that displays selected tickers (Yields and Benchmarks)
# Replaced 'ticker-summary-display' with 'bonds-summary-display' and 'index-summary-display' with 'bonds-benchmark-summary-display'
selected_tickers_display = [
    dbc.Alert(
        [
            html.H5("Selected Yields:", className="alert-heading"),
            html.Div(id='bonds-summary-display', children="No Yields Selected"),
        ],
        color="info",
        className="mb-2",
        style={'fontSize': '0.9rem', 'padding': '0.5rem'}
    ),
    dbc.Alert(
        [
            html.H5("Selected Benchmarks:", className="alert-heading"),
            html.Div(id='bonds-benchmark-summary-display', children="No Benchmarks Selected"),
        ],
        color="secondary",
        className="mb-2",
        style={'fontSize': '0.9rem', 'padding': '0.5rem'}
    ),
]

# The main content area where graphs and tables are displayed
# Replaced 'analysis-pane-content' and 'table-pane-content'
main_content_pane = [
    dbc.Tabs(
        id="bonds-analysis-tabs",
        active_tab="tab-yield-history",
        className="nav-justified",
        # --- 4.5 Graph Tabs Content Changed ---
        children=[
            dbc.Tab(label="HISTORICAL YIELDS", tab_id="tab-yield-history"),
            dbc.Tab(label="YIELD CURVE", tab_id="tab-yield-curve"),
            dbc.Tab(label="YIELD SPREAD", tab_id="tab-yield-spread"),
        ]
    ),
    html.Div(id='bonds-analysis-pane-content', className="mt-3 card-body shadow p-3 mb-5 bg-white rounded"),

    html.Hr(className="my-4"),

    dbc.Row([
        dbc.Col(
            dbc.Tabs(
                id="bonds-table-tabs",
                active_tab="tab-rates-summary",
                className="nav-justified",
                # --- 4.5 Table Tabs Content Changed ---
                children=[
                    dbc.Tab(label="RATES SUMMARY", tab_id="tab-rates-summary"),
                ]
            ),
            width=12
        ),
    ]),
    html.Div(id='bonds-table-pane-content', className="mt-3 card-body shadow p-3 mb-5 bg-white rounded"),
]

# --- Modal for Metric Definitions ---
# Replaced all 'definitions-modal' and related IDs
definitions_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Definition Dictionary")),
        dbc.ModalBody(
            id="bonds-definitions-modal-content",
            children=[
                html.P("Select a tab or click the info icon (ⓘ) next to a metric to see its definition."),
            ]
        ),
        dbc.ModalFooter(
            dbc.Button("Close", id="bonds-close-definitions-modal", className="ms-auto", n_clicks=0)
        ),
    ],
    id="bonds-definitions-modal",
    size="lg",
    is_open=False,
)

# --- 4.1 Main Layout Function Changed ---
def create_bonds_layout():
    """Generates the layout for the /bonds page."""
    return dbc.Container(
        [
            # --- Hidden Data Stores (4.2 IDs Changed) ---
            dcc.Store(id='bonds-user-selections-store', data={'tickers': ['^TNX'], 'indices': ['^GSPC']}),
            dcc.Store(id='bonds-forecast-assumptions-store', data={}),
            dcc.Store(id='bonds-dcf-assumptions-store', data={}),

            dbc.Row(
                [
                    # --- Left Sidebar (Controls) ---
                    dbc.Col(
                        [
                            html.H3("Bonds & Yields Analysis 💰"),
                            html.Hr(),

                            # --- Yield Selection (4.3 IDs & Text Changed) ---
                            html.Div([
                                html.Label("Add Yields to Analysis", className="mt-2 text-primary font-weight-bold"),
                                dcc.Dropdown(
                                    id='bonds-yield-type-dropdown',
                                    options=[{'label': k, 'value': k} for k in BOND_YIELD_MAP.keys()], # Uses BOND_YIELD_MAP keys
                                    value=list(BOND_YIELD_MAP.keys())[0], # Default to the first key
                                    clearable=False,
                                    className="mb-2"
                                ),
                                dcc.Dropdown(
                                    id='bonds-yield-select-dropdown',
                                    options=[{'label': v, 'value': k} for k, v in BOND_YIELD_MAP.items()], # Options populated in callback
                                    placeholder="Select a Yield Ticker",
                                    className="mb-2"
                                ),
                                dbc.Button("Add Yield", id='bonds-add-yield-button', color="primary", className="mb-3", n_clicks=0),
                            ], className="border rounded p-3 mb-3"),

                            # --- Benchmark Selection (4.3 IDs & Text Changed) ---
                            html.Div([
                                html.Label("Add Benchmarks", className="mt-2 text-info font-weight-bold"),
                                dcc.Dropdown(
                                    id='bonds-benchmark-select-dropdown',
                                    options=[{'label': v, 'value': k} for k, v in BOND_BENCHMARK_MAP.items()], # Uses BOND_BENCHMARK_MAP
                                    placeholder="Select an Index or ETF",
                                    className="mb-2"
                                ),
                                dbc.Button("Add Benchmark", id='bonds-add-benchmark-button', color="info", className="mb-3", n_clicks=0),
                            ], className="border rounded p-3 mb-3"),

                            # --- Selected Items Display ---
                            *selected_tickers_display,

                            # --- Peer Comparison (4.3 IDs & Text Changed & Disabled) ---
                            html.Div([
                                html.Label("Find Similar (Coming Soon)", className="mt-2 text-warning font-weight-bold"),
                                dcc.Dropdown(
                                    id='bonds-peer-reference-dropdown',
                                    placeholder="Reference Yield Ticker",
                                    className="mb-2",
                                    disabled=True  # Disabled for bonds page
                                ),
                                dcc.Dropdown(
                                    id='bonds-peer-select-dropdown',
                                    placeholder="Select Peer Tickers",
                                    multi=True,
                                    className="mb-2",
                                    disabled=True # Disabled for bonds page
                                ),
                                dbc.Button("Add Peer", id='bonds-add-peer-button', color="warning", className="mb-3", n_clicks=0, disabled=True), # Disabled for bonds page
                            ], className="border rounded p-3 mb-3", style={'opacity': 0.6}),

                            html.Div(
                                [
                                    html.Hr(),
                                    html.Small(f"Data from Yahoo Finance. Last update: {date.today().strftime('%Y-%m-%d')}"),
                                    dbc.Button(
                                        "Definitions ⓘ",
                                        id="bonds-open-definitions-modal-btn-graphs",
                                        color="secondary",
                                        size="sm",
                                        className="float-end",
                                        n_clicks=0,
                                    ),
                                ],
                                className="mt-3",
                            ),
                        ],
                        md=3,
                        className="sidebar"
                    ),

                    # --- Right Content (Graphs & Tables) ---
                    dbc.Col(
                        [
                            # --- Sort & Modal Buttons (4.4 IDs Changed) ---
                            dbc.Row([
                                dbc.Col(
                                    html.Div([
                                        dbc.Button("Valuation Assumptions (Coming Soon)", id="bonds-open-dcf-modal-btn", className="me-2", n_clicks=0, disabled=True),
                                        dbc.Button("Definitions ⓘ", id="bonds-open-definitions-modal-btn-tables", className="me-2", n_clicks=0),
                                        dcc.Dropdown(
                                            id='bonds-sort-by-dropdown',
                                            options=[
                                                {'label': 'Sort by Yield (%)', 'value': 'latest_yield'},
                                                {'label': 'Sort by 1-Day Change (bps)', 'value': 'change_bps'},
                                            ],
                                            value='latest_yield',
                                            placeholder="Sort Table By...",
                                            style={'width': '200px', 'display': 'inline-block', 'verticalAlign': 'middle'}
                                        ),
                                    ], className="d-flex justify-content-end mb-3"),
                                    width=12
                                )
                            ]),

                            # --- Main Content ---
                            *main_content_pane,

                        ],
                        md=9
                    ),
                ]
            ),
            # --- Modals ---
            definitions_modal,
            # DCF/Forecast Modal (IDs Changed and Disabled for Bonds)
            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle("Valuation Assumptions (Coming Soon)")),
                dbc.ModalBody(html.P("Valuation models like DCF are not applicable to interest rate data and are disabled for this view.")),
                dbc.ModalFooter(dbc.Button("Close", id="bonds-close-forecast-assumptions-modal", className="ms-auto", n_clicks=0)),
            ], id="bonds-forecast-assumptions-modal", is_open=False, size="lg"),
        ],
        fluid=True,
    )
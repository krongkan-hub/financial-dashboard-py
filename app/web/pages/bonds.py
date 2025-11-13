from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from app.constants import BOND_YIELD_MAP, BOND_BENCHMARK_MAP, TOP_5_DEFAULT_TICKERS, HISTORICAL_START_DATE
from datetime import date

# --- 4.6 (Optional) Bond Specific Metric Definitions (Unchanged) ---
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

# --- Shared Components ---

# The main content area where graphs and tables are displayed (Corrected structure)
main_content_pane = [
    # --- Graph Tabs ---
    html.Div(className="custom-tabs-container", children=[
        dbc.Tabs(
            id="bonds-analysis-tabs",
            active_tab="tab-yield-history",
            children=[
                dbc.Tab(label="HISTORICAL YIELDS", tab_id="tab-yield-history"),
                dbc.Tab(label="YIELD CURVE", tab_id="tab-yield-curve"),
                dbc.Tab(label="YIELD SPREAD", tab_id="tab-yield-spread"),
            ]
        )
    ]),

    # --- Graph Content Pane ---
    dbc.Card(dbc.CardBody(
        dcc.Loading(html.Div(id='bonds-analysis-pane-content'))
    ), className="mt-3"),

    html.Hr(className="my-5"),

    # --- Table Tabs ---
    dbc.Row(
        [
            dbc.Col(
                html.Div(className="custom-tabs-container", children=[
                    dbc.Tabs(
                        id="bonds-table-tabs",
                        active_tab="tab-rates-summary",
                        children=[
                            dbc.Tab(label="RATES SUMMARY", tab_id="tab-rates-summary"),
                        ]
                    )
                ]),
                md=7
            ),
             # Buttons/Dropdown Col
             dbc.Col(
                dbc.Stack(
                    [
                        # Hidden Gear button for Forecast Modal (per instructions)
                        dbc.Button(html.I(className="bi bi-gear-fill"), id="bonds-open-forecast-modal-btn", color="secondary", outline=True, style={'display': 'none'}),
                        # Info button (Icon only)
                        dbc.Button(html.I(className="bi bi-info-circle-fill"), id="bonds-open-definitions-modal-btn-tables", color="secondary", outline=True),
                        dcc.Dropdown(id='bonds-sort-by-dropdown', placeholder="Sort by", style={'minWidth': '180px'})
                    ],
                    direction="horizontal",
                    gap=2,
                    className="justify-content-start justify-content-lg-end"
                ),
                md=5
            )
        ],
        align="center",
        className="control-row"
    ),
    
    # --- Table Content Pane ---
    dbc.Card(dbc.CardBody(
        dcc.Loading(html.Div(id='bonds-table-pane-content'))
    ), className="mt-2")
]

# --- Modal for Metric Definitions (Unchanged) ---
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

# --- Main Layout Function ---
def create_bonds_layout():
    """Generates the layout for the /bonds page."""
    return dbc.Container(
        [
            # --- Hidden Data Stores ---
            dcc.Store(id='bonds-user-selections-store', data={'tickers': ['^TNX'], 'indices': ['^GSPC']}),
            dcc.Store(id='bonds-forecast-assumptions-store', data={}),
            dcc.Store(id='bonds-dcf-assumptions-store', data={}),
            html.Div(id="navbar-container"), 

            dbc.Row(
                [
                    # --- Left Sidebar (Controls) ---
                    dbc.Col(
                        dbc.Card(dbc.CardBody([
                            # [FIX] ลบ "BONDS & YIELDS ANALYSIS 💰" ออกแล้ว
                            html.Label("Add Yields to Analysis", className="fw-bold"),
                            dcc.Dropdown(
                                id='bonds-yield-type-dropdown',
                                options=[{'label': k, 'value': k} for k in BOND_YIELD_MAP.keys()],
                                value=list(BOND_YIELD_MAP.keys())[0],
                                clearable=False,
                                className="mb-2"
                            ),
                            dcc.Dropdown(
                                id='bonds-yield-select-dropdown',
                                options=[{'label': v, 'value': k} for k, v in BOND_YIELD_MAP.items()],
                                placeholder="Select one or more yields...",
                                multi=True, 
                                className="mt-2 sidebar-dropdown"
                            ),
                            # [FIX] ปุ่ม Add Yield: ใช้ color="primary" และสไตล์ Stocks
                            dbc.Button([html.I(className="bi bi-plus-circle-fill me-2"), "Add Yield(s)"], id='bonds-add-yield-button', color="primary", className="mt-2 w-100", n_clicks=0),
                            
                            html.Hr(),

                            html.Label("Add Benchmarks to Compare", className="fw-bold"),
                            dcc.Dropdown(
                                id='bonds-benchmark-select-dropdown',
                                options=[{'label': v, 'value': k} for k, v in BOND_BENCHMARK_MAP.items()],
                                placeholder="Select one or more benchmarks...",
                                multi=True, 
                                # [MODIFIED] ลบ mt-2 ออก เพื่อให้ระยะห่างระหว่าง Label กับ Dropdown เท่ากับส่วน Yields
                                className="sidebar-dropdown"
                            ),
                            # [FIX] ปุ่ม Add Benchmark: ใช้ color="primary" เหมือน Add Yield
                            dbc.Button([html.I(className="bi bi-plus-circle-fill me-2"), "Add Benchmark(s)"], id='bonds-add-benchmark-button', color="primary", className="mt-2 w-100", n_clicks=0),

                            # [FIX] ลบ Find Similar ออกไปแล้ว

                            html.Hr(className="my-4"),

                            # --- Selected Items Display (ใช้ Div ว่างเปล่าเหมือน Stocks) ---
                            html.Div(id='bonds-summary-display', className="mb-2"),
                            html.Div(id='bonds-benchmark-summary-display', className="pt-0"), # pt-0 เพื่อให้ชิดกับ Div บน

                            
                        ])),
                        md=3,
                        className="sidebar-fixed"
                    ),

                    # --- Right Content (Graphs & Tables) ---
                    dbc.Col(
                        [
                            # Graph Controls Row (Top Right)
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(className="custom-tabs-container", children=[
                                            dbc.Tabs(id="bonds-analysis-tabs", active_tab="tab-yield-history", children=[
                                                dbc.Tab(label="HISTORICAL YIELDS", tab_id="tab-yield-history"),
                                                dbc.Tab(label="YIELD CURVE", tab_id="tab-yield-curve"),
                                                dbc.Tab(label="YIELD SPREAD", tab_id="tab-yield-spread"),
                                            ])
                                        ]),
                                        md=8
                                    ),
                                    dbc.Col(
                                        dbc.Stack(
                                            [
                                                # [FIX] Gear button (DCF/Forecast Modal - Hidden)
                                                dbc.Button(html.I(className="bi bi-gear-fill"), id="bonds-open-dcf-modal-btn", color="secondary", outline=True, style={'display': 'none'}),
                                                # [FIX] Info button (Icon only)
                                                dbc.Button(html.I(className="bi bi-info-circle-fill"), id="bonds-open-definitions-modal-btn-graphs", color="secondary", outline=True),
                                            ],
                                            direction="horizontal",
                                            gap=2,
                                            className="justify-content-start justify-content-lg-end"
                                        ),
                                        md=4
                                    )
                                ],
                                align="center",
                                className="control-row"
                            ),

                            # Graph Content (ใช้ dbc.Card(dbc.CardBody(...)))
                            dbc.Card(dbc.CardBody(
                                dcc.Loading(html.Div(id='bonds-analysis-pane-content'))
                            ), className="mt-3"),

                            html.Hr(className="my-5"),

                            # Table Controls Row (Bottom Right)
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(className="custom-tabs-container", children=[
                                            dbc.Tabs(id="bonds-table-tabs", active_tab="tab-rates-summary", children=[
                                                dbc.Tab(label="RATES SUMMARY", tab_id="tab-rates-summary"),
                                            ])
                                        ]),
                                        md=7
                                    ),
                                     dbc.Col(
                                        dbc.Stack(
                                            [
                                                # [FIX] Gear button for forecast (Hidden for bonds)
                                                dbc.Button(html.I(className="bi bi-gear-fill"), id="bonds-open-forecast-modal-btn", color="secondary", outline=True, style={'display': 'none'}),
                                                # [FIX] Info button (Icon only)
                                                dbc.Button(html.I(className="bi bi-info-circle-fill"), id="bonds-open-definitions-modal-btn-tables", color="secondary", outline=True),
                                                # [FIX] Sort by dropdown
                                                dcc.Dropdown(id='bonds-sort-by-dropdown', placeholder="Sort by", style={'minWidth': '180px'})
                                            ],
                                            direction="horizontal",
                                            gap=2,
                                            className="justify-content-start justify-content-lg-end"
                                        ),
                                        md=5
                                    )
                                ],
                                align="center",
                                className="control-row" 
                            ),

                            # Table Content
                            dcc.Loading(html.Div(id="bonds-table-pane-content", className="mt-2"))
                        ],
                        md=9,
                        className="content-offset"
                    ),
                ], className="g-4")
        ],
        fluid=True,
        className="p-4 main-content-container"
    )
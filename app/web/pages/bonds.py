# app/web/pages/bonds.py

from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
# นำเข้า BOND_YIELD_MAP, BOND_BENCHMARK_MAP จาก app.constants
from app.constants import BOND_YIELD_MAP, BOND_BENCHMARK_MAP, TOP_5_DEFAULT_TICKERS, HISTORICAL_START_DATE
from datetime import date

# --- 4.6 (Optional) Bond Specific Metric Definitions (UPDATED) ---
# This dictionary is used by the Definitions Modal on the bonds page.
BOND_METRIC_DEFINITIONS = {
    "tab-yield-history": {
        "title": "HISTORICAL YIELDS",
        "description": "Historical performance data for selected Treasury Yields and Benchmarks.",
        "metrics": [
            {"metric": "Yield (%)", "definition": "The daily closing yield or price return of the selected instrument."},
            {"metric": "THB Hedged Yield", "definition": "The theoretical yield (in THB terms) after hedging currency risk, calculated based on the THB/USD interest rate differential."},
            {"metric": "Date Range", "definition": f"Data is currently pulled from {HISTORICAL_START_DATE.strftime('%Y-%m-%d')} to the latest available day."},
        ]
    },
    "tab-yield-curve": {
        "title": "YIELD CURVE",
        "description": "Visualizes the current and past yield curves for comparison, highlighting shifts in market expectations. The curve is plotted using the selected US Treasury Yields.",
        "metrics": [
            {"metric": "Yield Curve (Normal)", "definition": "Upward sloping (short-term < long-term). Signals expected economic growth."},
            {"metric": "Yield Curve (Inverted)", "definition": "Downward sloping (short-term > long-term). Historically a strong predictor of economic recession."},
            {"metric": "Comparison Lines", "definition": "Shows the curve's shape today vs. a week, month, or year ago, reflecting the Fed's impact on rates."},
        ]
    },
    "tab-yield-spread": {
        "title": "YIELD SPREAD",
        "description": "The difference in yield between multiple Series (e.g., HYG, LQD, 2Y T-Note) and a Single Benchmark (e.g., 10-Year Treasury).",
        "metrics": [
            {"metric": "Spread (bps)", "definition": "The difference between the yields of two instruments, measured in basis points (bps). Widening or narrowing spreads reflect changes in market sentiment and credit risk perception. (100 bps = 1.00%)"},
            {"metric": "Credit Spread", "definition": "The difference between Corporate Bond Yield (HYG/LQD) and a risk-free Treasury yield (^TNX). High spread = High credit risk."},
        ]
    },
    "tab-yield-volatility": {
        "title": "YIELD VOLATILITY",
        "description": "Historical trends of bond market volatility using the MOVE Index (Merrill Lynch Option Volatility Estimate).",
        "metrics": [
            {"metric": "MOVE Index", "definition": "A volatility index for US Treasury bonds. A rising index indicates higher uncertainty and expected future volatility in the bond market."},
            {"metric": "Duration", "definition": "Measures a bond's price sensitivity to changes in interest rates. A higher Duration implies higher volatility for a bond's price."},
        ]
    },
    "tab-rates-summary": {
        "title": "RATES SUMMARY",
        "description": "A summary table showing the latest yield, daily/weekly/YTD change, key spreads, and relevant benchmarks.",
        "metrics": [
            {"metric": "Latest Yield (%)", "definition": "The closing yield on the last trading day."},
            {"metric": "Change (bps)", "definition": "The change in yield over a period, measured in basis points."},
            {"metric": "THB Hedged Yield", "definition": "The theoretical yield an investor would receive if the USD Bond yield was hedged back to THB, calculated using the THB/USD interest rate differential."},
            {"metric": "Key Spreads", "definition": "Calculated spreads such as 10Y-2Y and Credit Spread (HYG-10Y)."},
        ]
    },
    # [NEW TAB 1: CREDIT & STATUS]
    "tab-bond-credit": {
        "title": "BOND CREDIT & STATUS",
        "description": "Analysis of the bond's default risk and its current market pricing status.",
        "metrics": [
            {"metric": "Credit Rating (S&P/Moody's/Fitch)", "definition": "The solvency rating (e.g., AAA, BBB-) assigned by major rating agencies."},
            {"metric": "Status (Premium/Par/Discount)", "definition": "Determines if the bond's price is above, equal to, or below its Par Value ($100)."},
        ]
    },
    # [NEW TAB 2: DURATION & RISK]
    "tab-bond-risk": {
        "title": "BOND DURATION & RISK",
        "description": "Measures of the bond's price sensitivity to interest rate changes.",
        "metrics": [
            {"metric": "Duration (Modified)", "definition": "Measures a bond's price sensitivity to changes in interest rates. A higher Duration implies higher volatility for a bond's price."},
            {"metric": "Convexity", "definition": "Measures the rate of change of the bond's Duration. It refines the Duration estimate."},
            {"metric": "Valuation Spread (%)", "definition": "The percentage difference between the Intrinsic Value and the current market Dirty Price."},
        ]
    },
    # [NEW TAB 3: YIELD & PRICING]
    "tab-bond-pricing": {
        "title": "BOND YIELD & PRICING",
        "description": "Key measures of income, expected total return, and price components.",
        "metrics": [
            {"metric": "Coupon Rate (%)", "definition": "The annual interest rate paid by the bond issuer."},
            {"metric": "YTM (%)", "definition": "The total anticipated return if the bond is held until maturity."},
            {"metric": "Clean Price ($)", "definition": "The bond's price excluding any accrued interest."},
            {"metric": "Accrued Interest ($)", "definition": "The portion of the next coupon the buyer pays the seller."},
            {"metric": "Dirty Price ($)", "definition": "The actual market price including accrued interest (Clean Price + Accrued Interest)."},
        ]
    },
}

# --- Shared Components ---
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
    
    # ดึงค่าล่าสุดจาก BOND_YIELD_MAP และ BOND_BENCHMARK_MAP 
    yield_options = [{'label': v, 'value': k} for k, v in BOND_YIELD_MAP.items()]
    benchmark_options = [{'label': v, 'value': k} for k, v in BOND_BENCHMARK_MAP.items()]
    
    return dbc.Container(
        [
            # --- Hidden Data Stores ---
            # [MODIFIED] Set default to Yield Curve components
            dcc.Store(id='bonds-user-selections-store', data={'tickers': ['^TNX', '^TWS', '^TYX'], 'indices': ['^GSPC']}),
            # [NEW] Store the current exchange/rate environment
            dcc.Store(id='bonds-fx-rates-store', data={'THB_RATE_1Y': 0.025, 'USD_RATE_1Y': 0.055}), 
            dcc.Store(id='bonds-forecast-assumptions-store', data={}),
            dcc.Store(id='bonds-dcf-assumptions-store', data={}),
            html.Div(id="navbar-container"), 
            
            # [NEW] Definition Modal
            definitions_modal,

            dbc.Row(
                [
                    # --- Left Sidebar (Controls) ---
                    dbc.Col(
                        dbc.Card(dbc.CardBody([
                            html.Label("Add Yields to Analysis", className="fw-bold"),
                            # Dropdown นี้ถูกใช้แค่เป็น Trigger ใน Callback
                            dcc.Dropdown(
                                id='bonds-yield-type-dropdown',
                                options=yield_options, 
                                value=list(BOND_YIELD_MAP.keys())[0],
                                clearable=False,
                                className="mb-2"
                            ),
                            dcc.Dropdown(
                                id='bonds-yield-select-dropdown',
                                options=yield_options,
                                placeholder="Select one or more yields...",
                                multi=True, 
                                className="mt-2 sidebar-dropdown"
                            ),
                            dbc.Button([html.I(className="bi bi-plus-circle-fill me-2"), "Add Yield(s)"], id='bonds-add-yield-button', color="primary", className="mt-2 w-100", n_clicks=0),
                            
                            html.Hr(),

                            html.Label("Add Benchmarks to Compare", className="fw-bold"),
                            dcc.Dropdown(
                                id='bonds-benchmark-select-dropdown',
                                options=benchmark_options,
                                placeholder="Select one or more benchmarks...",
                                multi=True, 
                                className="sidebar-dropdown"
                            ),
                            dbc.Button([html.I(className="bi bi-plus-circle-fill me-2"), "Add Benchmark(s)"], id='bonds-add-benchmark-button', color="primary", className="mt-2 w-100", n_clicks=0),

                            html.Hr(className="my-4"),
                            
                            # --- [NEW] THB Hedged View Toggle ---
                            html.Label("VIEW OPTIONS", className="fw-bold"),
                            dbc.Checklist(
                                options=[{"label": "Show THB Hedged Yields (THB View)", "value": True}],
                                value=[],
                                id="bonds-thb-view-toggle",
                                switch=True,
                                className="mb-3"
                            ),
                            # --- END NEW ---

                            html.Div(id='bonds-summary-display', className="mb-2"),
                            html.Div(id='bonds-benchmark-summary-display', className="pt-0"),
                            
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
                                                dbc.Tab(label="YIELD VOLATILITY", tab_id="tab-yield-volatility"),
                                            ])
                                        ]),
                                        md=8
                                    ),
                                    dbc.Col(
                                        dbc.Stack(
                                            [
                                                dbc.Button(html.I(className="bi bi-gear-fill"), id="bonds-open-dcf-modal-btn", color="secondary", outline=True, style={'display': 'none'}),
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
                                                # --- [MODIFIED: Split INDIVIDUAL METRICS into 3 tabs] ---
                                                dbc.Tab(label="CREDIT & STATUS", tab_id="tab-bond-credit"),
                                                dbc.Tab(label="DURATION & RISK", tab_id="tab-bond-risk"),
                                                dbc.Tab(label="YIELD & PRICING", tab_id="tab-bond-pricing"),
                                                # --- [END MODIFIED] ---
                                            ])
                                        ]),
                                        md=7
                                    ),
                                     dbc.Col(
                                        dbc.Stack(
                                            [
                                                dbc.Button(html.I(className="bi bi-gear-fill"), id="bonds-open-forecast-modal-btn", color="secondary", outline=True, style={'display': 'none'}),
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
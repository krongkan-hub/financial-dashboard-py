# layout.py (Responsive Version with Monte Carlo DCF Modal & MathJax Fix & Smart Peer Finder UI)
# [MODIFIED] Changed "Add Selected Peer(s)" button to match "Add Stock(s)" button style

from dash import dcc, html
import dash_bootstrap_components as dbc
from flask_login import current_user
from ..constants import SECTORS, TOP_5_DEFAULT_TICKERS # [MODIFIED] Added constants
from .auth import create_login_modal
from app.web.pages.stocks_services import generate_ytd_performance_figure # [NEW] Import Service

# --- Dictionary for Metric Definitions (FIXED) ---
METRIC_DEFINITIONS = {
    # Graph Definitions
    "tab-performance": dcc.Markdown("""
    **PERFORMANCE (Year-to-Date)**
    * **Definition:** This chart displays the percentage change in the price of selected stocks and benchmarks from the first trading day of the current year to the present day. It helps visualize and compare the relative performance of different assets over the same period.
    * **Calculation:**
    $$
    Performance = \\Big( \\frac{\\text{Current Price}}{\\text{Price at Start of Year}} - 1 \\Big) \\times 100
    $$
    * **Interpretation:** A rising line indicates positive returns (the asset has increased in value), while a falling line indicates negative returns. The steeper the line, the more significant the price change.
    """, mathjax=True),
    "tab-drawdown": dcc.Markdown("""
    **DRAWDOWN (1-Year)**
    * **Definition:** A drawdown is the peak-to-trough decline during a specific period for an investment. This chart shows the percentage loss from the most recent highest point (peak) over the last year.
    * **Calculation:**
    $$
    Drawdown = \\Big( \\frac{\\text{Current Price}}{\\text{Rolling 1-Year Max Price}} - 1 \\Big) \\times 100
    $$
    * **Interpretation:** This chart is a key indicator of risk. A larger negative value (e.g., -25%) indicates a greater loss from its peak, suggesting higher volatility or a potential downturn. It helps assess how much an asset has fallen from its high.
    """, mathjax=True),
    "tab-scatter": dcc.Markdown("""
    **VALUATION VS. QUALITY**
    * **Definition:** This scatter plot positions companies on a 2x2 grid to compare their valuation against their operational quality.
    * **Axes:**
        * **X-Axis (Quality):** `EBITDA Margin`. A measure of a company's operating profitability as a percentage of its revenue. Higher is generally better.
        * **Y-Axis (Valuation):** `EV/EBITDA`. A ratio comparing a company's total value (Enterprise Value) to its earnings before interest, taxes, depreciation, and amortization. Lower is generally considered cheaper or more attractive.
    * **Interpretation:**
        * **Top-Left:** Expensive valuation, low quality.
        * **Top-Right:** Expensive valuation, high quality (e.g., growth stocks).
        * **Bottom-Left:** Cheap valuation, low quality (e.g., potential value traps).
        * **Bottom-Right:** Cheap valuation, high quality (e.g., ideal value investments).
    """, mathjax=True),
    "tab-dcf": dcc.Markdown("""
    **MARGIN OF SAFETY (MONTE CARLO DCF)**
    * **Definition:** This chart visualizes the distribution of possible intrinsic values for a stock, calculated using a Discounted Cash Flow (DCF) model combined with Monte Carlo simulation. Instead of single-point estimates, this approach uses a range of probable inputs to reflect real-world uncertainty.

    * **Core Calculation (Simplified):**
        1.  **Calculate Free Cash Flow to the Firm (FCFF):**
            $$
            FCFF = (EBIT \\times (1 - Tax Rate)) + D\\&A - CapEx
            $$
        2.  **Calculate Terminal Value (TV):** (The value of the company beyond the forecast period)
            $$
            TV = \\frac{\\text{Final Year FCFF} \\times (1 + g)}{WACC - g}
            $$
            (Where: $g$ = Perpetual Growth Rate, $WACC$ = Discount Rate)
        3.  **Calculate Intrinsic Value:**
            $$
            Value = \\frac{\\sum(\\text{Discounted Future FCFFs}) + \\text{Discounted TV} - \\text{Net Debt}}{\\text{Shares Outstanding}}
            $$

    * **Simulation Process:**
        1.  **Define Assumption Ranges:** You provide a range (Minimum, Most Likely, Maximum) for key drivers: `Forecast Growth Rate (g)`, `Perpetual Growth Rate (g)`, and `Discount Rate (WACC)`.
        2.  **Run Simulations:** The model runs thousands of DCF calculations. In each run, it randomly picks a value for each assumption from within your defined ranges, using a triangular distribution.
        3.  **Generate Distribution:** The output is a histogram showing the frequency of each calculated intrinsic value. This distribution represents the range of probable fair values for the stock.
    * **Interpretation:**
        * **The Histogram:** Shows which intrinsic value outcomes are most likely. A taller bar means a higher probability.
        * **Mean Value (Red Line):** The average of all simulation outcomes. It serves as a central point of the valuation.
        * **Current Price (Black Line):** The stock's current market price.
        * **Probability Analysis:** By comparing the distribution to the current price, you can assess the probability that the stock is undervalued (i.e., the percentage of simulation outcomes where the intrinsic value is higher than the current price).
    **A limitation to note:** This model is less reliable for early-stage companies (Startups) or rapidly growing companies, as these companies often have consistently negative Free Cash Flow during the expansion phase.
    """, mathjax=True),

    # --- NEW TAB DEFINITIONS ---
    "tab-historical": dcc.Markdown("""
    **HISTORICAL VALUATION BANDS**
    * **Definition:** Plots historical price vs. P/E bands (15x, 20x, 25x).
    * **Purpose:** visuals valuation trends relative to earnings power. Green=15x (Value), Yellow=20x (Moderate), Red=25x (Premium).
    """, mathjax=True),
    "tab-health": dcc.Markdown("""
    **FINANCIAL HEALTH**
    * **Definition:** Assesses solvency (long-term) and liquidity (short-term) stability.
    """, mathjax=True),
    "tab-analyst": dcc.Markdown("""
    **ANALYST CONSENSUS**
    * **Definition:** Aggregates Wall Street analyst ratings and target prices to gauge sentiment.
    """, mathjax=True),

    # Valuation Metrics
    "P/E": dcc.Markdown("""
    **P/E (Price-to-Earnings) Ratio**
    * **Definition:** A valuation ratio that compares a company's current share price to its per-share earnings.
    * **Formula:**
    $$
    P/E\\ Ratio = \\frac{\\text{Market Price per Share}}{\\text{Earnings per Share (EPS)}}
    $$
    * **Interpretation:** A high P/E can suggest a stock is overvalued or that investors expect high future growth. A low P/E might indicate it's undervalued.
    """, mathjax=True),
    "P/B": dcc.Markdown("""
    **P/B (Price-to-Book) Ratio**
    * **Definition:** Compares a company's market capitalization to its book value.
    * **Formula:**
    $$
    P/B\\ Ratio = \\frac{\\text{Market Price per Share}}{\\text{Book Value per Share}}
    $$
    * **Interpretation:** A ratio under 1.0 is often considered a sign of an undervalued stock. It is most useful for asset-heavy companies.
    """, mathjax=True),
    "EV/EBITDA": dcc.Markdown("""
    **EV/EBITDA (Enterprise Value-to-EBITDA) Ratio**
    * **Definition:** Compares the total value of a company (including debt) to its cash earnings. It is often considered more comprehensive than P/E.
    * **Formula:**
    $$
    EV/EBITDA = \\frac{\\text{Enterprise Value}}{\\text{EBITDA}}
    $$
    * **Interpretation:** A low EV/EBITDA ratio suggests that a company might be undervalued.
    """, mathjax=True),

    # Growth Metrics
    "Revenue Growth (YoY)": dcc.Markdown("""
    **Revenue Growth (Year-over-Year)**
    * **Definition:** Measures the percentage increase in a company's revenue over the most recent year.
    * **Formula:**
    $$
    YoY\\ Growth = \\Big( \\frac{\\text{Current Period Revenue}}{\\text{Prior Period Revenue}} - 1 \\Big) \\times 100
    $$
    * **Interpretation:** Indicates the pace at which a company's sales are growing.
    """, mathjax=True),
    "Revenue CAGR (3Y)": dcc.Markdown("""
    **Revenue CAGR (3-Year)**
    * **Definition:** The Compound Annual Growth Rate of revenue over a three-year period, providing a smoothed growth rate.
    * **Formula:**
    $$
    CAGR = \\bigg( \\Big( \\frac{\\text{Ending Value}}{\\text{Beginning Value}} \\Big)^{\\frac{1}{\\text{No. of Years}}} - 1 \\bigg) \\times 100
    $$
    * **Interpretation:** A more stable indicator of long-term growth trends.
    """, mathjax=True),
    "Net Income Growth (YoY)": dcc.Markdown("""
    **Net Income Growth (Year-over-Year)**
    * **Definition:** Measures the percentage increase in a company's net profit over the past year.
    * **Formula:**
    $$
    YoY\\ Growth = \\Big( \\frac{\\text{Current Period Net Income}}{\\text{Prior Period Net Income}} - 1 \\Big) \\times 100
    $$
    * **Interpretation:** Shows how effectively a company is translating revenue into actual profit.
    """, mathjax=True),

    # Fundamentals Metrics
    "Operating Margin": dcc.Markdown("""
    **Operating Margin**
    * **Definition:** Measures how much profit a company makes on a dollar of sales from its core operations.
    * **Formula:**
    $$
    Operating\\ Margin = \\frac{\\text{Operating Income}}{\\text{Revenue}} \\times 100
    $$
    * **Interpretation:** A higher operating margin indicates greater efficiency.
    """, mathjax=True),
    "ROE": dcc.Markdown("""
    **ROE (Return on Equity)**
    * **Definition:** A measure of how effectively a company uses shareholder investments to generate profits.
    * **Formula:**
    $$
    ROE = \\frac{\\text{Net Income}}{\\text{Average Shareholder's Equity}} \\times 100
    $$
    * **Interpretation:** A consistently high ROE can be a sign of a strong competitive advantage.
    """, mathjax=True),
    "D/E Ratio": dcc.Markdown("""
    **D/E (Debt-to-Equity) Ratio**
    * **Definition:** A ratio that evaluates a company's financial leverage.
    * **Formula:**
    $$
    D/E\\ Ratio = \\frac{\\text{Total Debt}}{\\text{Total Shareholder's Equity}}
    $$
    * **Interpretation:** A high D/E ratio can mean aggressive financing with debt, which can increase risk. What is considered "high" varies by industry.
    """, mathjax=True),
    "Cash Conversion": dcc.Markdown("""
    **Cash Conversion**
    * **Definition:** Measures how efficiently a company converts its net income into operating cash flow.
    * **Formula:**
    $$
    Cash\\ Conversion = \\frac{\\text{Operating Cash Flow}}{\\text{Net Income}}
    $$
    * **Interpretation:** A ratio greater than 1.0 is generally strong, indicating the company generates more cash than its reported profit.
    """, mathjax=True),

    # Target/Forecast Metrics
    "Target Price": dcc.Markdown("""
    **Target Price**
    * **Definition:** The projected future price of a stock based on the assumptions entered in the 'Forecast Assumptions' modal.
    * **Formula:**
    $$
    Target\\ Price = \\text{Current EPS} \\times (1 + \\text{EPS Growth})^{\\text{Years}} \\times \\text{Terminal P/E}
    $$
    * **Interpretation:** This is an estimated future value, not a guarantee. Its accuracy depends on the validity of the assumptions.
    """, mathjax=True),
    "Target Upside": dcc.Markdown("""
    **Target Upside**
    * **Definition:** The potential percentage return if the stock reaches its calculated Target Price.
    * **Formula:**
    $$
    Target\\ Upside = \\Big( \\frac{\\text{Target Price}}{\\text{Current Price}} - 1 \\Big) \\times 100
    $$
    * **Interpretation:** It quantifies the potential reward based on your forecast.
    """, mathjax=True),
    "IRR %": dcc.Markdown("""
    **IRR (Internal Rate of Return) %**
    * **Definition:** The projected compound annual growth rate (CAGR) of an investment if it moves from the current price to the target price over the forecast period.
    * **Formula:**
    $$
    IRR = \\bigg( \\Big( \\frac{\\text{Target Price}}{\\text{Current Price}} \\Big)^{\\frac{1}{\\text{Forecast Years}}} - 1 \\bigg) \\times 100
    $$
    * **Interpretation:** Represents the annualized rate of return for the investment scenario, useful for comparing opportunities.
    """, mathjax=True),

    # --- Financial Health Metrics ---
    "Net Debt/EBITDA": dcc.Markdown("""
    **Net Debt / EBITDA**
    * **Definition:** Measures leverage. How many years of earnings (EBITDA) would it take to pay off net debt.
    * **Target:** Generally < 3.0x is healthy. High values indicate high leverage.
    """, mathjax=True),
    "Interest Coverage": dcc.Markdown("""
    **Interest Coverage Ratio**
    * **Definition:** Ability to pay interest on outstanding debt.
    * **Formula:** `EBIT / Interest Expense`
    * **Target:** > 3.0x is generally safe. < 1.5x is risky.
    """, mathjax=True),
    "Current Ratio": dcc.Markdown("""
    **Current Ratio**
    * **Definition:** Ability to pay short-term obligations with short-term assets.
    * **Formula:** `Current Assets / Current Liabilities`
    * **Target:** > 1.0 is essential. > 1.5 is healthy.
    """, mathjax=True),
    "D/E Ratio": dcc.Markdown("""
    **Debt-to-Equity Ratio**
    * **Definition:** Relative proportion of shareholder equity and debt used to finance a company's assets.
    """, mathjax=True),

    # --- Analyst Metrics ---
    "Consensus": dcc.Markdown("""
    **Analyst Consensus**
    * **Definition:** The prevailing buy/hold/sell recommendation from analysts (e.g., "Strong Buy").
    """, mathjax=True),
    "Upside %": dcc.Markdown("""
    **Analyst Target Upside**
    * **Definition:** The percentage difference between the average analyst price target and the current price.
    * **Formula:** `(Target Price / Current Price) - 1`
    """, mathjax=True),
}


def create_navbar():
    """
    Creates the main navbar with:
    1. Logo/Brand on the left.
    2. Centered navigation icons (Stocks, Bonds, Derivatives) like Facebook.
    3. Login/Logout button on the far right.
    """
    # 1. Setup Login/Logout Button
    if current_user.is_authenticated:
        login_button = dbc.Button("Logout", href="/logout", color="secondary", external_link=True)
    else:
        login_button = dbc.Button("Login", id="open-login-modal-button", color="primary")

    # 2. [NEW] Create centered navigation icons (like Facebook)
    nav_links = dbc.Nav(
        [
            # Stocks (Main Dashboard)
            dbc.NavItem(
                dbc.NavLink(html.I(className="bi bi-graph-up fs-4"), href="/", id="nav-stocks-link", active="exact")
            ),
            dbc.Tooltip("Stocks", target="nav-stocks-link", placement="bottom"),

            # Bonds (New Page)
            dbc.NavItem(
                dbc.NavLink(html.I(className="bi bi-file-text fs-4"), href="/bonds", id="nav-bonds-link", active="exact")
            ),
            dbc.Tooltip("Bonds", target="nav-bonds-link", placement="bottom"),

            # Derivatives (New Page)
            dbc.NavItem(
                dbc.NavLink(html.I(className="bi bi-layers fs-4"), href="/derivatives", id="nav-derivatives-link", active="exact")
            ),
            dbc.Tooltip("Derivatives", target="nav-derivatives-link", placement="bottom"),
        ],
        className="mx-auto", # "mx-auto" centers the Nav component in flexbox
        navbar=True
    )

    # 3. Assemble the Navbar
    return dbc.Navbar(
        dbc.Container(
            [
                # Left: Brand/Logo
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

                # Center: New Nav Links
                nav_links,

                # Right: Login/Logout Button
                dbc.Stack(login_button, direction="horizontal", className="ms-auto") # ms-auto pushes this to the right
            ],
            fluid=True
        ),
        # color="dark",  <-- REMOVED: conflicting with custom CSS background-color
        dark=True,       # KEPT: Helper to make text white (suitable for dark purple background)
        className="py-2 fixed-top"
    )

def create_forecast_modal():
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

def create_dcf_modal():
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("DCF MONTE CARLO ASSUMPTIONS")),
            dbc.ModalBody([
                dbc.Label("Number of Simulations:", className="fw-bold"),
                dbc.Input(id="mc-dcf-simulations-input", type="number", value=10000, min=1000, max=50000, step=1000, className="mb-4"),

                dbc.Label("Forecast Growth Rate (%)", className="fw-bold"),
                dbc.Row([
                    dbc.Col(dbc.Input(id="mc-dcf-growth-min", type="number", placeholder="Min", value=3)),
                    dbc.Col(dbc.Input(id="mc-dcf-growth-mode", type="number", placeholder="Most Likely", value=5)),
                    dbc.Col(dbc.Input(id="mc-dcf-growth-max", type="number", placeholder="Max", value=8)),
                ], className="mb-4"),

                dbc.Label("Perpetual Growth Rate (%)", className="fw-bold"),
                dbc.Row([
                    dbc.Col(dbc.Input(id="mc-dcf-perpetual-min", type="number", placeholder="Min", value=1.5)),
                    dbc.Col(dbc.Input(id="mc-dcf-perpetual-mode", type="number", placeholder="Most Likely", value=2.5)),
                    dbc.Col(dbc.Input(id="mc-dcf-perpetual-max", type="number", placeholder="Max", value=3.0)),
                ], className="mb-4"),

                dbc.Label("Discount Rate (WACC) (%)", className="fw-bold"),
                dbc.Row([
                    dbc.Col(dbc.Input(id="mc-dcf-wacc-min", type="number", placeholder="Min", value=7.0)),
                    dbc.Col(dbc.Input(id="mc-dcf-wacc-mode", type="number", placeholder="Most Likely", value=8.0)),
                    dbc.Col(dbc.Input(id="mc-dcf-wacc-max", type="number", placeholder="Max", value=10.0)),
                ]),
                 html.P("These ranges will be used for the simulation.", className="text-muted small mt-3")
            ]),
            dbc.ModalFooter(
                dbc.Button("Run Simulation", id="apply-dcf-changes-btn", color="primary")
            ),
        ],
        id="dcf-assumptions-modal",
        is_open=False,
    )

def create_definitions_modal():
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
    """Builds the main dashboard layout."""
    return html.Div([
        dcc.Store(id='user-selections-store', storage_type='memory'),
        dcc.Store(id='forecast-assumptions-store', storage_type='memory'),
        dcc.Store(id='dcf-assumptions-store', storage_type='memory'),
        
        dbc.Container([
            dbc.Row([
                # --- [SIDEBAR] ---
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("STOCK COMPARISON", className="fw-bold mb-3 text-light"),
                        
                        # 1. Stock Selection
                        html.Label("Filter by Sector", className="small text-light fw-bold"),
                        dcc.Dropdown(
                            id='sector-dropdown',
                            options=[{'label': 'All Sectors', 'value': 'All'}] + [{'label': k, 'value': k} for k in SECTORS.keys()],
                            value='All',
                            clearable=False,
                            className="mb-2"
                        ),
                        
                        html.Label("Select Ticker(s)", className="small text-light fw-bold"),
                        dcc.Dropdown(id='ticker-select-dropdown', className="sidebar-dropdown mb-2", placeholder="Search ticker...", multi=True),
                        dbc.Button([html.I(className="bi bi-plus-lg me-2"), "Add Stock"], id="add-ticker-button", n_clicks=0, color="primary", className="w-100 mb-3 rounded-2"),
                        
                        html.Div(id='ticker-summary-display', className="mb-4"),

                        html.Hr(),

                        # 2. Benchmark Selection
                        html.Label("Compare vs Benchmark", className="small text-light fw-bold"),
                        dcc.Dropdown(id='index-select-dropdown', placeholder="Select benchmark...", multi=True, className="mb-2"),
                        dbc.Button([html.I(className="bi bi-graph-up me-2"), "Add Benchmark"], id="add-index-button", n_clicks=0, color="primary", className="w-100 mb-3 rounded-2"),
                        
                        html.Div(id='index-summary-display')

                    ], className="p-3")  
                ], className="border-0 shadow-sm"), width=12, md=3, className="mb-4 mb-md-0 sidebar-fixed", align="center"),

                # --- [MAIN CONTENT] ---
                dbc.Col([
                    # Control Bar (Top Right Tools)
                    dbc.Row([
                        dbc.Col(
                            html.H2("Market Overview", className="mb-0"), 
                            width=True, 
                            align="center"
                        ),
                        dbc.Col(
                            dbc.Stack([
                                dbc.Button([html.I(className="bi bi-gear-fill me-2"), "Forecast"], id="open-forecast-modal-btn", color="light", size="sm", className="text-secondary fw-bold"),
                                dbc.Button([html.I(className="bi bi-magic me-2"), "DCF Settings"], id="open-dcf-modal-btn", color="light", size="sm", className="text-secondary fw-bold"),
                                dcc.Dropdown(id='sort-by-dropdown', placeholder="Sort Table by...", style={'minWidth': '150px'}, className="small")
                            ], direction="horizontal", gap=2),
                            width="auto",
                            align="center"
                        )
                    ], className="mb-3"),

                    # Top Level Tabs
                    dbc.Tabs(id="top-level-tabs", active_tab="tab-valuation", children=[
                        
                        # TAB 1: OVERVIEW
                        dbc.Tab(label="OVERVIEW", tab_id="tab-valuation", children=[
                            dbc.Card(dbc.CardBody([
                                html.H5("Year-to-Date Performance", className="mb-3 text-light fw-bold"),
                                html.Div(id='content-performance', children=[dcc.Loading(html.Div())]),
                                html.Hr(className="my-4"),
                                html.H5("Comparables: Valuation Metrics", className="mb-3 text-light fw-bold"),
                                html.Div(id='content-valuation', children=[dcc.Loading(html.Div())])
                            ]), className="border-top-0 rounded-bottom shadow-sm")
                        ]),

                        # TAB 2: GROWTH
                        dbc.Tab(label="GROWTH", tab_id="tab-growth", children=[
                            dbc.Card(dbc.CardBody([
                                html.H5("Financial Trends: Revenue & Earnings", className="mb-3 text-light fw-bold"),
                                html.Div(id='content-historical', children=[dcc.Loading(html.Div())]),
                                html.Hr(className="my-4"),
                                html.H5("Comparables: Growth Metrics", className="mb-3 text-light fw-bold"),
                                html.Div(id='content-growth', children=[dcc.Loading(html.Div())])
                            ]), className="border-top-0 rounded-bottom shadow-sm")
                        ]),

                        # TAB 3: QUALITY
                        dbc.Tab(label="QUALITY", tab_id="tab-fundamentals", children=[
                             dbc.Card(dbc.CardBody([
                                html.H5("Quality vs. Valuation (Scatter)", className="mb-3 text-light fw-bold"),
                                html.Div(id='content-scatter', children=[dcc.Loading(html.Div())]),
                                html.Hr(className="my-4"),
                                html.H5("Comparables: Fundamental Metrics", className="mb-3 text-light fw-bold"),
                                html.Div(id='content-fundamentals', children=[dcc.Loading(html.Div())])
                            ]), className="border-top-0 rounded-bottom shadow-sm")
                        ]),

                        # TAB 4: FINANCIAL HEALTH
                        dbc.Tab(label="HEALTH", tab_id="tab-health", children=[
                             dbc.Card(dbc.CardBody([
                                html.H5("Risk Analysis: Max Drawdown (1Y)", className="mb-3 text-light fw-bold"),
                                html.Div(id='content-drawdown', children=[dcc.Loading(html.Div())]),
                                html.Hr(className="my-4"),
                                html.H5("Comparables: Financial Health", className="mb-3 text-light fw-bold"),
                                html.Div(id='content-health', children=[dcc.Loading(html.Div())])
                            ]), className="border-top-0 rounded-bottom shadow-sm")
                        ]),

                        # TAB 5: CONSENSUS
                        dbc.Tab(label="CONSENSUS", tab_id="tab-analyst", children=[
                            dbc.Card(dbc.CardBody([
                                html.H5("Intrinsic Value Distribution (Monte Carlo DCF)", className="mb-3 text-light fw-bold"),
                                html.Div(id='content-dcf', children=[dcc.Loading(html.Div())]),
                                html.Hr(className="my-4"),
                                html.H5("Comparables: Analyst Ratings", className="mb-3 text-light fw-bold"),
                                html.Div(id='content-analyst', children=[dcc.Loading(html.Div())])
                            ]), className="border-top-0 rounded-bottom shadow-sm")
                        ]),

                    ], className="nav-fill fw-bold checkbox-style-tabs"),

                    # Footer / Info Buttons (Hidden but kept for ID preservation if needed, or moved above)
                    # Note: I moved the settings buttons to the top right.
                    # 'open-definitions-modal-btn-graphs' and 'tables' are missing. I should add them back somewhere or consolidate.
                    html.Div([
                        dbc.Button(html.I(className="bi bi-info-circle"), id="open-definitions-modal-btn-graphs", color="link", className="text-muted"),
                        dbc.Button(html.I(className="bi bi-info-circle"), id="open-definitions-modal-btn-tables", color="link", className="text-muted", style={'display': 'none'}) # Hide duplicate
                    ], className="text-end mt-2")


                ], width=12, md=9, className="content-offset"),
            ], className="g-4")
        ], fluid=True, className="p-4 main-bg"),

        create_login_modal(),
        create_forecast_modal(),
        create_dcf_modal(),
        create_definitions_modal()
    ])
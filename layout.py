# layout.py (Responsive Version with Monte Carlo DCF Modal & MathJax Fix)

from dash import dcc, html
import dash_bootstrap_components as dbc
from flask_login import current_user
from constants import SECTORS

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
    from auth import create_login_modal
    return html.Div([
        dcc.Store(id='user-selections-store', storage_type='memory'),
        dcc.Store(id='forecast-assumptions-store', storage_type='memory'),
        dcc.Store(id='dcf-assumptions-store', storage_type='memory'),
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
                    # Graph Controls Row (Responsive)
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(className="custom-tabs-container", children=[
                                    dbc.Tabs(id="analysis-tabs", active_tab="tab-performance", children=[
                                        dbc.Tab(label="PERFORMANCE", tab_id="tab-performance"),
                                        dbc.Tab(label="DRAWDOWN", tab_id="tab-drawdown"),
                                        dbc.Tab(label="VALUATION VS. QUALITY", tab_id="tab-scatter"),
                                        dbc.Tab(label="MARGIN OF SAFETY", tab_id="tab-dcf"),
                                    ])
                                ]),
                                md=8
                            ),
                            dbc.Col(
                                dbc.Stack(
                                    [
                                        dbc.Button(html.I(className="bi bi-gear-fill"), id="open-dcf-modal-btn", color="secondary", outline=True, style={'display': 'none'}),
                                        dbc.Button(html.I(className="bi bi-info-circle-fill"), id="open-definitions-modal-btn-graphs", color="secondary", outline=True),
                                    ],
                                    direction="horizontal",
                                    gap=2,
                                    className="justify-content-start justify-content-lg-end pt-2 pt-lg-0" 
                                ),
                                md=4
                            )
                        ],
                        align="center",
                        className="control-row"
                    ),
                    
                    dcc.Loading(html.Div(id='analysis-pane-content', className="mt-3")),
                    html.Hr(className="my-5"),

                    # Table Controls Row (Responsive)
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(className="custom-tabs-container", children=[
                                    dbc.Tabs(id="table-tabs", active_tab="tab-valuation", children=[
                                        dbc.Tab(label="VALUATION", tab_id="tab-valuation"),
                                        dbc.Tab(label="GROWTH", tab_id="tab-growth"),
                                        dbc.Tab(label="FUNDAMENTALS", tab_id="tab-fundamentals"),
                                        dbc.Tab(label="TARGET", tab_id="tab-forecast"),
                                    ])
                                ]),
                                md=7
                            ),
                             dbc.Col(
                                dbc.Stack(
                                    [
                                        dbc.Button(html.I(className="bi bi-gear-fill"), id="open-forecast-modal-btn", color="secondary", outline=True),
                                        dbc.Button(html.I(className="bi bi-info-circle-fill"), id="open-definitions-modal-btn-tables", color="secondary", outline=True),
                                        dcc.Dropdown(id='sort-by-dropdown', placeholder="Sort by", style={'minWidth': '180px'})
                                    ],
                                    direction="horizontal",
                                    gap=2,
                                    className="justify-content-start justify-content-lg-end pt-2 pt-lg-0"
                                ),
                                md=5
                            )
                        ],
                        align="center",
                        className="control-row mt-3"
                    ),
                    
                    dcc.Loading(html.Div(id="table-pane-content", className="mt-2"))
                ], width=12, md=9, className="content-offset"),
            ], className="g-4")
        ], fluid=True, className="p-4 main-content-container"),
        create_login_modal(),
        create_forecast_modal(),
        create_dcf_modal(),
        create_definitions_modal()
    ])
# layout.py (Final Version with Full Definitions)

from dash import dcc, html
import dash_bootstrap_components as dbc
from flask_login import current_user
from constants import SECTORS

# --- Dictionary for Metric Definitions ---
METRIC_DEFINITIONS = {
    # Valuation Metrics
    "P/E": dcc.Markdown("""
    **P/E (Price-to-Earnings) Ratio**
    * **Definition:** A valuation ratio that compares a company's current share price to its per-share earnings.
    * **Formula:**
    $$
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
    * **Formula:**
    $$
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
    * **Formula:**
    $$
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
    * **Formula:**
    $$
    YoY\\ Growth = \\Big( \\frac{\\text{Current Period Revenue}}{\\text{Prior Period Revenue}} - 1 \\Big) \\times 100
    $$
    * **Formula Components:**
        * `Current Period Revenue`: Revenue from the most recent 12-month period.
        * `Prior Period Revenue`: Revenue from the 12-month period before the current one.
    * **Interpretation:** Indicates the pace at which a company's sales are growing. Consistently high growth is a positive sign, but it's important to understand the drivers behind it.
    """, mathjax=True),
    "Revenue CAGR (3Y)": dcc.Markdown("""
    **Revenue CAGR (3-Year)**
    * **Definition:** The Compound Annual Growth Rate of revenue over a three-year period. It provides a smoothed, annualized growth rate that irons out volatility.
    * **Formula:**
    $$
    CAGR = \\bigg( \\Big( \\frac{\\text{Ending Value}}{\\text{Beginning Value}} \\Big)^{\\frac{1}{\\text{No. of Years}}} - 1 \\bigg) \\times 100
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
    * **Formula:**
    $$
    YoY\\ Growth = \\Big( \\frac{\\text{Current Period Net Income}}{\\text{Prior Period Net Income}} - 1 \\Big) \\times 100
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
    * **Formula:**
    $$
    Operating\\ Margin = \\frac{\\text{Operating Income}}{\\text{Revenue}} \\times 100
    $$
    * **Formula Components:**
        * `Operating Income`: The profit realized from a business's own, core operations.
        * `Revenue`: The total amount of income generated by the sale of goods or services.
    * **Interpretation:** A higher operating margin indicates greater efficiency in the company's core business operations. It reflects the profitability of the business before the effects of financing and taxes.
    """, mathjax=True),
    "ROE": dcc.Markdown("""
    **ROE (Return on Equity)**
    * **Definition:** A measure of financial performance calculated by dividing net income by shareholders' equity.
    * **Formula:**
    $$
    ROE = \\frac{\\text{Net Income}}{\\text{Average Shareholder's Equity}} \\times 100
    $$
    * **Formula Components:**
        * `Net Income`: The company's profit after all expenses, including taxes and interest, have been deducted.
        * `Average Shareholder's Equity`: The average value of shareholder's equity over a period (usually the beginning and ending equity divided by 2).
    * **Interpretation:** ROE is considered a gauge of a corporation's profitability and how efficiently it generates profits. A consistently high ROE can be a sign of a strong competitive advantage (a "moat").
    """, mathjax=True),
    "D/E Ratio": dcc.Markdown("""
    **D/E (Debt-to-Equity) Ratio**
    * **Definition:** A ratio used to evaluate a company's financial leverage. It is a measure of the degree to which a company is financing its operations through debt versus wholly-owned funds.
    * **Formula:**
    $$
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
    * **Formula:**
    $$
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
    * **Formula:**
    $$
    Target\\ Price = \\text{Current EPS} \\times (1 + \\text{EPS Growth})^{\\text{Years}} \\times \\text{Terminal P/E}
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
    * **Formula:**
    $$
    Target\\ Upside = \\Big( \\frac{\\text{Target Price}}{\\text{Current Price}} - 1 \\Big) \\times 100
    $$
    * **Formula Components:**
        * `Target Price`: The estimated future stock price from your forecast.
        * `Current Price`: The current market stock price.
    * **Interpretation:** It quantifies the potential reward based on your forecast. A higher upside suggests a more attractive investment, assuming the forecast is accurate.
    """, mathjax=True),
    "IRR %": dcc.Markdown("""
    **IRR (Internal Rate of Return) %**
    * **Definition:** The projected compound annual growth rate (CAGR) of an investment if it moves from the current price to the target price over the forecast period.
    * **Formula:**
    $$
    IRR = \\bigg( \\Big( \\frac{\\text{Target Price}}{\\text{Current Price}} \\Big)^{\\frac{1}{\\text{Forecast Years}}} - 1 \\bigg) \\times 100
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
                                    dbc.Tab(label="TARGET", tab_id="tab-forecast"),
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
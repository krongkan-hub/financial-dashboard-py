# pages/deep_dive.py (Corrected version)
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd

# Import the data fetching function
from data_handler import get_deep_dive_data

# This is the main layout function for the deep dive page.
# It is called by app.py when a user navigates to a deep dive URL.
def create_deep_dive_layout(ticker=None):
    if not ticker:
        return html.Div(id='deep-dive-content-container') # Return an empty container for the callback to populate

    # Fetch all data needed for this page in one go
    data = get_deep_dive_data(ticker)

    if data.get("error"):
        return dbc.Container([
            html.H1("Error", className="text-danger"),
            html.P(f"Could not retrieve data for {ticker}. Reason: {data['error']}")
        ], fluid=True, className="mt-5 text-center")

    info = data.get("info", {})
    key_stats = data.get("key_stats", {})
    financial_trends = data.get("financial_trends", pd.DataFrame())
    price_history = data.get("price_history", pd.DataFrame())

    # --- Create Figures ---
    # Financial Trends Figure
    fig_trends = go.Figure()
    fig_trends.add_trace(go.Bar(x=financial_trends.index, y=financial_trends['Revenue'], name='Revenue', marker_color='royalblue'))
    fig_trends.add_trace(go.Scatter(x=financial_trends.index, y=financial_trends['Net Income'], name='Net Income', yaxis='y2', mode='lines+markers', line=dict(color='darkorange')))
    fig_trends.update_layout(
        title=f'Annual Financial Trends for {ticker}',
        yaxis=dict(title='Revenue ($)'),
        yaxis2=dict(title='Net Income ($)', overlaying='y', side='right'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Price History Figure
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=price_history.index, y=price_history['Close'], mode='lines', name='Close Price'))
    fig_price.update_layout(title=f'5-Year Stock Price History for {ticker}', xaxis_title='Date', yaxis_title='Price ($)')

    # --- Build Layout ---
    layout = dbc.Container([
        dcc.Store(id='deep-dive-ticker-store', data={'ticker': ticker}),
        # Section 1: Company Overview
        dbc.Row([
            dbc.Col([
                html.H2(f"{info.get('longName', ticker)} ({ticker})"),
                html.P(f"Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}", className="text-muted"),
                html.P(info.get('longBusinessSummary', 'No business summary available.'), className="mt-3", style={'maxHeight': '200px', 'overflowY': 'auto'})
            ], width=12, lg=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Key Statistics"),
                    dbc.ListGroup([
                        dbc.ListGroupItem(f"Forward P/E: {key_stats.get('Forward P/E', 'N/A')}"),
                        dbc.ListGroupItem(f"PEG Ratio: {key_stats.get('PEG Ratio', 'N/A')}"),
                        dbc.ListGroupItem(f"P/S Ratio: {key_stats.get('P/S Ratio', 'N/A')}"),
                        dbc.ListGroupItem(f"Return on Equity (ROE): {key_stats.get('ROE', 'N/A')}"),
                        dbc.ListGroupItem(f"Debt to Equity: {key_stats.get('Debt to Equity', 'N/A')}"),
                        dbc.ListGroupItem(f"Dividend Yield: {key_stats.get('Dividend Yield', 'N/A')}"),
                    ], flush=True),
                ])
            ], width=12, lg=4)
        ], className="mb-4"),

        # Section 2: Snapshot Graphs
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_price))), width=12, lg=6, className="mb-3"),
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_trends))), width=12, lg=6, className="mb-3"),
        ], className="mb-4"),

        # Section 3: Detailed Financials (Tables)
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(
                        dbc.Tabs(
                            [
                                dbc.Tab(label="Income Statement", tab_id="tab-income"),
                                dbc.Tab(label="Balance Sheet", tab_id="tab-balance"),
                                dbc.Tab(label="Cash Flow", tab_id="tab-cashflow"),
                            ],
                            id="financial-statement-tabs",
                            active_tab="tab-income",
                        )
                    ),
                    dbc.CardBody(dcc.Loading(html.Div(id="financial-statement-content")))
                ]),
                width=12
            )
        ], className="mb-4"),

        # Section 4: Valuation Workspace
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Interactive DCF Model"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(dbc.Label("Your Forecast Growth Rate (%):"), width="auto"),
                            dbc.Col(dcc.Input(id='dcf-growth-rate-input', type='number', value=5, step=0.5, className="mb-2"), width="auto")
                        ], align="center"),
                        dcc.Loading(dcc.Graph(id='interactive-dcf-chart'))
                    ])
                ]),
                width=12
            )
        ])

    ], fluid=True, className="mt-4")

    return layout
# pages/deep_dive.py (New Design Version)

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd

from data_handler import get_deep_dive_data

def create_metric_card(title, value):
    """Helper function to create a small statistic card."""
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6(title, className="metric-card-title"),
                html.P(value, className="metric-card-value"),
            ]
        ),
        className="metric-card h-100" # h-100 to make cards same height
    )

def create_deep_dive_layout(ticker=None):
    if not ticker:
        return dbc.Container(html.P("Please provide a ticker by navigating from the main page."), className="p-5 text-center")

    # Fetch all data needed for the page at once
    data = get_deep_dive_data(ticker)

    if data.get("error"):
        return dbc.Container([
            html.H2("Data Error", className="text-danger"),
            html.P(f"Could not retrieve data for {ticker}. Reason: {data['error']}"),
            dcc.Link("Go back to Dashboard", href="/")
        ], fluid=True, className="mt-5 text-center")

    # --- Section 1: The Marquee ---
    marquee = dbc.Row([
        dbc.Col(
            html.Img(src=data.get('logo_url', '/assets/placeholder_logo.png'), className="company-logo"),
            width="auto", align="center"
        ),
        dbc.Col([
            html.H1(data.get('company_name', ticker), className="company-name mb-0"),
            html.P(f"{data.get('exchange', '')}: {ticker}", className="text-muted small")
        ], width=True, align="center"),
        dbc.Col([
            html.Div([
                html.H2(f"${data.get('current_price', 0):,.2f}", className="price-display mb-0"),
                html.P(
                    f"{data.get('daily_change_str', 'N/A')} ({data.get('daily_change_pct_str', 'N/A')})",
                    className=f"price-change {'price-positive' if data.get('daily_change', 0) >= 0 else 'price-negative'}"
                )
            ], className="text-end")
        ], width="auto", align="center")
    ], align="center", className="marquee-header g-3")

    # --- Section 2: At-a-Glance Dashboard ---
    key_stats = data.get("key_stats", {})
    at_a_glance = html.Div([
        dbc.Row([
            dbc.Col(create_metric_card("Market Cap", data.get('market_cap_str', 'N/A'))),
            dbc.Col(create_metric_card("P/E Ratio", key_stats.get('P/E Ratio', 'N/A'))),
            dbc.Col(create_metric_card("Forward P/E", key_stats.get('Forward P/E', 'N/A'))),
            dbc.Col(create_metric_card("PEG Ratio", key_stats.get('PEG Ratio', 'N/A'))),
            dbc.Col(create_metric_card("Dividend Yield", key_stats.get('Dividend Yield', 'N/A'))),
        ], className="g-3 mb-4"),
        dbc.Card(
            dbc.CardBody([
                html.H5("Business Summary", className="card-title"),
                html.P(data.get('business_summary', 'Business summary not available.'), className="small")
            ])
        )
    ])

    # --- Section 3: Analysis Workspace ---
    analysis_workspace = html.Div([
        dbc.Tabs(
            [
                dbc.Tab(label="CHARTS", tab_id="tab-charts-deep-dive", label_class_name="fw-bold"),
                dbc.Tab(label="FINANCIALS", tab_id="tab-financials-deep-dive", label_class_name="fw-bold"),
                dbc.Tab(label="VALUATION", tab_id="tab-valuation-deep-dive", label_class_name="fw-bold"),
            ],
            id="deep-dive-main-tabs",
            active_tab="tab-charts-deep-dive",
            className="custom-tabs-container mt-4" # Use the same class as main page
        ),
        dbc.Card(
            dbc.CardBody(
                dcc.Loading(html.Div(id="deep-dive-tab-content"))
            ),
            className="mt-3"
        )
    ])

    return dbc.Container([
        dcc.Store(id='deep-dive-ticker-store', data={'ticker': ticker}),
        marquee,
        html.Hr(),
        at_a_glance,
        analysis_workspace,
    ], fluid=True, className="p-4 main-content-container")
# pages/deep_dive.py (Final Layout Version)

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import json

from data_handler import get_deep_dive_data, calculate_dcf_intrinsic_value

def create_metric_card(title, value, className=""):
    """Helper function to create a small statistic card. Returns None if value is invalid."""
    if value is None or value == "N/A" or (isinstance(value, str) and "N/A" in value):
        return None
    return dbc.Card(dbc.CardBody([
        html.H6(title, className="metric-card-title"),
        html.P(value, className="metric-card-value"),
    ]), className=f"metric-card h-100 {className}")

def create_deep_dive_layout(ticker=None):
    if not ticker:
        return dbc.Container(html.P("Please provide a ticker by navigating from the main page."), className="p-5 text-center")

    data = get_deep_dive_data(ticker)

    if data.get("error"):
        return dbc.Container([
            html.H2(f"Data Error for {ticker}", className="text-danger mt-5"),
            html.P(f"Could not retrieve data. Reason: {data['error']}"),
            dcc.Link("Go back to Dashboard", href="/")
        ], fluid=True, className="mt-5 text-center")

    # --- Section 1: Marquee ---
    marquee = dbc.Row([
        dbc.Col(html.Img(src=data.get('logo_url', '/assets/placeholder_logo.png'), className="company-logo"), width="auto", align="center"),
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

    # --- [LAYOUT REVISED] Section 2: At-a-Glance ---
    key_stats = data.get("key_stats", {})
    target_price = data.get('target_mean_price')
    target_price_str = f"${target_price:,.2f}" if target_price is not None and pd.notna(target_price) else "N/A"
    reco_key = data.get('recommendation_key', "N/A")
    
    # Re-order the list of cards
    all_cards = [
        create_metric_card("Market Cap", data.get('market_cap_str')),
        create_metric_card("Analyst Target", target_price_str, "bg-light-subtle"),
        create_metric_card("Recommendation", reco_key, "bg-light-subtle"),
        create_metric_card("P/E Ratio", key_stats.get('P/E Ratio')),
        create_metric_card("Forward P/E", key_stats.get('Forward P/E')),
        create_metric_card("Dividend Yield", key_stats.get('Dividend Yield')),
        create_metric_card("PEG Ratio", key_stats.get('PEG Ratio')),
    ]
    cards_to_show = [card for card in all_cards if card is not None]

    at_a_glance = html.Div([
        # Change column width from lg=3 to lg=2
        dbc.Row([dbc.Col(card, width=6, lg=2, className="mb-3") for card in cards_to_show], className="g-3"),
        dbc.Card(dbc.CardBody([
            html.H5("Business Summary", className="card-title"),
            html.P(data.get('business_summary', 'Business summary not available.'), className="small")
        ]))
    ])

    # --- Section 3: Analysis Workspace ---
    analysis_workspace = html.Div([
        dbc.Tabs(
            [
                dbc.Tab(label="CHARTS", tab_id="tab-charts-deep-dive", label_class_name="fw-bold"),
                dbc.Tab(label="FINANCIALS", tab_id="tab-financials-deep-dive", label_class_name="fw-bold"),
                dbc.Tab(label="VALUATION", tab_id="tab-valuation-deep-dive", label_class_name="fw-bold"),
                dbc.Tab(label="NEWS", tab_id="tab-news-deep-dive", label_class_name="fw-bold"),
            ],
            id="deep-dive-main-tabs", active_tab="tab-charts-deep-dive", className="custom-tabs-container mt-4"
        ),
        dbc.Card(dbc.CardBody(dcc.Loading(html.Div(id="deep-dive-tab-content"))), className="mt-3")
    ])
    
    # --- DEBUG ACCORDION ---
    debug_section = dbc.Accordion([
        dbc.AccordionItem(
            html.Pre(json.dumps({k: str(v) for k, v in data.items()}, indent=2)),
            title="Click to see Raw Data from yfinance (for debugging)"
        )
    ], start_collapsed=True, className="mt-5")

    return dbc.Container([
        dcc.Store(id='deep-dive-ticker-store', data={'ticker': ticker}),
        marquee,
        html.Hr(),
        at_a_glance,
        analysis_workspace,
        debug_section
    ], fluid=True, className="p-4 main-content-container")

# ==================================================================
# Callbacks for Deep Dive Page
# ==================================================================
@dash.callback(
    Output('deep-dive-tab-content', 'children'),
    Input('deep-dive-main-tabs', 'active_tab'),
    State('deep-dive-ticker-store', 'data')
)
def render_deep_dive_tab_content(active_tab, store_data):
    if not store_data or not store_data.get('ticker'): return ""
    ticker = store_data['ticker']
    data = get_deep_dive_data(ticker)

    if active_tab == "tab-charts-deep-dive":
        financial_trends = data.get("financial_trends", pd.DataFrame())
        margin_trends = data.get("margin_trends", pd.DataFrame())

        fig_trends = go.Figure()
        if not financial_trends.empty:
            fig_trends.add_trace(go.Bar(x=financial_trends.index, y=financial_trends.get('Revenue'), name='Revenue', marker_color='royalblue'))
            fig_trends.add_trace(go.Scatter(x=financial_trends.index, y=financial_trends.get('Net Income'), name='Net Income', yaxis='y2', mode='lines+markers', line=dict(color='darkorange')))
        fig_trends.update_layout(title='Annual Financial Trends', yaxis_title='Revenue ($)',
            yaxis2=dict(title='Net Income ($)', overlaying='y', side='right', showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(t=50))
        
        if not margin_trends.empty:
            fig_margins = px.line(margin_trends, markers=True, title="Annual Profitability Margins")
            fig_margins.update_layout(yaxis_tickformat=".2%", legend_title_text='Margin', yaxis_title='Percentage',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(t=50))
            margin_graph = dcc.Graph(figure=fig_margins)
        else:
            margin_graph = dbc.Alert("Margin data not available.", color="info", className="h-100 d-flex align-items-center justify-content-center")

        return dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_trends), md=6),
            dbc.Col(margin_graph, md=6),
        ], className="mt-3")

    elif active_tab == "tab-financials-deep-dive":
        return html.Div([
            dbc.Tabs(id="financial-statement-tabs", active_tab="tab-income", children=[
                dbc.Tab(label="Income Statement", tab_id="tab-income"),
                dbc.Tab(label="Balance Sheet", tab_id="tab-balance"),
                dbc.Tab(label="Cash Flow", tab_id="tab-cashflow"),
            ], className="mt-3"),
            dcc.Loading(html.Div(id="financial-statement-content", className="mt-3 table-responsive"))
        ])

    elif active_tab == "tab-valuation-deep-dive":
        return dbc.Row([
            dbc.Col([
                html.H5("DCF Assumptions", className="mt-3"), html.Hr(),
                dbc.Label("Your Forecast Growth Rate (%):", html_for="dcf-growth-rate-input-deep-dive"),
                dcc.Input(id='dcf-growth-rate-input-deep-dive', type='number', value=5, step=0.5, className="mb-2 form-control"),
                html.P("This model uses a standard WACC calculation and a perpetual growth rate of 2.5%.", className="text-muted small")
            ], md=4),
            dbc.Col(dcc.Loading(dcc.Graph(id='interactive-dcf-chart-deep-dive')), md=8)
        ], className="mt-3", align="center")

    elif active_tab == "tab-news-deep-dive":
        news_items = data.get("news", [])
        if not news_items:
            return dbc.Alert("No recent news found for this ticker.", color="info", className="mt-3")
        
        cards = [dbc.Card(dbc.CardBody([
            html.H5(html.A(item['title'], href=item['link'], target="_blank", className="stretched-link", style={'textDecoration':'none'})),
            html.P(f"{item.get('publisher', 'N/A')} - {item['providerPublishTime']}", className="small text-muted mb-0")
        ]), className="mb-3") for item in news_items]
        return html.Div(cards, className="mt-3")

    return html.P("Select a tab")


@dash.callback(
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

    df_formatted = df.applymap(lambda x: f"{x/1_000_000:,.1f}M" if isinstance(x, (int, float)) else x)
    df_reset = df_formatted.reset_index().rename(columns={'index': 'Metric'})
    return dbc.Table.from_dataframe(df_reset, striped=True, bordered=True, hover=True, class_name="small")

@dash.callback(
    Output('interactive-dcf-chart-deep-dive', 'figure'),
    Input('dcf-growth-rate-input-deep-dive', 'value'),
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
        mode = "gauge+number+delta", value = price,
        number = {'prefix': "$", 'font': {'size': 40}},
        delta = {'reference': iv, 'relative': False, 'valueformat': '.2f', 'increasing': {'color': "#dc3545"}, 'decreasing': {'color': "#198754"}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"<b>Margin of Safety: {margin_of_safety:.2%}</b><br><span style='font-size:0.8em;color:gray'>Intrinsic Value (DCF): ${iv:.2f}</span>", 'font': {'size': 20}},
        gauge = {'axis': {'range': [min(price, iv) * 0.75, max(price, iv) * 1.25], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "rgba(0,0,0,0)"},
            'steps' : [{'range': [0, iv], 'color': "rgba(25, 135, 84, 0.2)"}, {'range': [iv, max(price, iv) * 1.25], 'color': "rgba(220, 53, 69, 0.2)"}],
            'threshold' : {'line': {'color': "#636EFA", 'width': 5}, 'thickness': 1, 'value': price}
        }
    ))
    fig.update_layout(title_text=f"Valuation with {growth_rate:.1f}% Growth", height=400, margin=dict(t=80, b=40))
    return fig
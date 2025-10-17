# pages/deep_dive.py (Version with Trend Line Plotting Removed)

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.express as px
import json
import numpy as np

from data_handler import get_deep_dive_data, calculate_dcf_intrinsic_value, get_technical_analysis_data

def create_metric_card(title, value, className=""):
    if value is None or value == "N/A" or (isinstance(value, str) and "N/A" in value): return None
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

    marquee = dbc.Row([
        dbc.Col(html.Img(src=data.get('logo_url', '/assets/placeholder_logo.png'), className="company-logo"), width="auto", align="center"),
        dbc.Col([
            html.H1(data.get('company_name', ticker), className="company-name mb-0"),
            html.P(f"{data.get('exchange', '')}: {ticker}", className="text-muted small")
        ], width=True, align="center"),
        dbc.Col(
            html.Div([
                html.H2(f"${data.get('current_price', 0):,.2f}", className="price-display mb-0"),
                html.P(f"{data.get('daily_change_str', 'N/A')} ({data.get('daily_change_pct_str', 'N/A')})",
                    className=f"price-change {'price-positive' if data.get('daily_change', 0) >= 0 else 'price-negative'}")
            ], className="text-end"),
        width="auto", align="center")
    ], align="center", className="marquee-header g-3")

    key_stats = data.get("key_stats", {})
    target_price = data.get('target_mean_price')
    target_price_str = f"${target_price:,.2f}" if target_price is not None and pd.notna(target_price) else "N/A"
    reco_key = data.get('recommendation_key', "N/A")
    all_cards = [
        create_metric_card("Market Cap", data.get('market_cap_str')), create_metric_card("Analyst Target", target_price_str, "bg-light-subtle"),
        create_metric_card("Recommendation", reco_key, "bg-light-subtle"), create_metric_card("P/E Ratio", key_stats.get('P/E Ratio')),
        create_metric_card("Forward P/E", key_stats.get('Forward P/E')), create_metric_card("Dividend Yield", key_stats.get('Dividend Yield')),
        create_metric_card("PEG Ratio", key_stats.get('PEG Ratio')),
    ]
    cards_to_show = [card for card in all_cards if card is not None]
    at_a_glance = html.Div([
        dbc.Row([dbc.Col(card, width=6, lg=2, className="mb-3") for card in cards_to_show], className="g-3"),
        dbc.Card(dbc.CardBody([html.H5("Business Summary", className="card-title"), html.P(data.get('business_summary', 'Business summary not available.'), className="small")]))
    ])

    analysis_workspace = html.Div([
        html.Div(className="custom-tabs-container mt-4", children=[
            dbc.Tabs([
                    dbc.Tab(label="CHARTS", tab_id="tab-charts-deep-dive", label_class_name="fw-bold"),
                    dbc.Tab(label="TECHNICALS", tab_id="tab-technicals-deep-dive", label_class_name="fw-bold"),
                    dbc.Tab(label="FINANCIALS", tab_id="tab-financials-deep-dive", label_class_name="fw-bold"),
                    dbc.Tab(label="VALUATION", tab_id="tab-valuation-deep-dive", label_class_name="fw-bold"),
                ],
                id="deep-dive-main-tabs",
                active_tab="tab-charts-deep-dive",
                # --- [FIX] เพิ่ม 2 บรรทัดนี้เข้ามา ---
                persistence=True,
                persistence_type='session'
            )
        ]),
        dbc.Card(dbc.CardBody(dcc.Loading(html.Div(id="deep-dive-tab-content"))), className="mt-3")
    ])

    return dbc.Container([
        dcc.Store(id='deep-dive-ticker-store', data={'ticker': ticker}),
        marquee, html.Hr(), at_a_glance, analysis_workspace,
    ], fluid=True, className="p-4 main-content-container")

@dash.callback(Output('deep-dive-tab-content', 'children'), Input('deep-dive-main-tabs', 'active_tab'), State('deep-dive-ticker-store', 'data'))
def render_deep_dive_tab_content(active_tab, store_data):
    if not store_data or not store_data.get('ticker'): return ""
    ticker = store_data['ticker']
    data = get_deep_dive_data(ticker)

    if active_tab == "tab-charts-deep-dive":
        financial_trends = data.get("financial_trends", pd.DataFrame())
        margin_trends = data.get("margin_trends", pd.DataFrame())
        fig_trends = make_subplots(specs=[[{"secondary_y": True}]])
        if not financial_trends.empty:
            fig_trends.add_trace(go.Bar(x=financial_trends.index, y=financial_trends.get('Revenue'), name='Revenue', marker_color='royalblue'), secondary_y=False)
            fig_trends.add_trace(go.Scatter(x=financial_trends.index, y=financial_trends.get('Net Income'), name='Net Income', mode='lines+markers', line=dict(color='darkorange')), secondary_y=True)
        
        fig_trends.update_layout(
            title_text='Quarterly Financial Trends', 
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5), 
            margin=dict(t=50, b=50)
        )
        fig_trends.update_yaxes(title_text="Revenue ($)", secondary_y=False, rangemode='tozero')
        fig_trends.update_yaxes(title_text="Net Income ($)", secondary_y=True, rangemode='tozero', showgrid=False)

        if not margin_trends.empty:
            fig_margins = px.line(margin_trends, markers=True, title="Quarterly Profitability Margins")
            fig_margins.update_layout(
                yaxis_tickformat=".2%", 
                legend_title_text=None, 
                yaxis_title='Percentage', 
                legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
                margin=dict(t=50, b=50)
            )
            margin_graph = dcc.Graph(figure=fig_margins)
        else:
            margin_graph = dbc.Alert("Margin data not available.", color="info", className="h-100 d-flex align-items-center justify-content-center")

        return dbc.Row([dbc.Col(dcc.Graph(figure=fig_trends), md=6), dbc.Col(margin_graph, md=6)], className="mt-3")

    if active_tab == "tab-technicals-deep-dive":
        price_history = data.get("price_history")
        if price_history is None or price_history.empty:
            return dbc.Alert("Price history is not available for technical analysis.", color="warning")

        tech_data = get_technical_analysis_data(price_history)
        if "error" in tech_data:
            return dbc.Alert(tech_data["error"], color="danger")
        
        df = tech_data['data'].iloc[-365*2:]

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            row_heights=[0.6, 0.2, 0.2])

        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], mode='lines', name='EMA 20', line=dict(color='orange', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA 50', line=dict(color='blue', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], mode='lines', name='SMA 200', line=dict(color='purple', width=2)), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', name='BB Upper', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', name='BB Lower', line=dict(color='gray', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)

        # --- Trend Line Plotting Removed ---

        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=2, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], mode='lines', name='Signal', line=dict(color='red')), row=3, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram', marker_color=np.where(df['MACD_Hist'] > 0, 'green', 'red')), row=3, col=1)

        fig.update_layout(
            title_text=f'{ticker} - Technical Analysis',
            height=800,
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5)
        )
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)

        return dcc.Graph(figure=fig)

    if active_tab == "tab-financials-deep-dive":
        return html.Div([
            dbc.Row([
                dbc.Col(
                    dcc.Dropdown(
                        id='financial-statement-dropdown',
                        options=[
                            {'label': 'Income Statement', 'value': 'income'},
                            {'label': 'Balance Sheet', 'value': 'balance'},
                            {'label': 'Cash Flow', 'value': 'cashflow'},
                        ],
                        value='income',
                        clearable=False,
                    ),
                    width=12, md=4, lg=3
                )
            ], className="mb-4 mt-2"),
            dcc.Loading(html.Div(id="financial-statement-content"))
        ])

    if active_tab == "tab-valuation-deep-dive":
        return dbc.Row([
            dbc.Col([
                html.H5("DCF Assumptions", className="mt-3"), html.Hr(),
                dbc.Label("Your Forecast Growth Rate (%):", html_for="dcf-growth-rate-input-deep-dive"),
                dcc.Input(id='dcf-growth-rate-input-deep-dive', type='number', value=5, step=0.5, className="mb-2 form-control"),
                html.P("This model uses a standard WACC calculation and a perpetual growth rate of 2.5%.", className="text-muted small")
            ], md=4),
            dbc.Col(dcc.Loading(dcc.Graph(id='interactive-dcf-chart-deep-dive')), md=8)
        ], className="mt-3", align="center")

    return html.P("Select a tab")

@dash.callback(
    Output('financial-statement-content', 'children'),
    Input('financial-statement-dropdown', 'value'),
    State('deep-dive-ticker-store', 'data')
)
def render_financial_statement_table(selected_statement, store_data):
    if not store_data or not store_data.get('ticker'): return ""
    ticker = store_data['ticker']
    data = get_deep_dive_data(ticker)
    statements = data.get("financial_statements", {})
    df = pd.DataFrame()

    if selected_statement == 'income': df = statements.get('income', pd.DataFrame())
    elif selected_statement == 'balance': df = statements.get('balance', pd.DataFrame())
    elif selected_statement == 'cashflow': df = statements.get('cashflow', pd.DataFrame())

    if df.empty: return dbc.Alert("Financial data not available.", color="warning")

    df.columns = [col.strftime('%Y-%m-%d') if isinstance(col, pd.Timestamp) else col for col in df.columns]

    df_formatted = df.copy()
    for col in df_formatted.columns:
        numeric_col = pd.to_numeric(df_formatted[col], errors='coerce')
        df_formatted[col] = numeric_col.apply(
            lambda x: f"{x/1_000_000:,.1f}M" if pd.notna(x) else "-"
        )

    df_reset = df_formatted.reset_index().rename(columns={'index': 'Metric'})

    columns = [{"name": col, "id": col} for col in df_reset.columns]
    data = df_reset.to_dict('records')

    return dash_table.DataTable(
        data=data,
        columns=columns,
        style_header={
            'backgroundColor': 'transparent',
            'border': '0px',
            'borderBottom': '2px solid #dee2e6',
            'textTransform': 'uppercase',
            'fontWeight': '600',
        },
        style_cell={
            'textAlign': 'right',
            'padding': '14px',
            'border': '0px',
            'borderBottom': '1px solid #f0f0f0',
            'fontFamily': 'inherit',
            'fontSize': '0.9rem',
        },
        style_cell_conditional=[
            {
                'if': {'column_id': 'Metric'},
                'textAlign': 'left',
                'fontWeight': '600',
                'width': '35%'
            }
        ]
    )


@dash.callback(Output('interactive-dcf-chart-deep-dive', 'figure'), Input('dcf-growth-rate-input-deep-dive', 'value'), State('deep-dive-ticker-store', 'data'))
def update_interactive_dcf_chart(growth_rate, store_data):
    if not store_data or not store_data.get('ticker') or growth_rate is None:
        return go.Figure().update_layout(title_text="Enter a growth rate to calculate.")
    ticker = store_data['ticker']
    growth_rate_decimal = float(growth_rate) / 100.0
    result = calculate_dcf_intrinsic_value(ticker, growth_rate_decimal)
    if 'error' in result:
        return go.Figure().update_layout(title=f"Error Calculating DCF for {ticker}", annotations=[dict(text=result['error'], showarrow=False)])
    iv, price = result.get('intrinsic_value', 0), result.get('current_price', 0)
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
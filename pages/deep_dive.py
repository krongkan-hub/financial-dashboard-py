# pages/deep_dive.py (Version with Lazy Loading and All Fixes)

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
from datetime import datetime
import yfinance as yf # <-- FIX: Ensure yfinance is imported

# --- FIX: Import _get_logo_url for creating the initial layout ---
from data_handler import get_deep_dive_data, get_technical_analysis_data, _get_logo_url

# --- Helper function to create the sentiment layout (no changes) ---
def create_sentiment_layout(sentiment_data):
    if not sentiment_data or sentiment_data.get("error"):
        error_msg = sentiment_data.get("error", "Could not retrieve news sentiment.")
        return dbc.Alert(f"Error: {error_msg}", color="danger")

    summary = sentiment_data.get("summary", {})
    articles = sentiment_data.get("articles", [])

    if not articles:
        return dbc.Alert("No recent news articles found for sentiment analysis.", color="info")

    progress_bar = dbc.Progress(
        [
            dbc.Progress(value=summary.get('positive_pct', 0), color="success", bar=True),
            dbc.Progress(value=summary.get('neutral_pct', 0), color="warning", bar=True),
            dbc.Progress(value=summary.get('negative_pct', 0), color="danger", bar=True),
        ], style={"height": "30px", "fontSize": "1rem"}
    )
    
    summary_text = html.Div([
        html.Span(f"🟢 Positive: {summary.get('positive_count', 0)} articles ({summary.get('positive_pct', 0):.1f}%)", className="me-3"),
        html.Span(f"🟡 Neutral: {summary.get('neutral_count', 0)} articles ({summary.get('neutral_pct', 0):.1f}%)", className="me-3"),
        html.Span(f"🔴 Negative: {summary.get('negative_count', 0)} articles ({summary.get('negative_pct', 0):.1f}%)")
    ], className="text-center text-muted small mt-2")

    sentiment_color_map = {"positive": "success", "neutral": "warning", "negative": "danger"}
    
    news_list_items = []
    for article in articles[:10]:
        published_at = datetime.fromisoformat(article['publishedAt'].replace("Z", "+00:00")).strftime('%b %d, %Y')
        sentiment_label = article.get('sentiment', 'neutral')
        
        news_list_items.append(
            dbc.ListGroupItem([
                html.Div([
                    html.H6(article['title'], className="mb-1"),
                    html.P(article['description'], className="mb-1 small text-muted"),
                    html.Div([
                        dbc.Badge(sentiment_label.upper(), color=sentiment_color_map[sentiment_label], className="me-2"),
                        html.Small(f"Source: {article['source']['name']} | Published: {published_at}")
                    ])
                ])
            ], href=article['url'], target="_blank", action=True)
        )
    
    news_list = dbc.ListGroup(news_list_items, flush=True)

    return html.Div([
        html.H5("News Sentiment Analysis (Last 7 Days)", className="card-title"),
        progress_bar,
        summary_text,
        html.Hr(),
        news_list
    ])


def create_metric_card(title, value, className=""):
    if value is None or value == "N/A" or (isinstance(value, str) and "N/A" in value): return None
    card_title_content = [title]
    if title == "Recommendation":
        card_title_content.append(html.Span(" (Yahoo)", style={'fontSize': '0.7em', 'color': '#6c757d', 'marginLeft': '4px'}))
    return dbc.Card(dbc.CardBody([
        html.H6(card_title_content, className="metric-card-title"),
        html.P(value, className="metric-card-value"),
    ]), className=f"metric-card h-100 {className}")

def create_deep_dive_layout(ticker=None):
    if not ticker:
        return dbc.Container(html.P("Please provide a ticker by navigating from the main page."), className="p-5 text-center")
    
    try:
        tkr = yf.Ticker(ticker)
        info = tkr.info
        if not info or info.get('quoteType') != 'EQUITY':
             return dbc.Container([
                html.H2(f"Data Error for {ticker}", className="text-danger mt-5"),
                html.P("Invalid ticker or no data available."),
                dcc.Link("Go back to Dashboard", href="/")
             ], fluid=True, className="mt-5 text-center")

        # --- FIX: Call _get_logo_url to get the logo with fallback ---
        logo_url = _get_logo_url(info)

        current_price = info.get('currentPrice', 0)
        previous_close = info.get('previousClose', 1)
        daily_change = current_price - previous_close
        daily_change_pct = (daily_change / previous_close) * 100 if previous_close != 0 else 0
        
        def format_large_number(n):
            if pd.isna(n) or n is None: return "N/A"
            if abs(n) >= 1e12: return f'${n/1e12:,.2f}T'
            if abs(n) >= 1e9: return f'${n/1e9:,.2f}B'
            if abs(n) >= 1e6: return f'${n/1e6:,.2f}M'
            return f'${n:,.0f}'

        data = {
            "company_name": info.get('longName', ticker),
            "exchange": info.get('exchange', 'N/A'),
            "logo_url": logo_url, # <-- FIX: Use the variable from _get_logo_url
            "current_price": current_price,
            "daily_change_str": f"{'+' if daily_change >= 0 else ''}{daily_change:,.2f}",
            "daily_change_pct_str": f"{'+' if daily_change_pct >= 0 else ''}{daily_change_pct:.2f}%",
            "market_cap_str": format_large_number(info.get('marketCap')),
            "target_mean_price": info.get('targetMeanPrice'),
            "recommendation_key": info.get('recommendationKey', 'N/A').replace('_', ' ').title(),
            "key_stats": {
                "P/E Ratio": f"{info.get('trailingPE'):.2f}" if info.get('trailingPE') else "N/A",
                "Forward P/E": f"{info.get('forwardPE'):.2f}" if info.get('forwardPE') else "N/A",
                "Dividend Yield": f"{info.get('dividendYield',0)*100:.2f}%" if info.get('dividendYield') else "N/A",
                "PEG Ratio": f"{info.get('pegRatio'):.2f}" if info.get('pegRatio') else "N/A"
            },
            "business_summary": info.get('longBusinessSummary', 'Business summary not available.')
        }

    except Exception as e:
         return dbc.Container([
            html.H2(f"Data Error for {ticker}", className="text-danger mt-5"),
            html.P(f"Could not retrieve initial data. Reason: {e}"),
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
                    className=f"price-change {'price-positive' if daily_change >= 0 else 'price-negative'}")
            ], className="text-end"),
        width="auto", align="center")
    ], align="center", className="marquee-header g-3")

    key_stats = data.get("key_stats", {})
    target_price = data.get('target_mean_price')
    target_price_str = f"${target_price:,.2f}" if target_price is not None and pd.notna(target_price) else "N/A"
    reco_key = data.get('recommendation_key', "N/A")
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
        dbc.Row([dbc.Col(card, width=6, lg=2, className="mb-3") for card in cards_to_show], className="g-3"),
        dbc.Card(dbc.CardBody([html.H5("Business Summary", className="card-title"), html.P(data.get('business_summary', 'Business summary not available.'), className="small")]))
    ])

    analysis_workspace = html.Div([
        html.Div(className="custom-tabs-container mt-4", children=[
            dbc.Tabs([
                    dbc.Tab(label="CHARTS", tab_id="tab-charts-deep-dive", label_class_name="fw-bold"),
                    dbc.Tab(label="TECHNICALS", tab_id="tab-technicals-deep-dive", label_class_name="fw-bold"),
                    dbc.Tab(label="FINANCIALS", tab_id="tab-financials-deep-dive", label_class_name="fw-bold"),
                    dbc.Tab(label="SENTIMENT", tab_id="tab-sentiment-deep-dive", label_class_name="fw-bold"),
                ],
                id="deep-dive-main-tabs",
                active_tab="tab-charts-deep-dive",
                persistence=True,
                persistence_type='session'
            )
        ]),
        dbc.Card(dbc.CardBody(html.Div(id="deep-dive-tab-content")), className="mt-3")
    ])

    return dbc.Container([
        dcc.Store(id='deep-dive-ticker-store', data={'ticker': ticker}),
        marquee, html.Hr(), at_a_glance, analysis_workspace,
    ], fluid=True, className="p-4 main-content-container")

@dash.callback(
    Output('deep-dive-tab-content', 'children'),
    Input('deep-dive-main-tabs', 'active_tab')
)
def render_deep_dive_tab_content(active_tab):
    # This callback is now very fast, just returning the placeholder structure
    if active_tab == "tab-sentiment-deep-dive":
        return dcc.Loading(html.Div(id="sentiment-content-target"), type="default")
    
    if active_tab == "tab-financials-deep-dive":
        return html.Div([
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id='financial-statement-dropdown',
                    options=[
                        {'label': 'Income Statement', 'value': 'income'},
                        {'label': 'Balance Sheet', 'value': 'balance'},
                        {'label': 'Cash Flow', 'value': 'cashflow'},
                    ], value='income', clearable=False,
                ), width=12, md=4, lg=3)
            ], className="mb-4 mt-2"),
            dcc.Loading(html.Div(id="financial-statement-content"))
        ])

    return dcc.Loading(html.Div(id=f"{active_tab}-content-target"))

# --- Separate, slower callbacks for loading heavy content ---

@dash.callback(
    Output('tab-charts-deep-dive-content-target', 'children'),
    Input('deep-dive-main-tabs', 'active_tab'),
    State('deep-dive-ticker-store', 'data')
)
def load_charts_tab(active_tab, store_data):
    if active_tab != 'tab-charts-deep-dive' or not store_data:
        return dash.no_update
    
    ticker = store_data['ticker']
    data = get_deep_dive_data(ticker)
    
    financial_trends = data.get("financial_trends", pd.DataFrame())
    margin_trends = data.get("margin_trends", pd.DataFrame())
    
    fig_trends = make_subplots(specs=[[{"secondary_y": True}]])
    if not financial_trends.empty:
        fig_trends.add_trace(go.Bar(x=financial_trends.index, y=financial_trends.get('Revenue'), name='Revenue', marker_color='royalblue'), secondary_y=False)
        fig_trends.add_trace(go.Scatter(x=financial_trends.index, y=financial_trends.get('Net Income'), name='Net Income', mode='lines+markers', line=dict(color='darkorange')), secondary_y=True)
    fig_trends.update_layout(title_text='Quarterly Financial Trends', legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5), margin=dict(t=50, b=50))
    fig_trends.update_yaxes(title_text="Revenue ($)", secondary_y=False, rangemode='tozero')
    fig_trends.update_yaxes(title_text="Net Income ($)", secondary_y=True, rangemode='tozero', showgrid=False)

    if not margin_trends.empty:
        fig_margins = px.line(margin_trends, markers=True, title="Quarterly Profitability Margins")
        fig_margins.update_layout(yaxis_tickformat=".2%", legend_title_text=None, yaxis_title='Percentage', legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5), margin=dict(t=50, b=50))
        margin_graph = dcc.Graph(figure=fig_margins)
    else:
        margin_graph = dbc.Alert("Margin data not available.", color="info")

    return dbc.Row([dbc.Col(dcc.Graph(figure=fig_trends), md=6), dbc.Col(margin_graph, md=6)], className="mt-3")

@dash.callback(
    Output('tab-technicals-deep-dive-content-target', 'children'),
    Input('deep-dive-main-tabs', 'active_tab'),
    State('deep-dive-ticker-store', 'data')
)
def load_technicals_tab(active_tab, store_data):
    if active_tab != 'tab-technicals-deep-dive' or not store_data:
        return dash.no_update
        
    ticker = store_data['ticker']
    data = get_deep_dive_data(ticker)
    price_history = data.get("price_history")
    
    if price_history is None or price_history.empty:
        return dbc.Alert("Price history is not available.", color="warning")

    tech_data = get_technical_analysis_data(price_history)
    
    df = tech_data['data'].iloc[-365*2:]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], mode='lines', name='EMA 20', line=dict(color='orange', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA 50', line=dict(color='blue', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], mode='lines', name='SMA 200', line=dict(color='purple', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', name='BB Upper', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', name='BB Lower', line=dict(color='gray', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], mode='lines', name='Signal', line=dict(color='red')), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram', marker_color=np.where(df['MACD_Hist'] > 0, 'green', 'red')), row=3, col=1)
    fig.update_layout(title_text=f'{ticker} - Technical Analysis', height=800, xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5))
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    return dcc.Graph(figure=fig)

@dash.callback(
    Output('sentiment-content-target', 'children'),
    Input('deep-dive-main-tabs', 'active_tab'),
    State('deep-dive-ticker-store', 'data')
)
def load_sentiment_tab(active_tab, store_data):
    if active_tab != 'tab-sentiment-deep-dive' or not store_data:
        return dash.no_update
    
    ticker = store_data['ticker']
    data = get_deep_dive_data(ticker)
    sentiment_data = data.get("sentiment_data")
    
    return create_sentiment_layout(sentiment_data)

@dash.callback(
    Output('financial-statement-content', 'children'),
    Input('financial-statement-dropdown', 'value'),
    State('deep-dive-ticker-store', 'data')
)
def render_financial_statement_table(selected_statement, store_data):
    if not selected_statement or not store_data:
        return dash.no_update
        
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
        data=data, columns=columns,
        style_header={'backgroundColor': 'transparent','border': '0px','borderBottom': '2px solid #dee2e6','textTransform': 'uppercase','fontWeight': '600'},
        style_cell={'textAlign': 'right','padding': '14px','border': '0px','borderBottom': '1px solid #f0f0f0','fontFamily': 'inherit','fontSize': '0.9rem'},
        style_cell_conditional=[{'if': {'column_id': 'Metric'},'textAlign': 'left','fontWeight': '600','width': '35%'}]
    )
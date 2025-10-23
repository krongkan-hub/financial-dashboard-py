# pages/deep_dive.py (FIXED - Typo in fillcolor '1Z')

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
import yfinance as yf
from itertools import groupby

# --- [DELETED] ---
# from celery_worker import celery (ลบออก)

# --- [MODIFIED] ---
# เรียก get_news_and_sentiment โดยตรงจาก data_handler
from data_handler import get_deep_dive_data, get_technical_analysis_data, _get_logo_url, get_news_and_sentiment

# ==============================================================================
# SECTION 1: Layout Generating Functions
# ==============================================================================

# --- [FIXED] This layout matches your UI (Progress Bar) and link style ---
def create_sentiment_layout(sentiment_data):
    if not sentiment_data or sentiment_data.get("error"):
        return dbc.Alert(f"Error: {sentiment_data.get('error', 'Could not retrieve news.')}", color="danger")

    summary, articles_raw = sentiment_data.get("summary", {}), sentiment_data.get("articles", [])

    if not articles_raw:
        return dbc.Alert("No recent news articles found.", color="info")

    processed_articles = []
    for article in articles_raw:
        if not all([article.get('title'), article.get('publishedAt'), article.get('description'), article.get('url')]):
            continue
        try:
            article['published_dt'] = datetime.fromisoformat(article['publishedAt'].replace("Z", "+00:00"))
            processed_articles.append(article)
        except (ValueError, TypeError):
            continue

    print(f"DEBUG: {len(processed_articles)} articles survived filtering and date parsing.")
            
    processed_articles.sort(key=lambda x: x['published_dt'], reverse=True)

    # 1. Helper function for colors (Orange for Neutral)
    def get_sentiment_color_map(sentiment_label):
        if sentiment_label == 'positive': return "success" # Green
        if sentiment_label == 'negative': return "danger"  # Red
        return "warning" # Orange for Neutral (to match your screenshot)

    # 2. Create the stacked Progress Bar
    total_count = summary.get('total_count', 1) # Avoid division by zero
    pos_pct = summary.get('positive_pct', 0)
    neu_pct = summary.get('neutral_pct', 0)
    neg_pct = summary.get('negative_pct', 0)

    progress_bar = dbc.Progress(
        [
            dbc.Progress(value=pos_pct, color="success", bar=True, label=f"{pos_pct:.0f}%" if pos_pct > 10 else ""),
            dbc.Progress(value=neu_pct, color="warning", bar=True, label=f"{neu_pct:.0f}%" if neu_pct > 10 else ""),
            dbc.Progress(value=neg_pct, color="danger", bar=True, label=f"{neg_pct:.0f}%" if neg_pct > 10 else ""),
        ],
        style={"height": "15px"},
        className="mb-4"
    )

    # 3. Group articles by date
    layout_components = [
        html.H4("NEWS SENTIMENT ANALYSIS (LAST 7 DAYS)"),
        progress_bar
    ]
    
    def date_keyfn(article):
        return article['published_dt'].date()

    grouped_articles = groupby(processed_articles, key=date_keyfn)

    # 4. Create layout for each date group
    for date, articles in grouped_articles:
        date_header = html.P(
            date.strftime("%B %d, %Y").upper(), 
            className="text-muted small fw-bold mt-4 mb-2"
        )
        layout_components.append(date_header)
        
        article_layouts = []
        for article in articles:
            sentiment_label = article.get('sentiment', 'neutral')
            color = get_sentiment_color_map(sentiment_label)
            
            # This is the inner layout DIV
            article_layout = html.Div([
                dbc.Row([
                    dbc.Col(
                        dbc.Badge(sentiment_label.capitalize(), color=color, className="me-2"),
                        width="auto",
                        className="ps-3 pe-0"
                    ),
                    dbc.Col(
                        # Changed from html.A to html.Span
                        html.Span(
                            article['title'], 
                            className="news-headline-text" # This class is in style.css
                        )
                    )
                ], className="mb-1"),
                dbc.Row(dbc.Col(
                    html.Span(
                        f"{article['published_dt'].strftime('%I:%M %p')} | Source: {article.get('source', {}).get('name', 'N/A')}",
                        className="text-muted",
                        style={'fontSize': '0.85rem', 'paddingLeft': '20px'}
                    ),
                ))
            ], className="mb-3")
            
            # Wrap the entire DIV in a clickable link (html.A)
            wrapper_link = html.A(
                article_layout,
                href=article['url'],
                target="_blank",
                className="card-link-wrapper" # This class removes underline
            )
            article_layouts.append(wrapper_link)
        
        layout_components.append(html.Div(article_layouts))

    return html.Div(layout_components)


def create_technicals_layout(ticker: str):
    # (ฟังก์ชันนี้เหมือนเดิมทุกประการ... ยกเว้น 1Z)
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
    # --- [THE FIX] ---
    # Changed 'rgba(128,128,1Z,0.1)' to 'rgba(128,128,128,0.1)'
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', name='BB Lower', line=dict(color='gray', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
    # --- [END THE FIX] ---
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

def create_metric_card(title, value, className=""):
    # (ฟังก์ชันนี้เหมือนเดิมทุกประการ)
    if value is None or value == "N/A" or (isinstance(value, str) and "N/A" in value): return None
    return dbc.Card(dbc.CardBody([
        html.H6(title, className="metric-card-title"),
        html.P(value, className="metric-card-value"),
    ]), className=f"metric-card h-100 {className}")

def create_deep_dive_layout(ticker=None):
    if not ticker:
        return dbc.Container([html.P("Please provide a ticker.")], className="p-5 text-center")
    
    try:
        info = yf.Ticker(ticker).info
        if not info or info.get('quoteType') != 'EQUITY':
             return dbc.Container([html.H2(f"Data Error for {ticker}"), html.P("Invalid ticker."), dcc.Link("Go back", href="/")], fluid=True, className="mt-5 text-center")
        
        logo_url, company_name = _get_logo_url(info), info.get('longName', ticker)
        current_price, previous_close = info.get('currentPrice', 0), info.get('previousClose', 1)
        daily_change = current_price - previous_close
        daily_change_pct = (daily_change / previous_close) * 100 if previous_close != 0 else 0
        def format_large_number(n):
            if pd.isna(n) or n is None: return "N/A"
            if abs(n) >= 1e12: return f'${n/1e12:,.2f}T'
            if abs(n) >= 1e9: return f'${n/1e9:,.2f}B'
            return f'${n/1e6:,.2f}M'
        
    except Exception as e:
         return dbc.Container([html.H2(f"Data Error for {ticker}"), html.P(f"Reason: {e}"), dcc.Link("Go back", href="/")], fluid=True, className="mt-5 text-center")

    marquee = dbc.Row([
        dbc.Col(html.Img(src=logo_url or '/assets/placeholder_logo.png', className="company-logo"), width="auto", align="center"),
        dbc.Col([html.H1(company_name, className="company-name mb-0"), html.P(f"{info.get('exchange', '')}: {ticker}", className="text-muted small")], width=True, align="center"),
        dbc.Col(html.Div([html.H2(f"${current_price:,.2f}", className="price-display mb-0"), html.P(f"{'+' if daily_change >= 0 else ''}{daily_change:,.2f} ({'+' if daily_change_pct >= 0 else ''}{daily_change_pct:.2f}%)", className=f"price-change {'price-positive' if daily_change >= 0 else 'price-negative'}")], className="text-end"), width="auto", align="center")
    ], align="center", className="marquee-header g-3")

    key_stats = info
    cards_to_show = [
        create_metric_card("Market Cap", format_large_number(key_stats.get('marketCap'))),
        create_metric_card("Analyst Target", f"${key_stats.get('targetMeanPrice'):,.2f}" if key_stats.get('targetMeanPrice') else "N/A", "bg-light-subtle"),
        create_metric_card("Recommendation", key_stats.get('recommendationKey', 'N/A').replace('_', ' ').title(), "bg-light-subtle"),
        create_metric_card("P/E Ratio", f"{key_stats.get('trailingPE'):.2f}" if key_stats.get('trailingPE') else "N/A"),
        create_metric_card("Forward P/E", f"{key_stats.get('forwardPE'):.2f}" if key_stats.get('forwardPE') else "N/A"),
        create_metric_card("Dividend Yield", f"{key_stats.get('dividendYield', 0)*100:.2f}%" if key_stats.get('dividendYield') else "N/A"),
        create_metric_card("PEG Ratio", f"{key_stats.get('pegRatio'):.2f}" if key_stats.get('pegRatio') else "N/A"),
    ]
    at_a_glance = html.Div([
        dbc.Row([dbc.Col(card, width=6, lg=2, className="mb-3") for card in cards_to_show if card is not None], className="g-3"),
        dbc.Card(dbc.CardBody([html.H5("Business Summary", className="card-title"), html.P(info.get('longBusinessSummary', 'Not available.'), className="small")]))
    ])

    analysis_workspace = html.Div([
        html.Div(className="custom-tabs-container mt-4", children=[
            dbc.Tabs([
                    dbc.Tab(label="TECHNICALS", tab_id="tab-technicals-deep-dive", label_class_name="fw-bold"),
                    dbc.Tab(label="FINANCIALS", tab_id="tab-financials-deep-dive", label_class_name="fw-bold"),
                    dbc.Tab(label="NEWS", tab_id="tab-sentiment-deep-dive", label_class_name="fw-bold"),
                ],
                id="deep-dive-main-tabs",
                active_tab="tab-technicals-deep-dive", # <-- Default to Technicals
            )
        ]),
        
        # --- [FIXED] Spinner code (without 'children_style') ---
        dbc.Card(
            dbc.CardBody(
                dbc.Spinner(
                    html.Div(id="deep-dive-tab-content"), # This is the target
                    color="primary",
                    spinner_style={"width": "3rem", "height": "3rem"},
                    delay_show=250, # Only show if loading takes > 0.25s
                )
            ), 
            className="mt-3"
        )
        # --- [END SPINNER FIX] ---
    ])

    return dbc.Container([
        dcc.Store(id='deep-dive-ticker-store', data={'ticker': ticker, 'company_name': company_name}),
        marquee, html.Hr(), at_a_glance, analysis_workspace,
    ], fluid=True, className="p-4 main-content-container")

# ==============================================================================
# SECTION 2: Dash Callbacks
# ==============================================================================

# --- [FIXED] Removed dcc.Loading wrappers from return statements ---
@dash.callback(
    Output('deep-dive-tab-content', 'children'),
    Input('deep-dive-main-tabs', 'active_tab'),
    State('deep-dive-ticker-store', 'data')
)
def render_master_tab_content(active_tab, store_data):
    if not store_data or not store_data.get('ticker'):
        return dbc.Alert("Ticker not found.", color="danger")

    ticker, company_name = store_data.get('ticker'), store_data.get('company_name')
    
    if active_tab == "tab-sentiment-deep-dive":
        if not company_name: 
            return dbc.Alert("Company name not found.", color="warning")
        
        def get_and_render_news():
            sentiment_data = get_news_and_sentiment(company_name)
            return create_sentiment_layout(sentiment_data)

        return html.Div(get_and_render_news())

    elif active_tab == "tab-technicals-deep-dive":
        return create_technicals_layout(ticker)

    elif active_tab == "tab-financials-deep-dive":
        return html.Div([
            dbc.Row(dbc.Col(dcc.Dropdown(id='financial-statement-dropdown', options=[{'label': 'Income Statement', 'value': 'income'},{'label': 'Balance Sheet', 'value': 'balance'},{'label': 'Cash Flow', 'value': 'cashflow'},], value='income', clearable=False), width=12, md=4, lg=3), className="mb-4 mt-2"),
            # Note: We keep this dcc.Loading as it's for the *sub-content*
            dcc.Loading(html.Div(id="financial-statement-content"))
        ])

    return html.Div()


@dash.callback(
    Output('financial-statement-content', 'children'),
    Input('financial-statement-dropdown', 'value'),
    State('deep-dive-ticker-store', 'data')
)
def render_financial_statement_table(selected_statement, store_data):
    # (ฟังก์ชันนี้เหมือนเดิมทุกประการ)
    if not selected_statement or not store_data or not store_data.get('ticker'): return dash.no_update
    data = get_deep_dive_data(store_data['ticker'])
    statements = data.get("financial_statements", {})
    df = statements.get(selected_statement, pd.DataFrame())
    if df.empty: return dbc.Alert("Financial data not available.", color="warning")

    df.columns = [col.strftime('%Y-%m-%d') if isinstance(col, pd.Timestamp) else col for col in df.columns]
    df_formatted = df.copy()
    for col in df_formatted.columns:
        df_formatted[col] = pd.to_numeric(df_formatted[col], errors='coerce').apply(lambda x: f"{x/1e6:,.1f}M" if pd.notna(x) else "-")
    df_reset = df_formatted.reset_index().rename(columns={'index': 'Metric'})
    
    return dash_table.DataTable(
        data=df_reset.to_dict('records'),
        columns=[{"name": col, "id": col} for col in df_reset.columns],
        style_header={'backgroundColor': 'transparent','border': '0px','borderBottom': '2px solid #dee2e6','textTransform': 'uppercase','fontWeight': '600'},
        style_cell={'textAlign': 'right','padding': '14px','border': '0px','borderBottom': '1px solid #f0f0f0','fontFamily': 'inherit','fontSize': '0.9rem'},
        style_cell_conditional=[{'if': {'column_id': 'Metric'},'textAlign': 'left','fontWeight': '600','width': '35%'}]
    )
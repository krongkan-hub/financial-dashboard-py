# pages/deep_dive.py (Refactored - FINAL FIX with Fallback Functions)

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta, date
# import yfinance as yf # (ถูกลบออก - ไม่ได้ใช้ yf.Ticker() โดยตรงแล้ว)
from itertools import groupby
import logging

# --- Imports ---
from ...data_handler import (
    get_technical_analysis_data,
    get_news_and_sentiment,
    get_deep_dive_header_data,
    get_historical_prices,
    get_quarterly_financials
)
# --- [เพิ่ม Imports ใหม่] ---
from ... import db, server
from ...models import FactNewsSentiment # <<< เฉพาะ News ที่ยัง Query ตรง
from ...constants import ALL_TICKERS_SORTED_BY_MC # <<< [เพิ่ม] สำหรับเช็ค Top 20 (ยังคงใช้กับ News)
# --- End Imports ---

# ตั้งค่า logging (ถ้าต้องการให้ log แสดงผล)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ==============================================================================
# SECTION 1: Layout Generating Functions
# ==============================================================================

# --- create_sentiment_layout (เหมือนเดิม) ---
def create_sentiment_layout(sentiment_data):
    if not sentiment_data or sentiment_data.get("error"):
        return dbc.Alert(f"Error: {sentiment_data.get('error', 'Could not retrieve news.')}", color="danger")

    summary, articles_raw = sentiment_data.get("summary", {}), sentiment_data.get("articles", [])
    if not articles_raw:
        return dbc.Alert("No recent news articles found.", color="info")

    processed_articles = []
    for article in articles_raw:
        # ใช้ .get() เพื่อป้องกัน KeyError และใช้ 'url' เป็นตัวบ่งชี้หลัก
        if not all([article.get('title'), article.get('publishedAt'), article.get('url')]): continue
        try:
            # แก้ไข: ถ้า published_dt เป็น datetime object อยู่แล้ว (จาก DB)
            published_at_val = article['publishedAt']
            if isinstance(published_at_val, datetime):
                 article['published_dt'] = published_at_val
            else: # ถ้าเป็น string (จาก Live API)
                 article['published_dt'] = datetime.fromisoformat(published_at_val.replace("Z", "+00:00"))
            processed_articles.append(article)
        except (ValueError, TypeError) as e:
             logging.warning(f"Error parsing date for article: {e}")
             continue

    processed_articles.sort(key=lambda x: x['published_dt'], reverse=True)

    # Recalculate summary in case of live fetch (or to normalize structure)
    labels = [a['sentiment'] for a in processed_articles if a.get('sentiment')]
    total_count = len(labels)
    sentiment_scores = {'positive': labels.count('positive'), 'neutral': labels.count('neutral'), 'negative': labels.count('negative')}
    summary = {
        'total_count': total_count,
        'positive_count': sentiment_scores['positive'],
        'neutral_count': sentiment_scores['neutral'],
        'negative_count': sentiment_scores['negative'],
        'positive_pct': (sentiment_scores['positive'] / total_count) * 100 if total_count > 0 else 0,
        'neutral_pct': (sentiment_scores['neutral'] / total_count) * 100 if total_count > 0 else 0,
        'negative_pct': (sentiment_scores['negative'] / total_count) * 100 if total_count > 0 else 0,
    }


    def get_sentiment_color_map(sentiment_label):
        if sentiment_label == 'positive': return "success"
        if sentiment_label == 'negative': return "danger"
        return "warning"

    pos_pct, neu_pct, neg_pct = summary.get('positive_pct', 0), summary.get('neutral_pct', 0), summary.get('negative_pct', 0)

    progress_bar = dbc.Progress([
        dbc.Progress(value=pos_pct, color="success", bar=True, label=f"{pos_pct:.0f}%" if pos_pct > 10 else ""),
        dbc.Progress(value=neu_pct, color="warning", bar=True, label=f"{neu_pct:.0f}%" if neu_pct > 10 else ""),
        dbc.Progress(value=neg_pct, color="danger", bar=True, label=f"{neg_pct:.0f}%" if neg_pct > 10 else ""),
    ], style={"height": "15px"}, className="mb-4")

    layout_components = [ html.H4("NEWS SENTIMENT ANALYSIS (LAST 7 DAYS)"), progress_bar ]

    def date_keyfn(article): return article['published_dt'].date()
    grouped_articles = groupby(processed_articles, key=date_keyfn)

    for date, articles in grouped_articles:
        date_header = html.P(date.strftime("%B %d, %Y").upper(), className="text-muted small fw-bold mt-4 mb-2");
        layout_components.append(date_header)
        article_layouts = []
        for article in articles:
            sentiment_label = article.get('sentiment', 'neutral');
            color = get_sentiment_color_map(sentiment_label)
            article_layout = html.Div([
                dbc.Row([
                    dbc.Col(dbc.Badge(sentiment_label.capitalize(), color=color, className="me-2 sentiment-badge-fixed"), width="auto", className="ps-3 pe-0"),
                    dbc.Col(html.Span(article['title'], className="news-headline-text"))
                ], className="mb-1"),
                dbc.Row(dbc.Col(html.Span(f"{article['published_dt'].strftime('%I:%M %p')} | Source: {article.get('source', {}).get('name', 'N/A')}", className="text-muted", style={'fontSize': '0.85rem', 'paddingLeft': '20px'}),))
            ], className="mb-3")
            wrapper_link = html.A(article_layout, href=article['url'], target="_blank", className="card-link-wrapper");
            article_layouts.append(wrapper_link)
        layout_components.append(html.Div(article_layouts))

    return html.Div(layout_components)

# --- create_technicals_layout (MODIFIED: ใช้ get_historical_prices) ---
def create_technicals_layout(ticker: str):
    try:
        # --- [MODIFIED] Call the fallback function ---
        price_history, source = get_historical_prices(ticker, period="5y")
        logging.info(f"Technicals: Fetched price history for {ticker} from '{source}'.")
        # --- [END MODIFIED] ---

        # Check for error DataFrame returned by the fallback function
        if isinstance(price_history, pd.DataFrame) and "error" in price_history.columns:
             error_msg = price_history["error"].iloc[0]
             logging.error(f"Error fetching price history for {ticker}: {error_msg}")
             return dbc.Alert(f"Could not fetch price history: {error_msg}", color="danger")

        if price_history is None or price_history.empty:
            # Message specific to whether it failed from DB/Live or just no data
            alert_msg = f"Price history not found for {ticker} (Source: {source})."
            if source.startswith('live'):
                 alert_msg += " Please check the ticker symbol."
            else: # If source was database
                 alert_msg += " Data might be missing from the warehouse or the live fetch failed."
            return dbc.Alert(alert_msg, color="warning")

        # --- ส่วนที่เหลือเหมือนเดิม ---
        # Rename columns if they are still in DB format (lowercase)
        price_history = price_history.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        }, errors='ignore') # Use errors='ignore' in case columns are already correct

        tech_data = get_technical_analysis_data(price_history)
        if 'error' in tech_data or 'data' not in tech_data or tech_data['data'].empty:
             return dbc.Alert(f"Could not calculate technical indicators for {ticker}. Reason: {tech_data.get('error', 'Unknown')}", color="danger")

        df = tech_data['data'].iloc[-365*2:] # Plot last 2 years

        # --- Create Plotly Figure (โค้ดสร้างกราฟเหมือนเดิม) ---
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])
        # ... (Candlestick and Technical Indicators traces) ...
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], mode='lines', name='EMA 20', line=dict(color='orange', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA 50', line=dict(color='blue', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], mode='lines', name='SMA 200', line=dict(color='purple', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', name='BB Upper', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', name='BB Lower', line=dict(color='gray', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1) # Corrected fill color
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=2, col=1); fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], mode='lines', name='Signal', line=dict(color='red')), row=3, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram', marker_color=np.where(df['MACD_Hist'] > 0, 'green', 'red')), row=3, col=1)


        fig.update_layout(title_text=f'{ticker} - Technical Analysis (Source: {source.replace("_"," ").title()})', height=800, xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5))
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)

        return dcc.Graph(figure=fig)

    except Exception as e:
        logging.error(f"Error creating technicals layout for {ticker}: {e}", exc_info=True)
        return dbc.Alert(f"Could not generate technical chart: {e}", color="danger")


# --- create_metric_card (เหมือนเดิม) ---
def create_metric_card(title, value, className=""):
    # ... (โค้ดเหมือนเดิมทุกประการ) ...
    if value is None or value == "N/A" or (isinstance(value, (str, bytes)) and "N/A" in str(value)): return None # Check bytes too just in case
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value_str = f"{value:,.2f}" # Default formatting
        if abs(value) >= 1e12: value_str = f'${value/1e12:,.2f}T'
        elif abs(value) >= 1e9: value_str = f'${value/1e9:,.2f}B'
        elif abs(value) >= 1e6: value_str = f'${value/1e6:,.2f}M'
        # Check if title contains '%' for percentage formatting
        elif isinstance(title, str) and ('%' in title or title.lower() in ['roe', 'operating margin', 'ebitda margin', 'revenue growth (yoy)', 'net income growth (yoy)', 'revenue cagr (3y)']):
             value_str = f"{value*100:.2f}%" # Handle percentages based on title or known percentage metrics
    else: value_str = str(value)
    return dbc.Card(dbc.CardBody([ html.H6(title, className="metric-card-title"), html.P(value_str, className="metric-card-value"), ]), className=f"metric-card h-100 {className}")

def create_risk_gauge(predicted_prob):
    """สร้างการ์ดเกจวัดความเสี่ยงเครดิต (Credit Risk Gauge)"""
    
    # ถ้าไม่มีข้อมูล (เช่น หุ้นนอก coverage ของ ML หรือ live fetch)
    if predicted_prob is None or pd.isna(predicted_prob):
        return None

    prob_pct = predicted_prob * 100

    # กำหนดสีและข้อความตามระดับความเสี่ยง
    if prob_pct <= 10:
        bar_color = "#28a745" # สีเขียว (Low)
        level_text = "Low Risk"
    elif prob_pct <= 40:
        bar_color = "#ffc107" # สีเหลือง (Medium)
        level_text = "Medium Risk"
    else:
        bar_color = "#dc3545" # สีแดง (High)
        level_text = "High Risk"

    # สร้าง Gauge Figure ของ Plotly
    gauge_fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob_pct,
        number = {'suffix': '%', 'font': {'size': 28}},
        title = {'text': f"<b>{level_text}</b>", 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': bar_color, 'thickness': 0.8},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e9ecef",
            'steps': [
                {'range': [0, 10], 'color': 'rgba(40, 167, 69, 0.2)'},
                {'range': [10, 40], 'color': 'rgba(255, 193, 7, 0.2)'},
                {'range': [40, 100], 'color': 'rgba(220, 53, 69, 0.2)'}
            ],
            'threshold': { # เส้นแบ่ง Medium/High
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 40 # ค่าที่เส้น Threshold จะแสดง (เช่น 40%)
            }
        }
    ))

    gauge_fig.update_layout(
        height=200, # ปรับความสูง
        margin=dict(l=20, r=20, t=40, b=20), # ปรับขอบ
        font={'color': "#212529"}
    )

    # คืนค่าเป็น Card ที่มี Gauge อยู่ข้างใน
    return dbc.Card(
        dbc.CardBody([
            html.H6("ML Credit Risk", className="metric-card-title text-center"),
            dcc.Graph(
                figure=gauge_fig,
                config={'displayModeBar': False},
                style={'height': '150px', 'marginTop': '-10px'}
            )
        ]),
        className="metric-card h-100"
    )

# --- create_deep_dive_layout (MODIFIED: ใช้ get_deep_dive_header_data) ---
def create_deep_dive_layout(ticker=None):
    if not ticker: return dbc.Container([html.P("Please provide a ticker.")], className="p-5 text-center")

    try:
        # --- [MODIFIED] Call the fallback function for header data ---
        header_data = get_deep_dive_header_data(ticker)
        logging.info(f"Header: Fetched data for {ticker} from '{header_data.get('source', 'unknown')}'.")
        # --- [END MODIFIED] ---

        # Check for errors from the fallback function
        if header_data.get('error'):
             logging.error(f"Failed to fetch header data for {ticker}: {header_data['error']}")
             return dbc.Container([
                 html.H2(f"Data Error for {ticker}"),
                 html.P(f"Could not fetch essential company information. Reason: {header_data['error']}"),
                 dcc.Link("Go back", href="/")
             ], fluid=True, className="mt-5 text-center")

        # --- Extract data from the returned dictionary ---
        company_name = header_data.get('company_name', ticker) # Default to ticker if name is missing
        logo_url = header_data.get('logo_url')
        business_summary = header_data.get('long_business_summary')
        current_price = header_data.get('current_price')
        previous_close = header_data.get('previous_close')

        # --- Calculate Daily Change (handle None values) ---
        daily_change = None
        daily_change_pct = None
        if current_price is not None and previous_close is not None:
            daily_change = current_price - previous_close
            if previous_close != 0:
                daily_change_pct = (daily_change / previous_close) * 100
            else:
                daily_change_pct = 0 # Avoid division by zero

    except Exception as e:
         # Catch any unexpected error during processing
         logging.error(f"Unexpected error creating deep dive layout for {ticker}: {e}", exc_info=True)
         return dbc.Container([
             html.H2(f"Layout Error for {ticker}"),
             html.P(f"An unexpected error occurred while building the page. Reason: {e}"),
             dcc.Link("Go back", href="/")
         ], fluid=True, className="mt-5 text-center")

    # --- Layout Components ---
    # Marquee (Header section)
    price_display_text = f"${current_price:,.2f}" if current_price is not None else "N/A"
    change_text = "N/A"
    change_class = "text-muted"
    if daily_change is not None and daily_change_pct is not None:
        change_sign = '+' if daily_change >= 0 else ''
        change_text = f"{change_sign}{daily_change:,.2f} ({change_sign}{daily_change_pct:.2f}%)"
        change_class = 'price-positive' if daily_change >= 0 else 'price-negative'

    marquee = dbc.Row([
        dbc.Col(html.Img(src=logo_url or '/assets/placeholder_logo.png', className="company-logo"), width="auto", align="center"),
        dbc.Col([html.H1(company_name, className="company-name mb-0"), html.P(f"Ticker: {ticker}", className="text-muted small")], width=True, align="center"),
        dbc.Col(html.Div([
            html.H2(price_display_text, className="price-display mb-0"),
            html.P(change_text, className=f"price-change {change_class}")
        ], className="text-end"), width="auto", align="center")
    ], align="center", className="marquee-header g-3")

    risk_gauge_card = create_risk_gauge(header_data.get('predicted_default_prob'))

    # Key Stat Cards (using data from header_data dict)
    cards_to_show = [
        create_metric_card("Market Cap", header_data.get('market_cap')),
        create_metric_card("Analyst Target", header_data.get('analyst_target_price')),
        create_metric_card("P/E Ratio", header_data.get('pe_ratio')),
        create_metric_card("Forward P/E", header_data.get('forward_pe')),
        create_metric_card("Beta", header_data.get('beta'))
        # Add more cards if needed, using .get() for safety
    ]

    at_a_glance = html.Div([
        dbc.Row(
            # กรอง Card ที่เป็น None ออก (รวมถึง risk_gauge_card ถ้ามันเป็น None)
            [dbc.Col(card, width=6, lg=2, className="mb-3") for card in cards_to_show if card is not None] +
            # เพิ่ม risk_gauge_card ที่นี่ (ถ้ามันไม่เป็น None)
            ([dbc.Col(risk_gauge_card, width=6, lg=2, className="mb-3")] if risk_gauge_card else []),
            className="g-3"
        ),
        dbc.Card(dbc.CardBody([html.H5("Business Summary", className="card-title"), html.P(business_summary or 'Not available.', className="small")])) if business_summary else ""
    ])

    # Analysis Workspace (Tabs and Dropdown - เหมือนเดิม)
    analysis_workspace = html.Div([
        html.Div(className="custom-tabs-container mt-4", children=[
            dbc.Tabs([
                dbc.Tab(label="TECHNICALS", tab_id="tab-technicals-deep-dive", label_class_name="fw-bold"),
                dbc.Tab(label="FINANCIALS", tab_id="tab-financials-deep-dive", label_class_name="fw-bold"),
                dbc.Tab(label="NEWS", tab_id="tab-sentiment-deep-dive", label_class_name="fw-bold"),
            ], id="deep-dive-main-tabs", active_tab="tab-technicals-deep-dive", )
        ]),

        # Financials Dropdown (ซ่อนไว้)
        dbc.Row(id='financial-controls-row', style={'display': 'none'}, children=[
            dbc.Col(dcc.Dropdown(id='financial-statement-dropdown', options=[
                {'label': 'Income Statement', 'value': 'income'},
                {'label': 'Balance Sheet', 'value': 'balance'},
                {'label': 'Cash Flow', 'value': 'cashflow'},
            ], value='income', clearable=False), width=12, md=4, lg=3),
        ], className="mb-4 mt-2"),

        # Tab Content Area with Spinner
        dbc.Card(dbc.CardBody(dbc.Spinner(html.Div(id="deep-dive-tab-content"), color="primary", spinner_style={"width": "3rem", "height": "3rem"}, delay_show=250)), className="mt-3")
    ])

    # Final Layout Assembly
    return dbc.Container([
        dcc.Store(id='deep-dive-ticker-store', data={'ticker': ticker, 'company_name': company_name}), # Store company name for News fetch
        marquee,
        html.Hr(),
        at_a_glance,
        analysis_workspace,
    ], fluid=True, className="p-4 main-content-container")


# ==============================================================================
# SECTION 2: Dash Callbacks
# ==============================================================================

# --- Callback ควบคุมการแสดงผลของ Financial Dropdown (เหมือนเดิม) ---
@dash.callback(
    Output('financial-controls-row', 'style'),
    Input('deep-dive-main-tabs', 'active_tab')
)
def toggle_financial_dropdown_visibility(active_tab):
    """Controls whether the Financial Statement Dropdown is visible."""
    if active_tab == "tab-financials-deep-dive":
        return {'display': 'flex'}
    return {'display': 'none'}


# --- Callback หลักสำหรับเปลี่ยน Tab Content (MODIFIED: ปรับ News Logic) ---
@dash.callback(
    Output('deep-dive-tab-content', 'children'),
    Input('deep-dive-main-tabs', 'active_tab'),
    State('deep-dive-ticker-store', 'data')
)
def render_master_tab_content(active_tab, store_data):
    if not store_data or not store_data.get('ticker'): return dbc.Alert("Ticker not found in store.", color="danger")
    ticker = store_data.get('ticker')
    company_name = store_data.get('company_name') # Needed for live news fetch fallback

    if active_tab == "tab-sentiment-deep-dive":
        articles_list = []
        source = 'database' # Assume DB first

        # --- [MODIFIED NEWS LOGIC: Try DB first, then Fallback] ---
        logging.info(f"News: Trying to fetch news/sentiment for {ticker} from DATABASE.")
        try:
            with server.app_context():
                # Query DB for recent news (limit to last 7 days for consistency with live fetch)
                seven_days_ago = datetime.utcnow() - timedelta(days=7)
                articles_raw = db.session.query(FactNewsSentiment) \
                                         .filter(FactNewsSentiment.ticker == ticker,
                                                 FactNewsSentiment.published_at >= seven_days_ago) \
                                         .order_by(FactNewsSentiment.published_at.desc()) \
                                         .limit(20).all() # Limit results similar to API

            if articles_raw:
                logging.info(f"News: Found {len(articles_raw)} articles for {ticker} in DB.")
                # Convert SQLAlchemy objects to list of dicts
                for article in articles_raw:
                    articles_list.append({
                        'title': article.title,
                        'publishedAt': article.published_at, # datetime object
                        'description': article.description,
                        'url': article.article_url,
                        'source': {'name': article.source_name},
                        'sentiment': article.sentiment_label,
                        'sentiment_score': article.sentiment_score
                    })
            else:
                 logging.info(f"News: No recent articles found in database for {ticker}. Trying LIVE fetch...")
                 source = 'live'
                 if not company_name:
                     logging.warning(f"Cannot fetch live news for {ticker}: Company name missing.")
                     return dbc.Alert(f"No news found in database and cannot fetch live news (company name missing for {ticker}).", color="warning")

                 live_result = get_news_and_sentiment(company_name)

                 if 'error' in live_result:
                     logging.error(f"Live News Fetch Error for {ticker}: {live_result['error']}")
                     return dbc.Alert(f"Could not retrieve news (Source: Live). Error: {live_result['error']}", color="danger")

                 articles_list = live_result.get('articles', [])
                 if not articles_list:
                      logging.info(f"News: Live fetch returned no articles for {ticker}.")

        except Exception as e:
             logging.error(f"Error accessing news data for {ticker}: {e}", exc_info=True)
             return dbc.Alert(f"An error occurred while retrieving news data: {e}", color="danger")
        # --- [END MODIFIED NEWS LOGIC] ---

        # --- Common Logic for Summary and Display (เหมือนเดิม) ---
        if not articles_list:
            return dbc.Alert(f"No recent news articles found for this company (Source: {source.title()}).", color="info")

        # Recalculate summary (always useful)
        labels = [a['sentiment'] for a in articles_list if a.get('sentiment')]
        total_count = len(labels)

        summary = {
            'total_count': total_count,
            'positive_count': labels.count('positive'),
            'neutral_count': labels.count('neutral'),
            'negative_count': labels.count('negative'),
            'positive_pct': (labels.count('positive') / total_count) * 100 if total_count > 0 else 0,
            'neutral_pct': (labels.count('neutral') / total_count) * 100 if total_count > 0 else 0,
            'negative_pct': (labels.count('negative') / total_count) * 100 if total_count > 0 else 0,
        }

        sentiment_data = {"articles": articles_list, "summary": summary}
        return create_sentiment_layout(sentiment_data)

    elif active_tab == "tab-technicals-deep-dive":
        # create_technicals_layout now handles fallback internally
        return dcc.Loading(create_technicals_layout(ticker))

    elif active_tab == "tab-financials-deep-dive":
        # Placeholder for the financial statement table (rendered by another callback)
        return dcc.Loading(html.Div(id="financial-statement-content"))

    return html.Div(f"Tab content for {active_tab}") # Fallback


# --- Callback สำหรับ Render ตาราง Financial Statement (MODIFIED: ใช้ get_quarterly_financials) ---
@dash.callback(
    Output('financial-statement-content', 'children'),
    Input('financial-statement-dropdown', 'value'),
    State('deep-dive-ticker-store', 'data')
)
def render_financial_statement_table(selected_statement, store_data):
    if not selected_statement or not store_data or not store_data.get('ticker'): return dash.no_update
    ticker = store_data['ticker']

    try:
        # --- [MODIFIED] Call the fallback function ---
        df, source = get_quarterly_financials(ticker, selected_statement)
        logging.info(f"Financials: Fetched '{selected_statement}' for {ticker} from '{source}'.")
        # --- [END MODIFIED] ---

        # Check for error DataFrame from fallback
        if isinstance(df, pd.DataFrame) and "error" in df.columns:
            error_msg = df["error"].iloc[0]
            logging.error(f"Error fetching financial statement '{selected_statement}' for {ticker}: {error_msg}")
            return dbc.Alert(f"Could not display financial statement: {error_msg}", color="danger")

        if df is None or df.empty:
            alert_msg = f"Quarterly {selected_statement.replace('_', ' ').title()} data not available (Source: {source.title()}) for {ticker}."
            return dbc.Alert(alert_msg, color="warning")

        # --- ส่วน Format และ แสดงผล (เหมือนเดิม) ---
        # Convert date columns (which are columns in wide format) to strings
        df.columns = [col.strftime('%Y-%m-%d') if isinstance(col, (pd.Timestamp, datetime, date)) else str(col) for col in df.columns]

        df_formatted = df.copy()
        for col in df_formatted.columns:
            # Attempt numeric conversion, coercing errors
            numeric_col = pd.to_numeric(df_formatted[col], errors='coerce')
            # Apply formatting only to valid numbers
            df_formatted[col] = numeric_col.apply(lambda x: f"{x/1e6:,.1f}M" if pd.notna(x) else "-")

        # Reset index to make 'metric_name' a column
        df_reset = df_formatted.reset_index()
        # Rename the index column (now 'index' or similar) to 'Metric'
        metric_col_name = df_reset.columns[0] # Get the actual name of the first column
        df_reset.rename(columns={metric_col_name: 'Metric'}, inplace=True)

        return dash_table.DataTable(
            data=df_reset.to_dict('records'),
            columns=[{"name": col, "id": col} for col in df_reset.columns],
            style_header={'backgroundColor': 'transparent','border': '0px','borderBottom': '2px solid #dee2e6','textTransform': 'uppercase','fontWeight': '600'},
            style_cell={'textAlign': 'right','padding': '14px','border': '0px','borderBottom': '1px solid #f0f0f0','fontFamily': 'inherit','fontSize': '0.9rem'},
            style_cell_conditional=[{'if': {'column_id': 'Metric'},'textAlign': 'left','fontWeight': '600','width': '35%'}],
            page_size=len(df_reset) # Show all rows
        )
    except Exception as e:
        logging.error(f"Error rendering financial statement '{selected_statement}' for {ticker}: {e}", exc_info=True)
        return dbc.Alert(f"Could not display financial statement: {e}", color="danger")
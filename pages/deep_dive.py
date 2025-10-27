# pages/deep_dive.py (Refactored - FINAL FIX)

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
# import yfinance as yf # (ถูกลบออก - ไม่ได้ใช้ yf.Ticker() ในไฟล์นี้แล้ว)
from itertools import groupby
import logging 

# --- Imports ---
from data_handler import (
    # get_deep_dive_data, (ถูกลบออก)
    get_technical_analysis_data,
    # _get_logo_url, (ถูกลบออก)
    get_news_and_sentiment # <<< [เพิ่ม] สำหรับการดึง Live Data
)
# --- [เพิ่ม Imports ใหม่] ---
from app import db, server 
from app import FactDailyPrices, DimCompany, FactCompanySummary, FactFinancialStatements, FactNewsSentiment
from constants import ALL_TICKERS_SORTED_BY_MC # <<< [เพิ่ม] สำหรับเช็ค Top 20
# --- End Imports ---

# ตั้งค่า logging (ถ้าต้องการให้ log แสดงผล)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ==============================================================================
# SECTION 1: Layout Generating Functions 
# ==============================================================================

# --- create_sentiment_layout (ปรับปรุงให้รองรับข้อมูลจากทั้ง DB และ Live) ---
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

# --- create_technicals_layout (เหมือนเดิม) ---
def create_technicals_layout(ticker: str):
    try:
        price_history = pd.DataFrame() # Initialize empty
        with server.app_context():
            five_years_ago = datetime.utcnow().date() - timedelta(days=5*365)
            query = db.session.query(
                FactDailyPrices.date, FactDailyPrices.open, FactDailyPrices.high,
                FactDailyPrices.low, FactDailyPrices.close, FactDailyPrices.volume
            ).filter(
                FactDailyPrices.ticker == ticker,
                FactDailyPrices.date >= five_years_ago
            ).order_by(FactDailyPrices.date.asc())

            # --- [FIX #1 for TypeError] ---
            results = query.all() # Get list of Row objects (like tuples)

            if results:
                 price_history = pd.DataFrame(results, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                 price_history.set_index('date', inplace=True)
            else:
                 logging.warning(f"No price history found in DB for {ticker}")
            # --- [END OF FIX #1 for TypeError] ---

        if price_history is None or price_history.empty:
            return dbc.Alert(f"Price history not found in the warehouse for {ticker}. Please wait for the ETL job.", color="warning")

        price_history = price_history.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        })

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


        fig.update_layout(title_text=f'{ticker} - Technical Analysis', height=800, xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5))
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
        elif isinstance(title, str) and '%' in title:
             value_str = f"{value*100:.2f}%" # Handle percentages based on title
    else: value_str = str(value)
    return dbc.Card(dbc.CardBody([ html.H6(title, className="metric-card-title"), html.P(value_str, className="metric-card-value"), ]), className=f"metric-card h-100 {className}")

# --- create_deep_dive_layout (แก้ไขให้ย้าย Dropdown) ---
def create_deep_dive_layout(ticker=None):
    if not ticker: return dbc.Container([html.P("Please provide a ticker.")], className="p-5 text-center")
    
    try:
        with server.app_context():
            # 1. Query ข้อมูลบริษัท (DimCompany) และข้อมูลสรุปล่าสุด (FactCompanySummary)
            company_data = db.session.query(
                DimCompany.company_name, DimCompany.logo_url, DimCompany.sector,
                FactCompanySummary.price, FactCompanySummary.market_cap,
                FactCompanySummary.analyst_target_price, FactCompanySummary.pe_ratio,
                FactCompanySummary.forward_pe, FactCompanySummary.long_business_summary,
                FactCompanySummary.beta 
            ).join(
                FactCompanySummary, DimCompany.ticker == FactCompanySummary.ticker
            ).filter(
                DimCompany.ticker == ticker
            ).order_by(
                FactCompanySummary.date_updated.desc() 
            ).first()

            if not company_data:
                return dbc.Container([html.H2(f"Data Error for {ticker}"), html.P("Data not found in warehouse."), dcc.Link("Go back", href="/")], fluid=True, className="mt-5 text-center")

            # 2. Query ราคาปิดเมื่อวาน (จาก FactDailyPrices)
            price_history_query = db.session.query(FactDailyPrices.close) \
                                      .filter(FactDailyPrices.ticker == ticker) \
                                      .order_by(FactDailyPrices.date.desc()) \
                                      .limit(2) 
            
            price_history = price_history_query.all()
            
            current_price = company_data.price
            
            # หา Previous Close
            if len(price_history) == 2:
                previous_close = price_history[1][0] 
            elif len(price_history) == 1:
                previous_close = price_history[0][0]
            else:
                previous_close = current_price 

            daily_change = current_price - previous_close
            daily_change_pct = (daily_change / previous_close) * 100 if previous_close else 0
            
            logo_url = company_data.logo_url
            company_name = company_data.company_name
            business_summary = company_data.long_business_summary
            key_stats = company_data 

    except Exception as e:
         logging.error(f"Failed to fetch data from DB for {ticker}: {e}", exc_info=True)
         return dbc.Container([html.H2(f"Database Error for {ticker}"), html.P(f"Could not fetch data from warehouse. Reason: {e}"), dcc.Link("Go back", href="/")], fluid=True, className="mt-5 text-center")
    
    # --- Layout Components ---
    marquee = dbc.Row([ dbc.Col(html.Img(src=logo_url or '/assets/placeholder_logo.png', className="company-logo"), width="auto", align="center"), dbc.Col([html.H1(company_name, className="company-name mb-0"), html.P(f"Ticker: {ticker}", className="text-muted small")], width=True, align="center"), dbc.Col(html.Div([html.H2(f"${current_price:,.2f}", className="price-display mb-0"), html.P(f"{'+' if daily_change >= 0 else ''}{daily_change:,.2f} ({'+' if daily_change_pct >= 0 else ''}{daily_change_pct:.2f}%)", className=f"price-change {'price-positive' if daily_change >= 0 else 'price-negative'}")], className="text-end"), width="auto", align="center") ], align="center", className="marquee-header g-3")
    
    cards_to_show = [ 
        create_metric_card("Market Cap", key_stats.market_cap), 
        create_metric_card("Analyst Target", key_stats.analyst_target_price), 
        create_metric_card("P/E Ratio", key_stats.pe_ratio), 
        create_metric_card("Forward P/E", key_stats.forward_pe),
        create_metric_card("Beta", key_stats.beta)
    ]
    
    at_a_glance = html.Div([ dbc.Row([dbc.Col(card, width=6, lg=2, className="mb-3") for card in cards_to_show if card is not None], className="g-3"), dbc.Card(dbc.CardBody([html.H5("Business Summary", className="card-title"), html.P(business_summary or 'Not available.', className="small")])) if business_summary else "" ])
    
    # --- [START FIX] ย้าย Dropdown ออกมาไว้ใน Analysis Workspace หลัก ---
    analysis_workspace = html.Div([ 
        html.Div(className="custom-tabs-container mt-4", children=[ 
            dbc.Tabs([ 
                dbc.Tab(label="TECHNICALS", tab_id="tab-technicals-deep-dive", label_class_name="fw-bold"), 
                dbc.Tab(label="FINANCIALS", tab_id="tab-financials-deep-dive", label_class_name="fw-bold"), 
                dbc.Tab(label="NEWS", tab_id="tab-sentiment-deep-dive", label_class_name="fw-bold"), 
            ], id="deep-dive-main-tabs", active_tab="tab-technicals-deep-dive", ) 
        ]), 
        
        # [NEW] Dropdown ถูกย้ายมาอยู่ที่นี่ และซ่อนไว้โดยค่าเริ่มต้น
        dbc.Row(id='financial-controls-row', style={'display': 'none'}, children=[
            dbc.Col(dcc.Dropdown(id='financial-statement-dropdown', options=[
                {'label': 'Income Statement', 'value': 'income'},
                {'label': 'Balance Sheet', 'value': 'balance'},
                {'label': 'Cash Flow', 'value': 'cashflow'},
            ], value='income', clearable=False), width=12, md=4, lg=3), 
        ], className="mb-4 mt-2"),

        dbc.Card(dbc.CardBody(dbc.Spinner(html.Div(id="deep-dive-tab-content"), color="primary", spinner_style={"width": "3rem", "height": "3rem"}, delay_show=250)), className="mt-3") 
    ])
    # --- [END FIX] ---

    return dbc.Container([ 
        dcc.Store(id='deep-dive-ticker-store', data={'ticker': ticker, 'company_name': company_name}), 
        marquee, html.Hr(), at_a_glance, analysis_workspace, 
    ], fluid=True, className="p-4 main-content-container")


# ==============================================================================
# SECTION 2: Dash Callbacks 
# ==============================================================================

# --- [NEW CALLBACK] ควบคุมการแสดงผลของ Financial Dropdown ---
@dash.callback(
    Output('financial-controls-row', 'style'),
    Input('deep-dive-main-tabs', 'active_tab')
)
def toggle_financial_dropdown_visibility(active_tab):
    """Controls whether the Financial Statement Dropdown is visible."""
    if active_tab == "tab-financials-deep-dive":
        # Show dropdown when Financials tab is active
        return {'display': 'flex'} 
    # Hide dropdown for other tabs
    return {'display': 'none'}


# --- Callback หลักสำหรับเปลี่ยน Tab Content (แก้ไข Financials Tab และเพิ่ม News Logic) ---
@dash.callback(
    Output('deep-dive-tab-content', 'children'),
    Input('deep-dive-main-tabs', 'active_tab'),
    State('deep-dive-ticker-store', 'data')
)
def render_master_tab_content(active_tab, store_data):
    if not store_data or not store_data.get('ticker'): return dbc.Alert("Ticker not found in store.", color="danger")
    ticker, company_name = store_data.get('ticker'), store_data.get('company_name')
    
    if active_tab == "tab-sentiment-deep-dive":
        ETL_TICKERS = ALL_TICKERS_SORTED_BY_MC[:20] 
        articles_list = []
        
        # --- ตรรกะการดึงข้อมูลข่าว ---
        if ticker in ETL_TICKERS:
            # Case 1: Ticker อยู่ในกลุ่ม Top 20 -> ดึงจาก Database (ETL)
            logging.info(f"Ticker {ticker} is in Top 20. Fetching news/sentiment from DATABASE.")
            with server.app_context():
                articles_raw = db.session.query(FactNewsSentiment) \
                                         .filter(FactNewsSentiment.ticker == ticker) \
                                         .order_by(FactNewsSentiment.published_at.desc()) \
                                         .limit(20).all()
            
            if not articles_raw:
                return dbc.Alert(f"No recent news articles found in database for {ticker}. Waiting for next ETL run.", color="info")

            # แปลง SQLAlchemy objects เป็น list of dicts
            for article in articles_raw:
                articles_list.append({
                    'title': article.title,
                    'publishedAt': article.published_at, # เป็น datetime object แล้ว
                    'description': article.description,
                    'url': article.article_url,
                    'source': {'name': article.source_name},
                    'sentiment': article.sentiment_label,
                    'sentiment_score': article.sentiment_score
                })

        else:
            # Case 2: Ticker ไม่อยู่ในกลุ่ม Top 20 -> ดึง Live Data จาก API
            logging.info(f"Ticker {ticker} is outside Top 20. Fetching LIVE news/sentiment for {company_name}.")
            if not company_name:
                return dbc.Alert(f"Cannot fetch live news: Company name not found for {ticker}.", color="danger")
                
            live_result = get_news_and_sentiment(company_name)

            if 'error' in live_result:
                return dbc.Alert(f"Live News Fetch Error for {ticker}: {live_result['error']}", color="danger")
            
            articles_list = live_result.get('articles', [])
        
        # --- Common Logic for Summary and Display ---
        if not articles_list:
            return dbc.Alert("No recent news articles found for this company.", color="info")

        # Recalculate summary (หรือใช้ summary ที่มากับ live_result)
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
        return create_technicals_layout(ticker) 
    
    elif active_tab == "tab-financials-deep-dive":
        # *** [FIX] คืนค่าเฉพาะ content placeholder เท่านั้น ***
        return dcc.Loading(html.Div(id="financial-statement-content"))
    
    return html.Div(f"Tab content for {active_tab}") # Fallback


# --- Callback สำหรับ Render ตาราง Financial Statement (แก้ไขให้ดึงจาก DB และ FIX read_sql) ---
@dash.callback(
    Output('financial-statement-content', 'children'),
    Input('financial-statement-dropdown', 'value'),
    State('deep-dive-ticker-store', 'data')
)
def render_financial_statement_table(selected_statement, store_data):
    if not selected_statement or not store_data or not store_data.get('ticker'): return dash.no_update
    ticker = store_data['ticker']
    
    try:
        with server.app_context():
            # Query งบการเงินแบบ Long Format จาก DB
            query_stmt = db.session.query(FactFinancialStatements.report_date, 
                                     FactFinancialStatements.metric_name, 
                                     FactFinancialStatements.metric_value) \
                              .filter(FactFinancialStatements.ticker == ticker,
                                      FactFinancialStatements.statement_type == selected_statement) \
                              .order_by(FactFinancialStatements.report_date.desc()) \
                              .limit(16 * 100) 
            
            # *** [FIXED] ใช้ query_stmt.statement และ db.engine แทน db.session.bind ***
            df_long = pd.read_sql(query_stmt.statement, db.engine)

        if df_long.empty:
            return dbc.Alert(f"Quarterly {selected_statement} data not available in DB for {ticker}.", color="warning")

        # Pivot ข้อมูลกลับเป็นตาราง (Wide format) เพื่อแสดงผล
        df = df_long.pivot(index='metric_name', columns='report_date', values='metric_value')
        
        # เรียง Columns (วันที่) จากใหม่ไปเก่า
        df = df[sorted(df.columns, reverse=True)]
        
        # เลือกมาแค่ 16 ไตรมาสล่าสุด (ถ้ามีเยอะเกิน)
        df = df.iloc[:, :16]
        
        # (ส่วนที่เหลือเหมือนเดิม)
        if df is None or df.empty: return dbc.Alert(f"Quarterly {selected_statement.replace('_', ' ').title()} data not available via API for {ticker}.", color="warning")
        
        df.columns = [col.strftime('%Y-%m-%d') if isinstance(col, (pd.Timestamp, datetime)) else str(col) for col in df.columns]
        df_formatted = df.copy()
        for col in df_formatted.columns: 
            numeric_col = pd.to_numeric(df_formatted[col], errors='coerce')
            df_formatted[col] = numeric_col.apply(lambda x: f"{x/1e6:,.1f}M" if pd.notna(x) else "-")
        
        df_reset = df_formatted.reset_index().rename(columns={'metric_name': 'Metric'})
        
        return dash_table.DataTable( 
            data=df_reset.to_dict('records'), 
            columns=[{"name": col, "id": col} for col in df_reset.columns], 
            style_header={'backgroundColor': 'transparent','border': '0px','borderBottom': '2px solid #dee2e6','textTransform': 'uppercase','fontWeight': '600'}, 
            style_cell={'textAlign': 'right','padding': '14px','border': '0px','borderBottom': '1px solid #f0f0f0','fontFamily': 'inherit','fontSize': '0.9rem'}, 
            style_cell_conditional=[{'if': {'column_id': 'Metric'},'textAlign': 'left','fontWeight': '600','width': '35%'}], 
            page_size=len(df_reset) 
        )
    except Exception as e: 
        logging.error(f"Error rendering financial statement '{selected_statement}' for {ticker}: {e}", exc_info=True)
        # ตรวจสอบ error ที่เฉพาะเจาะจงที่แก้ไขไปแล้ว (Query must be a string)
        if "Query must be a string using sqlalchemy" in str(e):
             return dbc.Alert(f"Could not display financial statement: {e} (Please ensure all code changes were applied correctly, including db.engine usage).", color="danger")
        return dbc.Alert(f"Could not display financial statement: {e}", color="danger")
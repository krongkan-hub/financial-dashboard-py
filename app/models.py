# app/models.py

from . import db  # <-- [NEW] Import db จาก __init__.py
from flask_login import UserMixin
from datetime import datetime, timezone

# --- Database Models ---
# (คัดลอก Class ทั้งหมดจาก app.py เดิมมาวางที่นี่)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class UserSelection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    symbol_type = db.Column(db.String(10), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)

class UserAssumptions(db.Model):
    # ... (โค้ด Class ทั้งหมด) ...
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)

    # Forecast Assumptions
    forecast_years = db.Column(db.Integer, default=5)
    eps_growth = db.Column(db.Float, default=10.0)
    terminal_pe = db.Column(db.Float, default=20.0)

    # DCF Assumptions (Expanded to store all 10 values)
    dcf_simulations = db.Column(db.Integer, default=10000)

    dcf_growth_min = db.Column(db.Float, default=3.0)
    dcf_growth_mode = db.Column(db.Float, default=5.0) 
    dcf_growth_max = db.Column(db.Float, default=8.0)

    dcf_perpetual_min = db.Column(db.Float, default=1.5)
    dcf_perpetual_mode = db.Column(db.Float, default=2.5) 
    dcf_perpetual_max = db.Column(db.Float, default=3.0)

    dcf_wacc_min = db.Column(db.Float, default=7.0)
    dcf_wacc_mode = db.Column(db.Float, default=8.0) 
    dcf_wacc_max = db.Column(db.Float, default=10.0)

class DimCompany(db.Model):
    # ... (โค้ด Class ทั้งหมด) ...
    __tablename__ = 'dim_company'
    ticker = db.Column(db.String(20), primary_key=True)
    company_name = db.Column(db.String(255), nullable=True)
    logo_url = db.Column(db.String(500), nullable=True)
    sector = db.Column(db.String(100), nullable=True) 
    credit_rating = db.Column(db.String(32), nullable=True)
    summaries = db.relationship('FactCompanySummary', backref='company', lazy=True)
    prices = db.relationship('FactDailyPrices', backref='company', lazy=True)
    financials = db.relationship('FactFinancialStatements', backref='company', lazy=True)
    news = db.relationship('FactNewsSentiment', backref='company', lazy=True)

class FactCompanySummary(db.Model):
    # ... (โค้ด Class ทั้งหมด) ...
    __tablename__ = 'fact_company_summary'
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(20), db.ForeignKey('dim_company.ticker'), nullable=False, index=True)
    date_updated = db.Column(db.Date, nullable=False, default=datetime.utcnow().date(), index=True)
    fcf_ttm = db.Column(db.Float, nullable=True)
    revenue_ttm = db.Column(db.Float, nullable=True)
    price = db.Column(db.Float, nullable=True)
    market_cap = db.Column(db.Float, nullable=True) 
    beta = db.Column(db.Float, nullable=True)
    pe_ratio = db.Column(db.Float, nullable=True)
    pb_ratio = db.Column(db.Float, nullable=True)
    ev_ebitda = db.Column(db.Float, nullable=True)
    revenue_growth_yoy = db.Column(db.Float, nullable=True)
    revenue_cagr_3y = db.Column(db.Float, nullable=True)
    net_income_growth_yoy = db.Column(db.Float, nullable=True)
    roe = db.Column(db.Float, nullable=True)
    de_ratio = db.Column(db.Float, nullable=True)
    operating_margin = db.Column(db.Float, nullable=True)
    cash_conversion = db.Column(db.Float, nullable=True)
    ebitda_margin = db.Column(db.Float, nullable=True)
    trailing_eps = db.Column(db.Float, nullable=True)
    forward_pe = db.Column(db.Float, nullable=True)
    analyst_target_price = db.Column(db.Float, nullable=True)
    long_business_summary = db.Column(db.Text, nullable=True)
    credit_rating = db.Column(db.String(20), nullable=True)
    peer_cluster_id = db.Column(db.Integer, nullable=True, index=True)
    predicted_default_prob = db.Column(db.Float, nullable=True)
    __table_args__ = (db.UniqueConstraint('ticker', 'date_updated', name='_ticker_date_uc'),)

class FactDailyPrices(db.Model):
    # ... (โค้ด Class ทั้งหมด) ...
    __tablename__ = 'fact_daily_prices'
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(20), db.ForeignKey('dim_company.ticker'), nullable=False, index=True)
    date = db.Column(db.Date, nullable=False, index=True)
    open = db.Column(db.Float)
    high = db.Column(db.Float)
    low = db.Column(db.Float)
    close = db.Column(db.Float)
    volume = db.Column(db.BigInteger) 
    __table_args__ = (db.UniqueConstraint('ticker', 'date', name='_ticker_price_date_uc'),)

class FactFinancialStatements(db.Model):
    # ... (โค้ด Class ทั้งหมด) ...
    __tablename__ = 'fact_financial_statements'
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(20), db.ForeignKey('dim_company.ticker'), nullable=False, index=True)
    report_date = db.Column(db.Date, nullable=False, index=True)
    statement_type = db.Column(db.String(20), nullable=False, index=True)
    metric_name = db.Column(db.String(255), nullable=False)
    metric_value = db.Column(db.Float, nullable=True)
    __table_args__ = (db.UniqueConstraint('ticker', 'report_date', 'statement_type', 'metric_name', name='_ticker_date_statement_metric_uc'),)

class FactNewsSentiment(db.Model):
    # ... (โค้ด Class ทั้งหมด) ...
    __tablename__ = 'fact_news_sentiment'
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(20), db.ForeignKey('dim_company.ticker'), nullable=False, index=True)
    published_at = db.Column(db.DateTime, nullable=False, index=True)
    title = db.Column(db.String(500), nullable=False)
    article_url = db.Column(db.String(1000), nullable=False, unique=True)
    source_name = db.Column(db.String(100), nullable=True)
    description = db.Column(db.Text, nullable=True)
    sentiment_label = db.Column(db.String(20), nullable=True)
    sentiment_score = db.Column(db.Float, nullable=True)
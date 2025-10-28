# app.py (FIXED - Core App Initialization with Absolute Path & Celery Integration)

import os
import dash
import dash_bootstrap_components as dbc
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin
from flask_migrate import Migrate
from config import Config
from datetime import timezone

# --- App Initialization ---
server = Flask(__name__)
server.config.from_object(Config)

# 2. เชื่อม Flask config เข้ากับ Celery config
#    เพื่อให้ Celery รู้จักค่า CELERY_BROKER_URL จากไฟล์ config.py
#celery.conf.update(server.config)

app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.LUX, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# --- Database & Login Manager Setup ---
instance_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'instance')
if not os.path.exists(instance_path):
    os.makedirs(instance_path)

db = SQLAlchemy(server)
login_manager = LoginManager()
login_manager.init_app(server)

# --- [NEW INITIALIZATION] ---
# Initialize Flask-Migrate
migrate = Migrate(server, db) 
# --- [END NEW INITIALIZATION] ---

# --- Database Models ---
# It's better to define models here to avoid circular dependencies
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class UserSelection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    symbol_type = db.Column(db.String(10), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)

# --- [MODIFIED] UserAssumptions Model ---
class UserAssumptions(db.Model):
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
# --- [END MODIFIED] ---


# --- [REFACTOR STEP 2: DATA WAREHOUSE SCHEMA] ---
from datetime import datetime

class DimCompany(db.Model):
    """
    Dimension Table: เก็บข้อมูลคงที่ของบริษัท
    """
    __tablename__ = 'dim_company'
    
    # ข้อมูลจาก yf.Ticker(ticker).info
    ticker = db.Column(db.String(20), primary_key=True)
    company_name = db.Column(db.String(255), nullable=True)
    logo_url = db.Column(db.String(500), nullable=True)
    sector = db.Column(db.String(100), nullable=True) 
    
    # สร้างความสัมพันธ์ (Relationships) เพื่อให้ query ง่ายขึ้น
    summaries = db.relationship('FactCompanySummary', backref='company', lazy=True)
    prices = db.relationship('FactDailyPrices', backref='company', lazy=True)
    # --- [เพิ่ม Relationships ใหม่] ---
    financials = db.relationship('FactFinancialStatements', backref='company', lazy=True)
    news = db.relationship('FactNewsSentiment', backref='company', lazy=True)

    def __repr__(self):
        return f'<DimCompany {self.ticker}>'

# --- [START OF MODIFICATION] ---
class FactCompanySummary(db.Model):
    """
    Fact Table: เก็บข้อมูลสรุปรายวันจาก data_handler.get_competitor_data
    (ขยายตารางเพื่อเก็บข้อมูลจาก tkr.info)
    """
    __tablename__ = 'fact_company_summary'
    
    id = db.Column(db.Integer, primary_key=True)
    # เชื่อมโยงไปยังตาราง DimCompany
    ticker = db.Column(db.String(20), db.ForeignKey('dim_company.ticker'), nullable=False, index=True)
    # วันที่ที่ข้อมูลนี้ถูกดึงมา (สำหรับ ETL)
    date_updated = db.Column(db.Date, nullable=False, default=datetime.utcnow().date(), index=True)
    fcf_ttm = db.Column(db.Float, nullable=True)
    revenue_ttm = db.Column(db.Float, nullable=True)
    # --- คอลัมน์ที่ตรงกับผลลัพธ์ของ get_competitor_data (เดิม) ---
    price = db.Column(db.Float, nullable=True)
    market_cap = db.Column(db.Float, nullable=True) # yfinance ส่งมาเป็น Float (e.g., 2.5e12)
    beta = db.Column(db.Float, nullable=True)
    pe_ratio = db.Column(db.Float, nullable=True)         # "P/E"
    pb_ratio = db.Column(db.Float, nullable=True)         # "P/B"
    ev_ebitda = db.Column(db.Float, nullable=True)      # "EV/EBITDA"
    revenue_growth_yoy = db.Column(db.Float, nullable=True) # "Revenue Growth (YoY)"
    revenue_cagr_3y = db.Column(db.Float, nullable=True)  # "Revenue CAGR (3Y)"
    net_income_growth_yoy = db.Column(db.Float, nullable=True) # "Net Income Growth (YoY)"
    roe = db.Column(db.Float, nullable=True)              # "ROE"
    de_ratio = db.Column(db.Float, nullable=True)         # "D/E Ratio"
    operating_margin = db.Column(db.Float, nullable=True) # "Operating Margin"
    cash_conversion = db.Column(db.Float, nullable=True)  # "Cash Conversion"
    
    # --- [คอลัมน์ใหม่ที่เพิ่มเข้ามา] ---
    # สำหรับกราฟ Scatter (Valuation vs. Quality)
    ebitda_margin = db.Column(db.Float, nullable=True)
    
    # สำหรับตาราง Target (Exit Multiple)
    trailing_eps = db.Column(db.Float, nullable=True)
    
    # สำหรับ Deep Dive Header / Key Stats
    forward_pe = db.Column(db.Float, nullable=True)
    analyst_target_price = db.Column(db.Float, nullable=True)
    long_business_summary = db.Column(db.Text, nullable=True) # db.Text สำหรับข้อความยาว
    # (สามารถเพิ่มคอลัมน์อื่นๆ ที่ดึงจาก tkr.info ได้ที่นี่)

    peer_cluster_id = db.Column(db.Integer, nullable=True, index=True)

    # สร้าง Unique Constraint เพื่อป้องกันข้อมูลซ้ำซ้อน
    # (ไม่อนุญาตให้มี Ticker เดียวกันในวันเดียวกัน 2 แถว)
    __table_args__ = (db.UniqueConstraint('ticker', 'date_updated', name='_ticker_date_uc'),)

    def __repr__(self):
        return f'<FactCompanySummary {self.ticker} on {self.date_updated}>'
# --- [END OF MODIFICATION] ---

class FactDailyPrices(db.Model):
    """
    Fact Table: เก็บข้อมูลราคาย้อนหลัง (OHLCV)
    """
    __tablename__ = 'fact_daily_prices'
    
    id = db.Column(db.Integer, primary_key=True)
    # เชื่อมโยงไปยังตาราง DimCompany
    ticker = db.Column(db.String(20), db.ForeignKey('dim_company.ticker'), nullable=False, index=True)
    date = db.Column(db.Date, nullable=False, index=True)
    
    open = db.Column(db.Float)
    high = db.Column(db.Float)
    low = db.Column(db.Float)
    close = db.Column(db.Float)
    volume = db.Column(db.BigInteger) # Volume สามารถใหญ่มากได้
    
    # ป้องกันข้อมูลซ้ำ (ราคาของ Ticker เดียวกัน ในวันเดียวกัน ควรมีแค่แถวเดียว)
    __table_args__ = (db.UniqueConstraint('ticker', 'date', name='_ticker_price_date_uc'),)

    def __repr__(self):
        return f'<FactDailyPrices {self.ticker} on {self.date}>'

# --- [START OF NEW MODELS] ---
class FactFinancialStatements(db.Model):
    """
    Fact Table: เก็บข้อมูลงบการเงินรายไตรมาส (Long Format)
    """
    __tablename__ = 'fact_financial_statements'
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(20), db.ForeignKey('dim_company.ticker'), nullable=False, index=True)
    report_date = db.Column(db.Date, nullable=False, index=True)
    statement_type = db.Column(db.String(20), nullable=False, index=True) # 'income', 'balance', 'cashflow'
    metric_name = db.Column(db.String(255), nullable=False)
    metric_value = db.Column(db.Float, nullable=True)

    # ป้องกันข้อมูลซ้ำ
    __table_args__ = (db.UniqueConstraint('ticker', 'report_date', 'statement_type', 'metric_name', name='_ticker_date_statement_metric_uc'),)

    def __repr__(self):
        return f'<FactFinancialStatements {self.ticker} {self.metric_name} {self.report_date}>'

class FactNewsSentiment(db.Model):
    """
    Fact Table: เก็บข่าวสารและผลการวิเคราะห์ Sentiment
    """
    __tablename__ = 'fact_news_sentiment'
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(20), db.ForeignKey('dim_company.ticker'), nullable=False, index=True)
    published_at = db.Column(db.DateTime, nullable=False, index=True)
    title = db.Column(db.String(500), nullable=False)
    article_url = db.Column(db.String(1000), nullable=False, unique=True) # ใช้ URL กันซ้ำ
    source_name = db.Column(db.String(100), nullable=True)
    description = db.Column(db.Text, nullable=True)
    sentiment_label = db.Column(db.String(20), nullable=True)
    sentiment_score = db.Column(db.Float, nullable=True)

    def __repr__(self):
        return f'<FactNewsSentiment {self.ticker} {self.title[:50]}>'
# --- [END OF NEW MODELS] ---

# --- [END REFACTOR STEP 2] ---


@login_manager.user_loader
def load_user(user_id):
    with server.app_context():
        return db.session.get(User, int(user_id))
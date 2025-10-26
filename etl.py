# etl.py (เวอร์ชันสมบูรณ์ - [FIXED] Job 1 ใช้ Batch UPSERT)
import logging
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf # Import yfinance
import numpy as np

# Import สิ่งที่จำเป็นจากโปรเจกต์ของเรา
from app import db, server, DimCompany, FactCompanySummary, FactDailyPrices # Import model ราคา
from data_handler import get_competitor_data, _get_logo_url # ยังใช้ Job 1
from constants import ALL_TICKERS_SORTED_BY_MC

# Import สำหรับ UPSERT (เลือกใช้ตาม DB ที่ Deploy จริง)
try:
    # พยายาม Import สำหรับ PostgreSQL ก่อน (สำหรับ Render)
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    sql_insert = pg_insert # ใช้ตัวแปรกลาง
    db_dialect = 'postgresql'
    logging.info("Using PostgreSQL dialect for UPSERT.")
except ImportError:
    # ถ้า Import ไม่ได้ ให้ลอง Import สำหรับ SQLite (สำหรับ Local)
    try:
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert
        sql_insert = sqlite_insert # ใช้ตัวแปรกลาง
        db_dialect = 'sqlite'
        logging.info("Using SQLite dialect for UPSERT.")
    except ImportError:
        sql_insert = None
        db_dialect = None
        logging.error("Could not import insert statement for PostgreSQL or SQLite.")


# ตั้งค่า logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def update_company_summaries():
    """
    ETL Job 1: ดึงข้อมูลสรุปบริษัททั้งหมด
    (ข้อมูลจาก get_competitor_data)
    """
    # --- [NEW] Check if sql_insert is available ---
    if sql_insert is None:
        logging.error("ETL Job: [update_company_summaries] cannot run because insert statement is not available.")
        return

    with server.app_context():
        logging.info("Starting ETL Job: [update_company_summaries]...")
        try:
            tickers_tuple = tuple(ALL_TICKERS_SORTED_BY_MC)
            df = get_competitor_data(tickers_tuple)
            if df.empty:
                logging.warning("ETL Job: get_competitor_data returned an empty DataFrame. Skipping update.")
                return
        except Exception as e:
            logging.error(f"ETL Job: Failed during data extraction (get_competitor_data): {e}", exc_info=True)
            return

        today = datetime.utcnow().date()
        
        # --- [START OF REFACTOR] ---
        # 1. รวบรวมข้อมูลใส่ List ก่อน (แทนการ merge ใน loop)
        dim_company_data = []
        fact_summary_data = []

        for _, row in df.iterrows():
            if pd.isna(row.get('Ticker')):
                continue
                
            try:
                # --- ดึงข้อมูล DimCompany ---
                try:
                    ticker_info = yf.Ticker(row['Ticker']).info
                    company_name = ticker_info.get('longName')
                    logo_url = _get_logo_url(ticker_info) # <--- [FIXED] ใช้ฟังก์ชัน helper
                    sector = ticker_info.get('sector')
                except Exception:
                    company_name = None; logo_url = None; sector = None
                    logging.warning(f"Could not fetch full info for {row['Ticker']} for DimCompany.")

                dim_company_data.append({
                    'ticker': row['Ticker'],
                    'company_name': company_name,
                    'logo_url': logo_url,
                    'sector': sector
                })
                
                # --- เตรียมข้อมูล FactCompanySummary ---
                fact_summary_data.append({
                    'ticker': row['Ticker'], 'date_updated': today,
                    'price': row.get('Price'), 'market_cap': row.get('Market Cap'), 'beta': row.get('Beta'),
                    'pe_ratio': row.get('P/E'), 'pb_ratio': row.get('P/B'), 'ev_ebitda': row.get('EV/EBITDA'),
                    'revenue_growth_yoy': row.get('Revenue Growth (YoY)'), 'revenue_cagr_3y': row.get('Revenue CAGR (3Y)'),
                    'net_income_growth_yoy': row.get('Net Income Growth (YoY)'), 'roe': row.get('ROE'),
                    'de_ratio': row.get('D/E Ratio'), 'operating_margin': row.get('Operating Margin'),
                    'cash_conversion': row.get('Cash Conversion')
                })
            
            except Exception as e:
                logging.error(f"ETL Job: Failed to process row data for ticker {row.get('Ticker')}: {e}")
                continue # ข้าม Ticker นี้ไป แต่ทำตัวอื่นต่อ

        # 2. สั่ง Batch UPSERT (เหมือน Job 2)
        try:
            # --- UPSERT DimCompany ---
            if dim_company_data:
                stmt_dim = sql_insert(DimCompany).values(dim_company_data)
                dim_set_ = {
                    'company_name': stmt_dim.excluded.company_name,
                    'logo_url': stmt_dim.excluded.logo_url,
                    'sector': stmt_dim.excluded.sector
                }
                if db_dialect == 'postgresql':
                    on_conflict_dim = stmt_dim.on_conflict_do_update(constraint='dim_company_pkey', set_=dim_set_)
                elif db_dialect == 'sqlite':
                    on_conflict_dim = stmt_dim.on_conflict_do_update(index_elements=['ticker'], set_=dim_set_)
                
                db.session.execute(on_conflict_dim)
                logging.info(f"ETL Job: Batch UPSERTED {len(dim_company_data)} rows to DimCompany.")

            # --- UPSERT FactCompanySummary ---
            if fact_summary_data:
                stmt_fact = sql_insert(FactCompanySummary).values(fact_summary_data)
                constraint_name = '_ticker_date_uc' # ชื่อที่เราตั้งใน Model
                
                # ดึงชื่อคอลัมน์ทั้งหมดในตาราง FactCompanySummary (ยกเว้น keys)
                all_cols = {c.name for c in FactCompanySummary.__table__.columns if c.name not in ['id', 'ticker', 'date_updated']}
                fact_set_ = {col: getattr(stmt_fact.excluded, col) for col in all_cols}

                if db_dialect == 'postgresql':
                    on_conflict_fact = stmt_fact.on_conflict_do_update(constraint=constraint_name, set_=fact_set_)
                elif db_dialect == 'sqlite':
                    on_conflict_fact = stmt_fact.on_conflict_do_update(index_elements=['ticker', 'date_updated'], set_=fact_set_)
                
                db.session.execute(on_conflict_fact)
                logging.info(f"ETL Job: Batch UPSERTED {len(fact_summary_data)} rows to FactCompanySummary.")

            # --- Commit Transaction ---
            db.session.commit()
            logging.info(f"ETL Job: [update_company_summaries]... COMPLETED. Processed {len(fact_summary_data)} tickers.")

        except Exception as e:
            logging.error(f"ETL Job: Failed during final batch UPSERT: {e}", exc_info=True)
            db.session.rollback()
            
    # --- [END OF REFACTOR] ---


def update_daily_prices(days_back=5*365): # ดึงย้อนหลัง 5 ปีโดย default
    """
    ETL Job 2: ดึงข้อมูลราคาย้อนหลังและ UPSERT ลง FactDailyPrices
    """
    if sql_insert is None:
        logging.error("ETL Job: [update_daily_prices] cannot run because insert statement is not available.")
        return

    with server.app_context():
        logging.info(f"Starting ETL Job: [update_daily_prices] for {days_back} days back...")
        # ดึง Ticker ทั้งหมดจาก DimCompany ที่ Job 1 เพิ่งเพิ่มเข้าไป
        try:
            tickers_to_update = db.session.query(DimCompany.ticker).all()
            tickers_list = [t[0] for t in tickers_to_update]
        except Exception as e:
            logging.error(f"ETL Job: Failed to query tickers from DimCompany: {e}", exc_info=True)
            return

        if not tickers_list:
            logging.warning("ETL Job: No companies found in DimCompany. Skipping price update.")
            return

        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=days_back)

        # ดึงข้อมูลจาก yfinance
        prices_df = pd.DataFrame() # Initialize empty
        try:
            logging.info(f"ETL Job: Downloading price history for {len(tickers_list)} tickers from {start_date} to {end_date}...")
            # auto_adjust=True ให้ราคา Adj Close ใน 'Close'
            prices_df = yf.download(tickers_list, start=start_date, end=end_date, auto_adjust=True, progress=False)
            logging.info("ETL Job: Price history download complete.")

            if prices_df.empty:
                logging.warning("ETL Job: yf.download returned empty DataFrame for prices.")
                return

            # --- Data Processing ---
            # ใช้ stack level=1 เพราะ yfinance ใหม่ๆ อาจมี multi-index ที่ level 0 เป็น metric ('Open', 'High', etc.)
            if isinstance(prices_df.columns, pd.MultiIndex):
                 prices_df = prices_df.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()
            else: # Handle single ticker case
                 prices_df = prices_df.rename_axis(['Date']).reset_index()
                 # If only one ticker was downloaded, yfinance might not add Ticker column
                 if 'Ticker' not in prices_df.columns and len(tickers_list) == 1:
                     prices_df['Ticker'] = tickers_list[0]

            prices_df.rename(columns={
                'Date': 'date', 'Ticker': 'ticker', 'Open': 'open',
                'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            }, inplace=True)

            # Convert date to date object if it's datetime
            prices_df['date'] = pd.to_datetime(prices_df['date']).dt.date

            # Clean data types and NaNs
            prices_df['volume'] = pd.to_numeric(prices_df['volume'], errors='coerce').fillna(0).astype(np.int64) # Use np.int64 for safety
            prices_df.dropna(subset=['close', 'ticker', 'date'], inplace=True) # Ensure essential columns are not null

        except Exception as e:
            logging.error(f"ETL Job: Failed during yf.download or processing for prices: {e}", exc_info=True)
            return

        # --- UPSERT ข้อมูลลง DB ---
        if not prices_df.empty:
            num_rows = len(prices_df)
            logging.info(f"ETL Job: Attempting to UPSERT {num_rows} price rows using {db_dialect} dialect...")
            data_to_upsert = prices_df[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']].to_dict(orient='records')

            if not data_to_upsert:
                 logging.warning("ETL Job: No price data to UPSERT after processing.")
                 return

            try:
                # สร้าง statement สำหรับ UPSERT
                stmt = sql_insert(FactDailyPrices).values(data_to_upsert)
                constraint_name = '_ticker_price_date_uc' # ชื่อที่เราตั้งใน Model
                
                price_set_ = {
                    'open': stmt.excluded.open,
                    'high': stmt.excluded.high,
                    'low': stmt.excluded.low,
                    'close': stmt.excluded.close,
                    'volume': stmt.excluded.volume
                }

                if db_dialect == 'postgresql':
                    # PostgreSQL ON CONFLICT DO UPDATE
                    on_conflict_stmt = stmt.on_conflict_do_update(
                        constraint=constraint_name,
                        set_=price_set_
                    )
                elif db_dialect == 'sqlite':
                     # SQLite ON CONFLICT DO UPDATE (Syntax ใหม่กว่า)
                     # หมายเหตุ: อาจต้องการ SQLite version 3.24+
                    on_conflict_stmt = stmt.on_conflict_do_update(
                         index_elements=['ticker', 'date'], # ระบุคอลัมน์ใน unique constraint
                         set_=price_set_
                     )
                else:
                    logging.error("ETL Job: Unsupported DB dialect for UPSERT.")
                    return

                # Execute statement in chunks to avoid memory issues if needed
                chunk_size = 5000 # ปรับขนาดตามความเหมาะสม
                for i in range(0, len(data_to_upsert), chunk_size):
                    chunk = data_to_upsert[i:i + chunk_size]
                    stmt_chunk = sql_insert(FactDailyPrices).values(chunk)
                    
                    price_set_chunk_ = {
                        'open': stmt_chunk.excluded.open, 'high': stmt_chunk.excluded.high,
                        'low': stmt_chunk.excluded.low, 'close': stmt_chunk.excluded.close,
                        'volume': stmt_chunk.excluded.volume
                    }

                    if db_dialect == 'postgresql':
                         on_conflict_chunk = stmt_chunk.on_conflict_do_update(constraint=constraint_name, set_=price_set_chunk_)
                    elif db_dialect == 'sqlite':
                         on_conflict_chunk = stmt_chunk.on_conflict_do_update(index_elements=['ticker', 'date'], set_=price_set_chunk_)
                    else: break # Should not happen

                    db.session.execute(on_conflict_chunk)
                    # logging.info(f"ETL Job: Executed UPSERT chunk {i // chunk_size + 1}...") # Comment out
                
                db.session.commit()
                logging.info(f"ETL Job: [update_daily_prices]... COMPLETED. Successfully processed UPSERT for {num_rows} rows potential.")

            except Exception as e:
                logging.error(f"ETL Job: Failed during database UPSERT for prices: {e}", exc_info=True)
                db.session.rollback()
        else:
             logging.warning("ETL Job: No price data prepared for UPSERT.")
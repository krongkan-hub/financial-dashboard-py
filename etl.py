# etl.py (เวอร์ชันสมบูรณ์ - [FIXED] Rate Limiting with Retry & Progress)
import logging
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf # Import yfinance
import numpy as np
import time # Import time for delays

# Import สิ่งที่จำเป็นจากโปรเจกต์ของเรา
from app import db, server, DimCompany, FactCompanySummary, FactDailyPrices, FactFinancialStatements, FactNewsSentiment
from data_handler import get_competitor_data, get_deep_dive_data, get_news_and_sentiment
from constants import ALL_TICKERS_SORTED_BY_MC

# Import สำหรับ UPSERT (เลือกใช้ตาม DB ที่ Deploy จริง)
try:
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    sql_insert = pg_insert # ใช้ตัวแปรกลาง
    db_dialect = 'postgresql'
    logging.info("Using PostgreSQL dialect for UPSERT.")
except ImportError:
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

# --- Helper Function for Retry and Progress ---
def process_tickers_with_retry(job_name, items_list, process_func, initial_delay=0.5, retry_delay=60, max_retries=3):
    """
    วน Loop ผ่านรายการ (tickers หรือ tuples) พร้อม Retry, Delay และ Progress Tracking.

    Args:
        job_name (str): ชื่อ Job สำหรับ Logging.
        items_list (list): ลิสต์ของ items (เช่น tickers หรือ tuples (ticker, name)).
        process_func (function): ฟังก์ชันที่จะเรียกใช้สำหรับแต่ละ item (รับ item เป็น argument).
        initial_delay (float): เวลา (วินาที) ที่จะหน่วงหลังการประมวลผลแต่ละ item (ถ้าสำเร็จ).
        retry_delay (int): เวลา (วินาที) ที่จะหน่วงก่อนลองใหม่ (ถ้าเกิด Error).
        max_retries (int): จำนวนครั้งสูงสุดที่จะลองใหม่.
    """
    total_items = len(items_list)
    processed_count = 0
    skipped_items = []
    start_time_job = time.time()

    logging.info(f"--- Starting {job_name} for {total_items} items ---")

    for i, item_data in enumerate(items_list):
        # Determine the identifier (ticker) for logging purposes
        if isinstance(item_data, tuple):
            identifier = item_data[0] # Assume ticker is the first element
        else:
            identifier = item_data # Assume the item itself is the ticker

        retries = 0
        success = False
        while retries < max_retries and not success:
            try:
                # --- เรียกฟังก์ชันประมวลผลหลัก ---
                process_func(item_data) # Pass the whole item (ticker or tuple)
                # ---------------------------------
                success = True
                processed_count += 1
                logging.info(f"[{job_name}] Successfully processed {identifier} ({i + 1}/{total_items})")
                # หน่วงเวลาก่อนไปตัวถัดไป (ถ้าสำเร็จ)
                time.sleep(initial_delay)

            except Exception as e:
                retries += 1
                error_msg = str(e)
                # Check for common rate limit indicators (adjust based on actual errors observed)
                is_rate_limit = "Too Many Requests" in error_msg or "429" in error_msg
                log_level = logging.WARNING if is_rate_limit else logging.ERROR
                
                logging.log(log_level, f"[{job_name}] Error processing {identifier} (Attempt {retries}/{max_retries}): {error_msg}")
                
                if retries < max_retries:
                    current_retry_delay = retry_delay * (2 ** (retries - 1)) # Exponential backoff
                    logging.info(f"[{job_name}] Retrying {identifier} after {current_retry_delay} seconds...")
                    time.sleep(current_retry_delay)
                else:
                    logging.error(f"[{job_name}] Max retries reached for {identifier}. Skipping.")
                    skipped_items.append(identifier)
                    # หน่วงเวลาสั้นๆ ก่อนไปตัวถัดไป (แม้จะ Fail)
                    time.sleep(initial_delay)

        # แสดง Progress ทุกๆ 10% หรือทุกๆ 50 ตัว (ปรับปรุงให้แสดงผลบ่อยขึ้น)
        if (i + 1) % 10 == 0 or (i + 1) == total_items: # Show progress every 10 items or at the end
             progress = (i + 1) / total_items * 100
             elapsed_time = time.time() - start_time_job
             logging.info(f"[{job_name}] Progress: {i + 1}/{total_items} ({progress:.1f}%) processed. Elapsed: {elapsed_time:.2f}s")

    elapsed_total_job = time.time() - start_time_job
    logging.info(f"--- Finished {job_name} in {elapsed_total_job:.2f} seconds ---")
    logging.info(f"Successfully processed: {processed_count}/{total_items}")
    if skipped_items:
        logging.warning(f"Skipped tickers due to errors: {skipped_tickers}")
    return skipped_items


# --- Job 1: Update Company Summaries (Modified to use helper) ---
def update_company_summaries():
    if sql_insert is None:
        logging.error("ETL Job: [update_company_summaries] cannot run because insert statement is not available.")
        return

    job_name = "Job 1: Update Summaries"
    tickers_tuple = tuple(ALL_TICKERS_SORTED_BY_MC)
    today = datetime.utcnow().date()

    def process_single_ticker_summary(ticker):
        # Function remains largely the same as previous example,
        # focusing on fetching data for ONE ticker and UPSERTING it.
        try:
            df_single = get_competitor_data((ticker,)) # Fetch data for only this ticker
            if df_single.empty:
                logging.warning(f"[{job_name}] get_competitor_data returned empty for {ticker}. Skipping DB insert.")
                return # Don't raise error, just skip

            row = df_single.iloc[0]

            dim_data = {
                'ticker': row['Ticker'],
                'company_name': row.get('company_name'),
                'logo_url': row.get('logo_url'),
                'sector': row.get('sector')
            }
            fact_data = {
                'ticker': row['Ticker'], 'date_updated': today,
                'price': row.get('Price'), 'market_cap': row.get('Market Cap'), 'beta': row.get('Beta'),
                'pe_ratio': row.get('P/E'), 'pb_ratio': row.get('P/B'), 'ev_ebitda': row.get('EV/EBITDA'),
                'revenue_growth_yoy': row.get('Revenue Growth (YoY)'), 'revenue_cagr_3y': row.get('Revenue CAGR (3Y)'),
                'net_income_growth_yoy': row.get('Net Income Growth (YoY)'), 'roe': row.get('ROE'),
                'de_ratio': row.get('D/E Ratio'), 'operating_margin': row.get('Operating Margin'),
                'cash_conversion': row.get('Cash Conversion'),
                'ebitda_margin': row.get('EBITDA Margin'),
                'trailing_eps': row.get('Trailing EPS'),
                'forward_pe': row.get('Forward P/E'),
                'analyst_target_price': row.get('Analyst Target Price'),
                'long_business_summary': row.get('Long Business Summary')
            }

            with server.app_context():
                # UPSERT DimCompany
                stmt_dim = sql_insert(DimCompany).values(dim_data)
                dim_set_ = { 'company_name': stmt_dim.excluded.company_name, 'logo_url': stmt_dim.excluded.logo_url, 'sector': stmt_dim.excluded.sector }
                if db_dialect == 'postgresql':
                    on_conflict_dim = stmt_dim.on_conflict_do_update(constraint='dim_company_pkey', set_=dim_set_)
                elif db_dialect == 'sqlite':
                    on_conflict_dim = stmt_dim.on_conflict_do_update(index_elements=['ticker'], set_=dim_set_)
                db.session.execute(on_conflict_dim)

                # UPSERT FactCompanySummary
                stmt_fact = sql_insert(FactCompanySummary).values(fact_data)
                all_cols = {c.name for c in FactCompanySummary.__table__.columns if c.name not in ['id', 'ticker', 'date_updated']}
                fact_set_ = {col: getattr(stmt_fact.excluded, col) for col in all_cols}
                if db_dialect == 'postgresql':
                    on_conflict_fact = stmt_fact.on_conflict_do_update(constraint='_ticker_date_uc', set_=fact_set_)
                elif db_dialect == 'sqlite':
                    on_conflict_fact = stmt_fact.on_conflict_do_update(index_elements=['ticker', 'date_updated'], set_=fact_set_)
                db.session.execute(on_conflict_fact)

                db.session.commit()

        except Exception as inner_e:
            logging.error(f"[{job_name}] Error during processing/DB UPSERT for {ticker}: {inner_e}", exc_info=True)
            with server.app_context(): # Ensure rollback happens within context
                 db.session.rollback()
            raise inner_e # Re-raise to trigger retry logic in helper

    # Use the helper function
    process_tickers_with_retry(job_name, list(tickers_tuple), process_single_ticker_summary, initial_delay=0.7, max_retries=3)


# --- Job 2: Update Daily Prices (Unchanged, less likely to hit rate limits) ---
def update_daily_prices(days_back=5*365):
    """
    ETL Job 2: ดึงข้อมูลราคาย้อนหลังและ UPSERT ลง FactDailyPrices
    """
    if sql_insert is None:
        logging.error("ETL Job: [update_daily_prices] cannot run because insert statement is not available.")
        return

    with server.app_context():
        logging.info(f"Starting ETL Job: [update_daily_prices] for {days_back} days back...")
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

        prices_df = pd.DataFrame() # Initialize empty
        try:
            logging.info(f"ETL Job: Downloading price history for {len(tickers_list)} tickers from {start_date} to {end_date}...")
            # yfinance download handles multiple tickers efficiently
            prices_df = yf.download(tickers_list, start=start_date, end=end_date, auto_adjust=True, progress=False)
            logging.info("ETL Job: Price history download complete.")

            if prices_df.empty:
                logging.warning("ETL Job: yf.download returned empty DataFrame for prices.")
                return

            # Data Processing
            if isinstance(prices_df.columns, pd.MultiIndex) and len(tickers_list) > 1:
                # Handle MultiIndex for multiple tickers
                prices_df = prices_df.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()
            elif not prices_df.empty: # Handle single ticker case or already processed format
                 prices_df = prices_df.rename_axis(['Date']).reset_index()
                 if 'Ticker' not in prices_df.columns and len(tickers_list) == 1:
                     prices_df['Ticker'] = tickers_list[0]
            else: # Empty DataFrame remains empty
                 pass

            prices_df.rename(columns={
                'Date': 'date', 'Ticker': 'ticker', 'Open': 'open',
                'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            }, inplace=True)

            if 'date' in prices_df.columns:
                prices_df['date'] = pd.to_datetime(prices_df['date']).dt.date
            else:
                 logging.warning("ETL Job: 'date' column not found after processing yfinance download.")
                 return # Cannot proceed without date

            if 'volume' in prices_df.columns:
                prices_df['volume'] = pd.to_numeric(prices_df['volume'], errors='coerce').fillna(0).astype(np.int64)
            else:
                prices_df['volume'] = 0 # Add volume column if missing

            # Ensure all required columns exist before dropping NA
            required_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in prices_df.columns:
                     prices_df[col] = np.nan # Add missing columns with NaN

            prices_df = prices_df[required_cols] # Ensure column order
            prices_df.dropna(subset=['close', 'ticker', 'date'], inplace=True) # Ensure essential columns are not null


        except Exception as e:
            logging.error(f"ETL Job: Failed during yf.download or processing for prices: {e}", exc_info=True)
            return

        # --- UPSERT ข้อมูลลง DB ---
        if not prices_df.empty:
            num_rows = len(prices_df)
            logging.info(f"ETL Job: Attempting to UPSERT {num_rows} price rows using {db_dialect} dialect...")
            data_to_upsert = prices_df.to_dict(orient='records')

            if not data_to_upsert:
                 logging.warning("ETL Job: No price data to UPSERT after processing.")
                 return

            try:
                stmt = sql_insert(FactDailyPrices).values(data_to_upsert)
                constraint_name = '_ticker_price_date_uc' 
                
                price_set_ = {
                    'open': stmt.excluded.open, 'high': stmt.excluded.high,
                    'low': stmt.excluded.low, 'close': stmt.excluded.close,
                    'volume': stmt.excluded.volume
                }

                if db_dialect == 'postgresql':
                    on_conflict_stmt = stmt.on_conflict_do_update(constraint=constraint_name, set_=price_set_)
                elif db_dialect == 'sqlite':
                    on_conflict_stmt = stmt.on_conflict_do_update(index_elements=['ticker', 'date'], set_=price_set_)
                else:
                    logging.error("ETL Job: Unsupported DB dialect for UPSERT.")
                    return

                # Execute in chunks (improves performance for large datasets)
                chunk_size = 5000 
                for i in range(0, len(data_to_upsert), chunk_size):
                    chunk = data_to_upsert[i:i + chunk_size]
                    # Create a new statement for each chunk to avoid parameter issues
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
                    else: break 
                    db.session.execute(on_conflict_chunk)
                    # Optional: Log chunk progress
                    # logging.info(f"ETL Job: UPSERTED chunk {i // chunk_size + 1}...")
                
                db.session.commit()
                logging.info(f"ETL Job: [update_daily_prices]... COMPLETED. Successfully processed UPSERT for {num_rows} rows potential.")

            except Exception as e:
                logging.error(f"ETL Job: Failed during database UPSERT for prices: {e}", exc_info=True)
                db.session.rollback()
        else:
             logging.warning("ETL Job: No price data prepared for UPSERT.")

# --- [START OF NEW JOB 3 - Updated with Helper] ---
def update_financial_statements():
    """
    ETL Job 3: ดึงข้อมูลงบการเงินรายไตรมาส (จาก get_deep_dive_data)
    และแปลงเป็น Long Format เก็บลง FactFinancialStatements
    """
    if sql_insert is None:
        logging.error("ETL Job: [update_financial_statements] cannot run because insert statement is not available.")
        return

    job_name = "Job 3: Update Financials"
    tickers_list = []
    with server.app_context():
        try:
            tickers_list = [t[0] for t in db.session.query(DimCompany.ticker).all()]
        except Exception as e:
            logging.error(f"[{job_name}] Failed to query tickers from DimCompany: {e}", exc_info=True)
            return

    def process_single_ticker_financials(ticker):
        try:
            # 1. ดึงข้อมูล (ใช้ cache จาก data_handler)
            # Make sure get_deep_dive_data includes a slight delay if not cached
            data = get_deep_dive_data(ticker)
            if 'error' in data or 'financial_statements' not in data:
                logging.warning(f"[{job_name}] No financial statement data found via get_deep_dive_data for {ticker}.")
                return # Treat as processed, skip DB part

            statements = data.get("financial_statements", {})
            all_statements_data = []

            # 2. วน Loop งบแต่ละประเภท (income, balance, cashflow)
            for statement_type in ['income', 'balance', 'cashflow']:
                df = statements.get(statement_type)
                if df is None or df.empty:
                    continue
                
                # Make sure index is named before melting if it's not already
                if df.index.name is None:
                    df.index.name = 'metric_name'
                
                # แปลงจาก Wide -> Long format
                # Ensure columns are Datetime objects if needed, else handle potential errors
                try:
                    df.columns = pd.to_datetime(df.columns)
                except (ValueError, TypeError) as date_err:
                     logging.warning(f"[{job_name}] Could not convert columns to datetime for {ticker}, {statement_type}: {date_err}. Skipping this statement.")
                     continue # Skip this statement type if columns aren't dates

                df_long = df.reset_index().melt(id_vars='metric_name', var_name='report_date', value_name='metric_value')
                
                # 4. เพิ่มคอลัมน์ที่จำเป็นสำหรับ DB
                df_long['ticker'] = ticker
                df_long['statement_type'] = statement_type
                df_long['report_date'] = pd.to_datetime(df_long['report_date']).dt.date
                df_long.dropna(subset=['metric_value'], inplace=True) # ไม่เก็บค่า Metric ที่เป็น NaT/None
                
                # Convert potential Pandas dtypes to standard Python types for SQLAlchemy
                df_long['metric_value'] = df_long['metric_value'].astype(float) # Ensure float
                
                all_statements_data.extend(df_long.to_dict('records'))

            # 5. UPSERT ข้อมูล (ทำภายใน process_single_ticker_financials)
            if all_statements_data:
                with server.app_context():
                    stmt = sql_insert(FactFinancialStatements).values(all_statements_data)
                    set_ = {'metric_value': stmt.excluded.metric_value}
                    constraint_name = '_ticker_date_statement_metric_uc'
                    
                    if db_dialect == 'postgresql':
                        on_conflict_stmt = stmt.on_conflict_do_update(constraint=constraint_name, set_=set_)
                    elif db_dialect == 'sqlite':
                        on_conflict_stmt = stmt.on_conflict_do_update(index_elements=['ticker', 'report_date', 'statement_type', 'metric_name'], set_=set_)
                    else:
                         raise Exception("Unsupported DB dialect for UPSERT.") # Raise error here
                    
                    db.session.execute(on_conflict_stmt)
                    db.session.commit()
            # No logging here, it's handled by the helper function on success

        except Exception as inner_e:
            logging.error(f"[{job_name}] Error during processing/DB UPSERT for {ticker}: {inner_e}", exc_info=True)
            with server.app_context():
                db.session.rollback()
            raise inner_e # โยนให้ Helper จัดการ Retry

    # เรียกใช้ Helper Function
    process_tickers_with_retry(job_name, tickers_list, process_single_ticker_financials, initial_delay=0.8, max_retries=2) # Slightly longer delay maybe needed

# --- [END OF NEW JOB 3] ---


# --- [START OF NEW JOB 4] ---
def update_news_sentiment():
    """
    ETL Job 4: ดึงข้อมูลข่าวและ Sentiment (จาก get_news_and_sentiment)
    และเก็บลง FactNewsSentiment
    (แก้ไข: ใช้ Helper ใหม่)
    """
    if sql_insert is None:
        logging.error("ETL Job: [update_news_sentiment] cannot run because insert statement is not available.")
        return

    job_name = "Job 4: Update News/Sentiment"
    companies_list = []
    with server.app_context():
        try:
            # ดึง list of tuples (ticker, company_name)
            companies_list = db.session.query(DimCompany.ticker, DimCompany.company_name).filter(DimCompany.company_name != None).all()
        except Exception as e:
            logging.error(f"[{job_name}] Failed to query companies from DimCompany: {e}", exc_info=True)
            return

    # ฟังก์ชันย่อยสำหรับประมวลผล Ticker เดียว
    def process_single_ticker_news(company_data_tuple):
        ticker, company_name = company_data_tuple # Unpack the tuple

        if not company_name:
            logging.warning(f"[{job_name}] Skipping news for {ticker}, no company name found.")
            return True # Treat as processed successfully

        try:
            # 1. ดึงข่าวและวิเคราะห์ (ใช้ cache จาก data_handler)
            # Consider adding a small delay *before* the API call if needed
            # time.sleep(0.1) # Small delay before NewsAPI/HF calls
            data = get_news_and_sentiment(company_name)
            if 'error' in data or 'articles' not in data or not data['articles']:
                logging.warning(f"[{job_name}] No news articles found or error fetching for {company_name} ({ticker}). Error: {data.get('error')}")
                return True # Treat as processed successfully

            articles_data = []
            # 2. รวบรวมข้อมูลสำหรับ DB
            for article in data['articles']:
                try:
                    # Check if essential fields exist and are not None
                    if not all(k in article and article[k] is not None for k in ['publishedAt', 'title', 'url']):
                        logging.warning(f"[{job_name}] Skipping article for {ticker} due to missing data: {article.get('title')}")
                        continue

                    published_dt_str = article['publishedAt']
                    # Handle potential timezone variations or None values from API more robustly
                    if isinstance(published_dt_str, str):
                        if published_dt_str.endswith('Z'):
                             published_dt = datetime.fromisoformat(published_dt_str.replace("Z", "+00:00"))
                        else:
                             # Attempt parsing without timezone if 'Z' is missing
                             try:
                                 published_dt = datetime.fromisoformat(published_dt_str)
                             except ValueError:
                                 logging.warning(f"[{job_name}] Could not parse datetime '{published_dt_str}' for article URL {article.get('url')}. Skipping article.")
                                 continue
                    elif isinstance(published_dt_str, datetime): # Already a datetime object
                        published_dt = published_dt_str
                    else:
                        logging.warning(f"[{job_name}] Invalid publishedAt format for article URL {article.get('url')}. Skipping article.")
                        continue


                    articles_data.append({
                        'ticker': ticker,
                        'published_at': published_dt,
                        'title': article.get('title'),
                        'article_url': article.get('url'), # ใช้เป็น Key กันซ้ำ
                        'source_name': article.get('source', {}).get('name'),
                        'description': article.get('description'), # Might be None, handle in DB or UI
                        'sentiment_label': article.get('sentiment'),
                        'sentiment_score': article.get('sentiment_score')
                    })
                except Exception as parse_e:
                    logging.warning(f"[{job_name}] Skipping article for {ticker} due to processing error: {parse_e}")
                    continue # ข้าม article ที่มีปัญหา

            # 3. UPSERT ข้อมูล (Insert or Ignore based on article_url)
            if articles_data:
                with server.app_context():
                    stmt = sql_insert(FactNewsSentiment).values(articles_data)
                    
                    # ON CONFLICT DO NOTHING based on the unique constraint on article_url
                    if db_dialect == 'postgresql':
                        # Assumes a unique constraint named 'fact_news_sentiment_article_url_key' exists
                        # You might need to create this constraint manually or via migration:
                        # ALTER TABLE fact_news_sentiment ADD CONSTRAINT fact_news_sentiment_article_url_key UNIQUE (article_url);
                        on_conflict_stmt = stmt.on_conflict_do_nothing(constraint='fact_news_sentiment_article_url_key')
                    elif db_dialect == 'sqlite':
                         # SQLite requires specifying the indexed column for ON CONFLICT
                        on_conflict_stmt = stmt.on_conflict_do_nothing(index_elements=['article_url'])
                    else:
                        raise Exception("Unsupported DB dialect for UPSERT with DO NOTHING.")

                    db.session.execute(on_conflict_stmt)
                    db.session.commit()
            # No specific logging here, success is logged by the helper

            return True # บอกว่าสำเร็จ

        except Exception as inner_e:
            # Log specific errors related to NewsAPI or Hugging Face if possible
            logging.error(f"[{job_name}] Error during processing/DB UPSERT for {ticker} ({company_name}): {inner_e}", exc_info=True)
            with server.app_context():
                db.session.rollback()
            raise inner_e # โยนให้ Helper จัดการ Retry

    # เรียกใช้ Helper Function (ส่ง list of tuples)
    # News APIs can also have rate limits, slightly longer delay might be safer
    process_tickers_with_retry(job_name, companies_list, process_single_ticker_news, initial_delay=1.0, retry_delay=90, max_retries=2)
# --- [END OF NEW JOB 4] ---
# etl.py (เวอร์ชันสมบูรณ์ - [FIXED] Rate Limiting with Retry & Progress & Batch UPSERT)
import logging
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf # Import yfinance
import numpy as np
import time # Import time for delays
from typing import Dict # Import Dict for type hinting

# Import สิ่งที่จำเป็นจากโปรเจกต์ของเรา
# ตรวจสอบว่า FactDailyPrices ถูก Import ถูกต้อง
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


# --- [NEW HELPER] FIX: Convert numpy types to Python native types for psycopg2 ---
def _clean_numpy_types(data_dict: Dict) -> Dict:
    """Converts numpy types (int64, float64, np.number) to native Python types (float, None)."""
    cleaned_item = {}
    for key, value in data_dict.items():
        if pd.isna(value) or value is None:
            cleaned_item[key] = None
        # Check for any NumPy number type (np.int64, np.float64, np.int32, etc.)
        elif isinstance(value, (np.int64, np.float64, np.number)):
            # Convert to standard Python float for consistency with DB schema (FactCompanySummary mostly uses Float)
            cleaned_item[key] = float(value)
        else:
            cleaned_item[key] = value
    return cleaned_item
# ---------------------------------------------------------------------------------


# --- Helper Function for Retry and Progress (ไม่เปลี่ยนแปลง) ---
def process_tickers_with_retry(job_name, items_list, process_func, initial_delay=0.5, retry_delay=60, max_retries=3):
    """
    วน Loop ผ่านรายการ (tickers หรือ tuples) พร้อม Retry, Delay และ Progress Tracking.
    """
    total_items = len(items_list)
    processed_count = 0
    skipped_items = []
    start_time_job = time.time()

    logging.info(f"--- Starting {job_name} for {total_items} items ---")

    for i, item_data in enumerate(items_list):
        if isinstance(item_data, tuple):
            identifier = item_data[0] 
        else:
            identifier = item_data 

        retries = 0
        success = False
        while retries < max_retries and not success:
            try:
                process_func(item_data) 
                success = True
                processed_count += 1
                logging.info(f"[{job_name}] Successfully processed {identifier} ({i + 1}/{total_items})")
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
                    time.sleep(initial_delay)

        if (i + 1) % 10 == 0 or (i + 1) == total_items: 
             progress = (i + 1) / total_items * 100
             elapsed_time = time.time() - start_time_job
             logging.info(f"[{job_name}] Progress: {i + 1}/{total_items} ({progress:.1f}%) processed. Elapsed: {elapsed_time:.2f}s")

    elapsed_total_job = time.time() - start_time_job
    logging.info(f"--- Finished {job_name} in {elapsed_total_job:.2f} seconds ---")
    logging.info(f"Successfully processed: {processed_count}/{total_items}")
    if skipped_items:
        logging.warning(f"Skipped tickers due to errors: {skipped_items}") 
    return skipped_items


# --- Job 1: Update Company Summaries (Modified) ---
def update_company_summaries():
    if sql_insert is None:
        logging.error("ETL Job: [update_company_summaries] cannot run because insert statement is not available.")
        return

    job_name = "Job 1: Update Summaries"
    tickers_tuple = tuple(ALL_TICKERS_SORTED_BY_MC)
    today = datetime.utcnow().date()

    def process_single_ticker_summary(ticker):
        try:
            df_single = get_competitor_data((ticker,)) 
            if df_single.empty:
                logging.warning(f"[{job_name}] get_competitor_data returned empty for {ticker}. Skipping DB insert.")
                return 

            row = df_single.iloc[0]

            dim_data = {
                'ticker': row['Ticker'],
                'company_name': row.get('company_name'),
                'logo_url': row.get('logo_url'),
                'sector': row.get('sector')
            }
            # fact_data ยังมี NumPy types อยู่
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
                # --- APPLY THE FIX HERE: Clean NumPy types ---
                fact_data_clean = _clean_numpy_types(fact_data) 
                
                # UPSERT DimCompany
                stmt_dim = sql_insert(DimCompany).values(dim_data)
                dim_set_ = { 'company_name': stmt_dim.excluded.company_name, 'logo_url': stmt_dim.excluded.logo_url, 'sector': stmt_dim.excluded.sector }
                if db_dialect == 'postgresql':
                    on_conflict_dim = stmt_dim.on_conflict_do_update(constraint='dim_company_pkey', set_=dim_set_)
                elif db_dialect == 'sqlite':
                    on_conflict_dim = stmt_dim.on_conflict_do_update(index_elements=['ticker'], set_=dim_set_)
                db.session.execute(on_conflict_dim)

                # UPSERT FactCompanySummary (ใช้ fact_data_clean)
                stmt_fact = sql_insert(FactCompanySummary).values(fact_data_clean)
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
            with server.app_context(): 
                 db.session.rollback()
            raise inner_e 

    # Use the helper function
    process_tickers_with_retry(job_name, list(tickers_tuple), process_single_ticker_summary, initial_delay=0.7, max_retries=3)


# --- Job 2: Update Daily Prices (FIXED: Numpy Type Conversion) ---
def update_daily_prices(days_back=5*365):
    """
    ETL Job 2: ดึงข้อมูลราคาย้อนหลังและ UPSERT ลง FactDailyPrices
    """
    if sql_insert is None:
        logging.error("ETL Job: [update_daily_prices] cannot run because insert statement is not available.")
        return

    with server.app_context():
        logging.info(f"Starting ETL Job: [update_daily_prices] for {days_back} days back...")
        
        # 1. Query Tickers from DimCompany
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
        prices_df = pd.DataFrame()

        # 2. Download Price History
        try:
            logging.info(f"ETL Job: Downloading price history for {len(tickers_list)} tickers from {start_date} to {end_date}...")
            prices_df = yf.download(tickers_list, start=start_date, end=end_date, auto_adjust=True, progress=False)
            logging.info("ETL Job: Price history download complete.")

            if prices_df.empty:
                logging.warning("ETL Job: yf.download returned empty DataFrame for prices.")
                return

            # 3. Data Processing (Convert from Wide/MultiIndex to Long Format)
            if isinstance(prices_df.columns, pd.MultiIndex) and len(tickers_list) > 1:
                prices_df = prices_df.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()
            elif not prices_df.empty: 
                 prices_df = prices_df.rename_axis(['Date']).reset_index()
                 if 'Ticker' not in prices_df.columns and len(tickers_list) == 1:
                     prices_df['Ticker'] = tickers_list[0]
            else: 
                 pass

            # 4. Cleanup and Format
            prices_df.rename(columns={
                'Date': 'date', 'Ticker': 'ticker', 'Open': 'open',
                'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            }, inplace=True)

            if 'date' in prices_df.columns:
                prices_df['date'] = pd.to_datetime(prices_df['date']).dt.date
            else:
                 logging.warning("ETL Job: 'date' column not found after processing yfinance download.")
                 return

            # Remove the old .astype(np.int64) and convert to native Python types
            if 'volume' in prices_df.columns:
                prices_df['volume'] = pd.to_numeric(prices_df['volume'], errors='coerce').fillna(0) 
            else:
                prices_df['volume'] = 0 

            required_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in prices_df.columns:
                     prices_df[col] = np.nan

            prices_df = prices_df[required_cols]
            prices_df.dropna(subset=['close', 'ticker', 'date'], inplace=True)
            
            # --- APPLY THE FIX HERE: Convert to standard Python int/float ---
            # volume (BigInteger in DB) must be converted to Python int
            prices_df['volume'] = prices_df['volume'].apply(lambda x: int(x) if pd.notna(x) and x is not None else None)
            # price columns to standard Python float
            prices_df['open'] = prices_df['open'].apply(lambda x: float(x) if pd.notna(x) and x is not None else None)
            prices_df['high'] = prices_df['high'].apply(lambda x: float(x) if pd.notna(x) and x is not None else None)
            prices_df['low'] = prices_df['low'].apply(lambda x: float(x) if pd.notna(x) and x is not None else None)
            prices_df['close'] = prices_df['close'].apply(lambda x: float(x) if pd.notna(x) and x is not None else None)
            # -----------------------------------------------------------------
            
            data_to_upsert = prices_df.to_dict(orient='records')
            num_rows = len(data_to_upsert)


        except Exception as e:
            logging.error(f"ETL Job: Failed during yf.download or processing for prices: {e}", exc_info=True)
            return

        # 5. --- UPSERT ข้อมูลลง DB (Chunking Logic) ---
        if data_to_upsert:
            logging.info(f"ETL Job: Attempting to UPSERT {num_rows} price rows using {db_dialect} dialect...")
            
            chunk_size = 1000 
            constraint_name = '_ticker_price_date_uc'
            
            for i in range(0, num_rows, chunk_size):
                chunk = data_to_upsert[i:i + chunk_size]
                
                stmt_chunk = sql_insert(FactDailyPrices).values(chunk)
                
                price_set_chunk_ = {
                    'open': stmt_chunk.excluded.open, 'high': stmt_chunk.excluded.high,
                    'low': stmt_chunk.excluded.low, 'close': stmt_chunk.excluded.close,
                    'volume': stmt_chunk.excluded.volume
                }

                try:
                    if db_dialect == 'postgresql':
                        on_conflict_chunk = stmt_chunk.on_conflict_do_update(constraint=constraint_name, set_=price_set_chunk_)
                    elif db_dialect == 'sqlite':
                        on_conflict_chunk = stmt_chunk.on_conflict_do_update(index_elements=['ticker', 'date'], set_=price_set_chunk_)
                    else:
                        logging.error("ETL Job: Unsupported DB dialect for UPSERT in chunking loop.")
                        break 

                    db.session.execute(on_conflict_chunk)
                    db.session.commit() 
                    logging.info(f"ETL Job: Successfully UPSERTED chunk {i // chunk_size + 1}/{(num_rows // chunk_size) + 1} ({len(chunk)} rows).")

                except Exception as e:
                    logging.error(f"ETL Job: Failed during database UPSERT for chunk {i // chunk_size + 1}: {e}", exc_info=True)
                    db.session.rollback()
                    
            logging.info(f"ETL Job: [update_daily_prices]... COMPLETED. Successfully processed UPSERT for {num_rows} rows potential.")

        else:
             logging.warning("ETL Job: No price data prepared for UPSERT.")


# --- Job 3: update_financial_statements (ไม่เปลี่ยนแปลง) ---
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
            data = get_deep_dive_data(ticker)
            if 'error' in data or 'financial_statements' not in data:
                logging.warning(f"[{job_name}] No financial statement data found via get_deep_dive_data for {ticker}.")
                return 

            statements = data.get("financial_statements", {})
            all_statements_data = []

            for statement_type in ['income', 'balance', 'cashflow']:
                df = statements.get(statement_type)
                if df is None or df.empty:
                    continue
                
                if df.index.name is None:
                    df.index.name = 'metric_name'
                
                try:
                    date_columns = [col for col in df.columns if isinstance(col, (datetime, pd.Timestamp)) or pd.api.types.is_datetime64_any_dtype(col)]
                    non_date_columns = [col for col in df.columns if col not in date_columns]
                    
                    new_date_cols = pd.to_datetime(date_columns, errors='coerce').dropna()
                    
                    df.columns = non_date_columns + new_date_cols.tolist()
                except Exception as date_err:
                     logging.warning(f"[{job_name}] Could not handle columns to datetime for {ticker}, {statement_type}: {date_err}. Skipping this statement.")
                     continue 

                df_to_melt = df.select_dtypes(include=[np.number, 'datetime', 'object']).reset_index()
                
                df_long = df_to_melt.melt(id_vars='metric_name', var_name='report_date', value_name='metric_value')
                
                df_long['ticker'] = ticker
                df_long['statement_type'] = statement_type
                df_long['report_date'] = pd.to_datetime(df_long['report_date']).dt.date
                df_long.dropna(subset=['metric_value'], inplace=True) 
                
                df_long['metric_value'] = pd.to_numeric(df_long['metric_value'], errors='coerce').astype(float) 
                
                df_long['metric_name'] = df_long['metric_name'].astype(str).str.strip()


                all_statements_data.extend(df_long.to_dict('records'))

            if all_statements_data:
                with server.app_context():
                    chunk_size = 1000 
                    constraint_name = '_ticker_date_statement_metric_uc'
                    
                    for i in range(0, len(all_statements_data), chunk_size):
                        chunk = all_statements_data[i:i + chunk_size]
                        
                        stmt = sql_insert(FactFinancialStatements).values(chunk)
                        set_ = {'metric_value': stmt.excluded.metric_value}
                        
                        try:
                            if db_dialect == 'postgresql':
                                on_conflict_stmt = stmt.on_conflict_do_update(constraint=constraint_name, set_=set_)
                            elif db_dialect == 'sqlite':
                                on_conflict_stmt = stmt.on_conflict_do_update(index_elements=['ticker', 'report_date', 'statement_type', 'metric_name'], set_=set_)
                            else:
                                raise Exception("Unsupported DB dialect for UPSERT.")
                            
                            db.session.execute(on_conflict_stmt)
                            db.session.commit()
                        except Exception as chunk_e:
                            logging.error(f"[{job_name}] Error inserting financial chunk for {ticker}: {chunk_e}")
                            db.session.rollback()
                            
        except Exception as inner_e:
            logging.error(f"[{job_name}] Error during processing/DB UPSERT for {ticker}: {inner_e}", exc_info=True)
            with server.app_context():
                db.session.rollback()
            raise inner_e 

    process_tickers_with_retry(job_name, tickers_list, process_single_ticker_financials, initial_delay=0.8, max_retries=2) 


# --- Job 4: update_news_sentiment (ไม่เปลี่ยนแปลง) ---
def update_news_sentiment():
    """
    ETL Job 4: ดึงข้อมูลข่าวและ Sentiment (จาก get_news_and_sentiment)
    และเก็บลง FactNewsSentiment
    """
    if sql_insert is None:
        logging.error("ETL Job: [update_news_sentiment] cannot run because insert statement is not available.")
        return

    job_name = "Job 4: Update News/Sentiment"
    companies_list = []
    with server.app_context():
        try:
            companies_list = db.session.query(DimCompany.ticker, DimCompany.company_name).filter(DimCompany.company_name != None).all()
        except Exception as e:
            logging.error(f"[{job_name}] Failed to query companies from DimCompany: {e}", exc_info=True)
            return

    def process_single_ticker_news(company_data_tuple):
        ticker, company_name = company_data_tuple 

        if not company_name:
            logging.warning(f"[{job_name}] Skipping news for {ticker}, no company name found.")
            return True 

        try:
            data = get_news_and_sentiment(company_name)
            if 'error' in data or 'articles' not in data or not data['articles']:
                logging.warning(f"[{job_name}] No news articles found or error fetching for {company_name} ({ticker}). Error: {data.get('error')}")
                return True 

            articles_data = []
            for article in data['articles']:
                try:
                    if not all(k in article and article[k] is not None for k in ['publishedAt', 'title', 'url']):
                        logging.warning(f"[{job_name}] Skipping article for {ticker} due to missing data: {article.get('title')}")
                        continue

                    published_dt_str = article['publishedAt']
                    if isinstance(published_dt_str, str):
                        if published_dt_str.endswith('Z'):
                             published_dt = datetime.fromisoformat(published_dt_str.replace("Z", "+00:00"))
                        else:
                             try:
                                 published_dt = datetime.fromisoformat(published_dt_str)
                             except ValueError:
                                 logging.warning(f"[{job_name}] Could not parse datetime '{published_dt_str}' for article URL {article.get('url')}. Skipping article.")
                                 continue
                    elif isinstance(published_dt_str, datetime): 
                        published_dt = published_dt_str
                    else:
                        logging.warning(f"[{job_name}] Invalid publishedAt format for article URL {article.get('url')}. Skipping article.")
                        continue


                    articles_data.append({
                        'ticker': ticker,
                        'published_at': published_dt,
                        'title': article.get('title'),
                        'article_url': article.get('url'), 
                        'source_name': article.get('source', {}).get('name'),
                        'description': article.get('description'), 
                        'sentiment_label': article.get('sentiment'),
                        'sentiment_score': article.get('sentiment_score')
                    })
                except Exception as parse_e:
                    logging.warning(f"[{job_name}] Skipping article for {ticker} due to processing error: {parse_e}")
                    continue 

            if articles_data:
                with server.app_context():
                    chunk_size = 500
                    
                    for i in range(0, len(articles_data), chunk_size):
                        chunk = articles_data[i:i + chunk_size]
                        
                        stmt = sql_insert(FactNewsSentiment).values(chunk)
                        
                        try:
                            if db_dialect == 'postgresql':
                                on_conflict_stmt = stmt.on_conflict_do_nothing(constraint='fact_news_sentiment_article_url_key')
                            elif db_dialect == 'sqlite':
                                on_conflict_stmt = stmt.on_conflict_do_nothing(index_elements=['article_url'])
                            else:
                                raise Exception("Unsupported DB dialect for UPSERT with DO NOTHING.")

                            db.session.execute(on_conflict_stmt)
                        except Exception as chunk_e:
                            logging.error(f"[{job_name}] Error inserting news chunk for {ticker}: {chunk_e}")
                            db.session.rollback()
                        
                    db.session.commit() 
            return True 

        except Exception as inner_e:
            logging.error(f"[{job_name}] Error during processing/DB UPSERT for {ticker} ({company_name}): {inner_e}", exc_info=True)
            with server.app_context():
                db.session.rollback()
            raise inner_e 

    process_tickers_with_retry(job_name, companies_list, process_single_ticker_news, initial_delay=1.0, retry_delay=90, max_retries=2)
# app/etl.py (MODIFIED: Job 1/3 = Top 1000, Job 2 = Top 50, with 5-Year Purge)

import logging
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import time
from typing import Dict, List, Optional
from sqlalchemy import func # Import SQL functions for MAX()

# Import สิ่งที่จำเป็นจากโปรเจกต์ของเรา
from . import db, server
from .models import DimCompany, FactCompanySummary, FactDailyPrices, FactFinancialStatements, FactNewsSentiment

from .data_handler import get_news_and_sentiment, get_competitor_data 
from .constants import ALL_TICKERS_SORTED_BY_MC, INDEX_TICKER_TO_NAME, HISTORICAL_START_DATE

# Import สำหรับ UPSERT
try:
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    sql_insert = pg_insert
    db_dialect = 'postgresql'
    logging.info("Using PostgreSQL dialect for UPSERT.")
except ImportError:
    try:
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert
        sql_insert = sqlite_insert
        db_dialect = 'sqlite'
        logging.info("Using SQLite dialect for UPSERT.")
    except ImportError:
        sql_insert = None
        db_dialect = None
        logging.error("Could not import insert statement for PostgreSQL or SQLite.")


# ตั้งค่า logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- [HELPER] FIX: Convert numpy types to Python native types for psycopg2 ---
def _clean_numpy_types(data_dict: Dict) -> Dict:
    """Converts numpy types (int64, float64, np.number) to native Python types (float, None)."""
    cleaned_item = {}
    for key, value in data_dict.items():
        if pd.isna(value) or value is None:
            cleaned_item[key] = None
        # Check for any NumPy number type (np.int64, np.float64, np.number)
        elif isinstance(value, (np.int64, np.float64, np.number)):
            # Convert to standard Python float for consistency with DB schema (FactCompanySummary mostly uses Float)
            cleaned_item[key] = float(value)
        else:
            cleaned_item[key] = value
    return cleaned_item
# ---------------------------------------------------------------------------------


# --- Helper Function for Retry and Progress (MODIFIED: Removed time.sleep) ---
def process_tickers_with_retry(job_name, items_list, process_func, initial_delay=0.5, retry_delay=60, max_retries=3):
    """
    วน Loop ผ่านรายการ (tickers หรือ tuples) พร้อม Retry, Delay และ Progress Tracking.
    *** MODIFIED: Removed mandatory time.sleep here. Delay must be inside process_func. ***
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
                # time.sleep(initial_delay) # <--- REMOVED: Delay is now inside process_func if API was called

            except Exception as e:
                retries += 1
                error_msg = str(e)
                # Check for common rate limit indicators (adjust based on actual errors observed)
                is_rate_limit = "Too Many Requests" in error_msg or "429" in error_msg or "400" in error_msg
                log_level = logging.WARNING if is_rate_limit else logging.ERROR

                logging.log(log_level, f"[{job_name}] Error processing {identifier} (Attempt {retries}/{max_retries}): {error_msg}")

                if retries < max_retries:
                    current_retry_delay = retry_delay * (2 ** (retries - 1)) # Exponential backoff
                    logging.info(f"[{job_name}] Retrying {identifier} after {current_retry_delay} seconds...")
                    time.sleep(current_retry_delay)
                else:
                    logging.error(f"[{job_name}] Max retries reached for {identifier}. Skipping.")
                    skipped_items.append(identifier)
                    time.sleep(initial_delay) # Keep delay on FINAL skip/fail

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


# --- Job 1: Update Company Summaries (MODIFIED for TOP 1000 Limit & 5-Year Purge) ---
def update_company_summaries(tickers_list_override: Optional[List[str]] = None):
    if sql_insert is None:
        logging.error("ETL Job: [update_company_summaries] cannot run because insert statement is not available.")
        return

    job_name = "Job 1: Update Summaries (Top 1000, 5-Year Purge)" # <<< MODIFIED Job Name
    
    # --- [MODIFIED: TOP 1000 Limit] ---
    if tickers_list_override is not None:
        tickers_to_process = tickers_list_override
        logging.info(f"ETL Job: [{job_name}] Using OVERRIDE list with {len(tickers_to_process)} tickers.")
    else:
        tickers_to_process = ALL_TICKERS_SORTED_BY_MC[:1000] # <<< Set to 1000
        logging.info(f"ETL Job: [{job_name}] Defaulting to Top 1000 Market Cap tickers ({len(tickers_to_process)} total).")
    # --- [END MODIFIED] ---

    today = datetime.utcnow().date()
    
    # --- [NEW: 5-Year Cutoff for Purge] ---
    days_back_5y = 5 * 365
    cutoff_date_5y = today - timedelta(days=days_back_5y)
    # --- [END NEW] ---

    # --- [START FIX: กำหนดวันที่เริ่มต้นถาวร 2010-01-01] ---
    FIXED_FULL_HISTORY_START_DATE = datetime(2010, 1, 1).date()
    logging.info(f"Starting ETL Job: [{job_name}] for fixed start date {FIXED_FULL_HISTORY_START_DATE}, implementing GAP FILLING...")
    # --- [END FIX] ---

    with server.app_context():
        max_date_results = db.session.query(
            FactCompanySummary.ticker,
            func.max(FactCompanySummary.date_updated)
        ).filter(
            FactCompanySummary.ticker.in_(tickers_to_process)
        ).group_by(FactCompanySummary.ticker).all()
        max_dates = {ticker: max_date for ticker, max_date in max_date_results}

    tickers_to_fetch = []
    for ticker in tickers_to_process:
        latest_date_in_db = max_dates.get(ticker)
        if latest_date_in_db == today:
             logging.info(f"[{job_name}] Skipping {ticker}: Already updated today ({latest_date_in_db}).")
             continue
        tickers_to_fetch.append(ticker)

    if not tickers_to_fetch:
         logging.info(f"ETL Job: [{job_name}] All {len(tickers_to_process)} tickers are already up-to-date for today. Nothing to fetch.")
    
    # --- Process tickers if needed ---
    if tickers_to_fetch:
        logging.info(f"ETL Job: Fetching summaries for {len(tickers_to_fetch)}/{len(tickers_to_process)} tickers.")

        def process_single_ticker_summary(ticker):
            try:
                # ใช้ get_competitor_data ที่ import มา
                df_single = get_competitor_data((ticker,))

                if df_single.empty:
                    logging.warning(f"[{job_name}] get_competitor_data returned empty for {ticker}. Skipping DB insert.")
                    time.sleep(0.3) 
                    return

                row = df_single.iloc[0]

                # --- [NEW STEP 1.5: Get last known prob AND cluster ID] ---
                last_prob = None
                last_cluster_id = None 
                try:
                    with server.app_context():
                        last_prob_result = db.session.query(FactCompanySummary.predicted_default_prob) \
                            .filter(FactCompanySummary.ticker == ticker, 
                                    FactCompanySummary.predicted_default_prob.isnot(None)) \
                            .order_by(FactCompanySummary.date_updated.desc()) \
                            .first()
                        if last_prob_result:
                            last_prob = last_prob_result[0]
                        
                        last_cluster_result = db.session.query(FactCompanySummary.peer_cluster_id) \
                            .filter(FactCompanySummary.ticker == ticker, 
                                    FactCompanySummary.peer_cluster_id.isnot(None)) \
                            .order_by(FactCompanySummary.date_updated.desc()) \
                            .first()
                        if last_cluster_result:
                            last_cluster_id = last_cluster_result[0]
                            
                except Exception as e:
                    logging.warning(f"[{job_name}] Could not query previous prob/cluster for {ticker}: {e}")
                # --- [END NEW STEP 1.5] ---

                dim_data = {
                    'ticker': row['Ticker'],
                    'company_name': row.get('company_name'),
                    'logo_url': row.get('logo_url'), 
                    'sector': row.get('sector'),
                    'credit_rating': row.get('credit_rating')
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
                    'long_business_summary': row.get('Long Business Summary'),
                    'fcf_ttm': row.get('freeCashflow'),
                    'revenue_ttm': row.get('totalRevenue'),
                    'predicted_default_prob': last_prob,
                    'peer_cluster_id': last_cluster_id
                }

                with server.app_context():
                    # --- [START] เพิ่มโค้ดเช็คโลโก้ที่มีอยู่ ---
                    try:
                        existing_company_logo = db.session.query(DimCompany.logo_url).filter_by(ticker=row['Ticker']).scalar()
                        if existing_company_logo and 'logo_url' in dim_data and dim_data['logo_url']:
                            logging.info(f"[{job_name}] Skipping logo_url update for {row['Ticker']}: Already exists in DB ('{existing_company_logo[:30]}...').")
                            del dim_data['logo_url'] 
                        elif 'logo_url' not in dim_data or not dim_data.get('logo_url'):
                             if 'logo_url' in dim_data:
                                del dim_data['logo_url']
                    except Exception as query_err:
                         logging.warning(f"[{job_name}] Could not query existing logo for {row['Ticker']}: {query_err}. Will proceed with fetched data.")
                    # --- [END] เพิ่มโค้ดเช็คโลโก้ ---

                    fact_data_clean = _clean_numpy_types(fact_data)

                    # --- UPSERT DimCompany ---
                    if dim_data:
                        stmt_dim = sql_insert(DimCompany).values(dim_data)
                        dim_columns_to_update = ['company_name', 'logo_url', 'sector', 'credit_rating']
                        dim_set_ = {
                            col: getattr(stmt_dim.excluded, col)
                            for col in dim_columns_to_update if col in dim_data
                        }
                        if dim_set_: 
                            if db_dialect == 'postgresql':
                                on_conflict_dim = stmt_dim.on_conflict_do_update(constraint='dim_company_pkey', set_=dim_set_)
                            elif db_dialect == 'sqlite':
                                on_conflict_dim = stmt_dim.on_conflict_do_update(index_elements=['ticker'], set_=dim_set_)
                            db.session.execute(on_conflict_dim)
                        else: 
                            if db_dialect == 'postgresql':
                                 on_conflict_dim_nothing = stmt_dim.on_conflict_do_nothing(constraint='dim_company_pkey')
                            elif db_dialect == 'sqlite':
                                 on_conflict_dim_nothing = stmt_dim.on_conflict_do_nothing(index_elements=['ticker'])
                            db.session.execute(on_conflict_dim_nothing)

                    # UPSERT FactCompanySummary
                    stmt_fact = sql_insert(FactCompanySummary).values(fact_data_clean)
                    all_cols = {c.name for c in FactCompanySummary.__table__.columns 
                                if c.name not in ['id', 'ticker', 'date_updated', 'peer_cluster_id', 'predicted_default_prob']}
                    
                    fact_set_ = {col: getattr(stmt_fact.excluded, col) for col in all_cols}
                    if db_dialect == 'postgresql':
                        on_conflict_fact = stmt_fact.on_conflict_do_update(constraint='_ticker_date_uc', set_=fact_set_)
                    elif db_dialect == 'sqlite':
                        on_conflict_fact = stmt_fact.on_conflict_do_update(index_elements=['ticker', 'date_updated'], set_=fact_set_)
                    db.session.execute(on_conflict_fact)

                    db.session.commit()

                    time.sleep(0.3) 

            except Exception as inner_e:
                logging.error(f"[{job_name}] Error during processing/DB UPSERT for {ticker}: {inner_e}", exc_info=True)
                with server.app_context():
                     db.session.rollback()
                raise inner_e

        process_tickers_with_retry(job_name, tickers_to_fetch, process_single_ticker_summary, initial_delay=0.7, max_retries=3)

    # --- [NEW] Purge data older than 5 years ---
    logging.info(f"[{job_name}] Purging old summary data (before {cutoff_date_5y})...")
    try:
        with server.app_context():
            deleted_count = db.session.query(FactCompanySummary).filter(
                FactCompanySummary.date_updated < cutoff_date_5y
            ).delete(synchronize_session=False)
            db.session.commit()
            logging.info(f"[{job_name}] Purged {deleted_count} old summary records.")
    except Exception as e:
        logging.error(f"[{job_name}] Error during old summary data purge: {e}")
        with server.app_context():
            db.session.rollback()
    # --- [END PURGE] ---


# --- Job 2: Update Daily Prices (MODIFIED for TOP 50 Limit & 5-Year Purge and Start Date) ---
# --- [NEW HELPER FUNCTION FOR OOM FIX - UNCHANGED] ---
def process_single_ticker_prices(data_tuple):
    ticker, start_date, end_date = data_tuple
    if sql_insert is None:
        raise Exception("SQL insert dialect not supported.")

    job_name = "Job 2: Update Daily Prices"

    try:
        prices_df = yf.download([ticker], start=start_date, end=end_date, auto_adjust=True, progress=False)

        if prices_df.empty:
            logging.warning(f"[{job_name}] yf.download returned empty for {ticker} (Start: {start_date}). Skipping DB insert.")
            return

        if isinstance(prices_df.columns, pd.MultiIndex):
            prices_df.columns = prices_df.columns.get_level_values(0)
            prices_df = prices_df.rename_axis(['Date']).reset_index()
            prices_df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        else:
             prices_df = prices_df.rename_axis(['Date']).reset_index()
             prices_df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)

        prices_df['ticker'] = ticker
        prices_df['date'] = pd.to_datetime(prices_df['date']).dt.date
        prices_df['volume'] = pd.to_numeric(prices_df['volume'], errors='coerce').fillna(0).apply(lambda x: int(x) if pd.notna(x) else None)
        prices_df['open'] = pd.to_numeric(prices_df['open'], errors='coerce').apply(lambda x: float(x) if pd.notna(x) else None)
        prices_df['high'] = pd.to_numeric(prices_df['high'], errors='coerce').apply(lambda x: float(x) if pd.notna(x) else None)
        prices_df['low'] = pd.to_numeric(prices_df['low'], errors='coerce').apply(lambda x: float(x) if pd.notna(x) else None)
        prices_df['close'] = pd.to_numeric(prices_df['close'], errors='coerce').apply(lambda x: float(x) if pd.notna(x) else None)

        required_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        prices_df = prices_df[required_cols]
        prices_df.dropna(subset=['close', 'ticker', 'date'], inplace=True)

        data_to_upsert = prices_df.to_dict(orient='records')
        num_rows = len(data_to_upsert)

        if data_to_upsert:
            with server.app_context():
                chunk_size = 1000
                constraint_name = '_ticker_price_date_uc'

                for i in range(0, num_rows, chunk_size):
                    chunk = data_to_upsert[i:i + chunk_size]
                    stmt_chunk = sql_insert(FactDailyPrices).values(chunk)
                    price_set_chunk_ = {'open': stmt_chunk.excluded.open, 'high': stmt_chunk.excluded.high, 'low': stmt_chunk.excluded.low, 'close': stmt_chunk.excluded.close, 'volume': stmt_chunk.excluded.volume}
                    if db_dialect == 'postgresql':
                        on_conflict_chunk = stmt_chunk.on_conflict_do_update(constraint=constraint_name, set_=price_set_chunk_)
                    elif db_dialect == 'sqlite':
                        on_conflict_chunk = stmt_chunk.on_conflict_do_update(index_elements=['ticker', 'date'], set_=price_set_chunk_)
                    else: raise Exception("Unsupported DB dialect for UPSERT.")
                    db.session.execute(on_conflict_chunk)
                    db.session.commit()
            time.sleep(0.3)

    except Exception as e:
        with server.app_context(): db.session.rollback()
        raise e
# --- [END NEW HELPER FUNCTION FOR OOM FIX - UNCHANGED] ---

def update_daily_prices(tickers_list_override: Optional[List[str]] = None):
    # --- [MODIFIED: 5-Year Window & TOP 50] ---
    days_back_5y = 5 * 365 
    job_name = "Job 2: Update Daily Prices (Top 50, 5-Year Window)" # <<< MODIFIED Job Name
    # --- [END MODIFIED] ---
    
    if sql_insert is None:
        logging.error(f"ETL Job: [update_daily_prices] cannot run because insert statement is not available.")
        return

    today = datetime.utcnow().date()
    # --- [NEW: 5-Year Cutoff] ---
    cutoff_date_5y = today - timedelta(days=days_back_5y)
    # --- [END NEW] ---

    with server.app_context():
        logging.info(f"Starting ETL Job: [{job_name}] for 5 years back (approx {days_back_5y} days), implementing GAP FILLING...")
        
        if tickers_list_override is not None and len(tickers_list_override) > 0:
            tickers_list = tickers_list_override
            logging.info(f"ETL Job: [{job_name}] Using OVERRIDE list with {len(tickers_list)} tickers.")
        else:
            tickers_for_price_update = ALL_TICKERS_SORTED_BY_MC[:50] # <<< Set to 50
            index_tickers = list(INDEX_TICKER_TO_NAME.keys())
            tickers_list = list(set(tickers_for_price_update + index_tickers))
            logging.info(f"ETL Job: [{job_name}] Defaulting to Top 50 Market Cap + Index tickers ({len(tickers_list)} total unique).") # <<< MODIFIED Log

        if not tickers_list:
            logging.warning(f"ETL Job: [{job_name}] No tickers to process. Skipping price update.")
            logging.info(f"ETL Job: [{job_name}]... COMPLETED.")
            return 

        max_date_results = db.session.query(FactDailyPrices.ticker, func.max(FactDailyPrices.date)).filter(FactDailyPrices.ticker.in_(tickers_list)).group_by(FactDailyPrices.ticker).all()
        max_dates = {ticker: max_date for ticker, max_date in max_date_results}

        is_monday = today.weekday() == 0
        required_days_back = 3 if is_monday else 1
        ticker_tuples_to_process = []
        
        # --- [MODIFIED: 5-Year Window] ---
        full_history_start_date_for_job = cutoff_date_5y 
        # --- [END MODIFIED] ---

        for ticker in tickers_list:
            latest_date_in_db = max_dates.get(ticker)
            if latest_date_in_db:
                if (today - latest_date_in_db).days <= required_days_back:
                     logging.info(f"[{job_name}] Skipping {ticker}: Data is very recent (Latest: {latest_date_in_db}, Buffer: {required_days_back} days).")
                     continue
                new_start_date = latest_date_in_db + timedelta(days=1)
            else:
                # --- [FIX] ใช้ 5-year cutoff สำหรับ Ticker ที่ไม่มีข้อมูลเลย ---
                new_start_date = full_history_start_date_for_job
                logging.info(f"[{job_name}] No data for {ticker}. Fetching from 5-year start date: {new_start_date}")

            # --- [NEW] Ensure we don't fetch *before* the 5-year cutoff ---
            if new_start_date < cutoff_date_5y:
                logging.warning(f"[{job_name}] {ticker} has latest date {latest_date_in_db}, but start date {new_start_date} is before 5y cutoff. Forcing start date to {cutoff_date_5y}.")
                new_start_date = cutoff_date_5y
            # --- [END NEW] ---

            if new_start_date >= today: 
                logging.info(f"[{job_name}] Skipping {ticker}: Already up-to-date or start date is in the future (Start: {new_start_date}).")
                continue
            fetch_end_date = today
            ticker_tuples_to_process.append((ticker, new_start_date, fetch_end_date))

        if not ticker_tuples_to_process:
             logging.info(f"ETL Job: [{job_name}] All {len(tickers_list)} tickers are already up-to-date. Nothing to process.")
        else:
            logging.info(f"ETL Job: [{job_name}] Processing {len(ticker_tuples_to_process)}/{len(tickers_list)} tickers. Fetching missing data.")
            process_tickers_with_retry(job_name, ticker_tuples_to_process, process_single_ticker_prices, initial_delay=0.3, retry_delay=60, max_retries=3)
        
    # --- [NEW] Purge data older than 5 years ---
    logging.info(f"[{job_name}] Purging old price data (before {cutoff_date_5y})...")
    try:
        with server.app_context():
            deleted_count = db.session.query(FactDailyPrices).filter(
                FactDailyPrices.date < cutoff_date_5y
            ).delete(synchronize_session=False)
            db.session.commit()
            logging.info(f"[{job_name}] Purged {deleted_count} old price records.")
    except Exception as e:
        logging.error(f"[{job_name}] Error during old price data purge: {e}")
        with server.app_context():
            db.session.rollback()
    # --- [END PURGE] ---

    logging.info(f"ETL Job: [{job_name}]... COMPLETED.")


# --- Job 3: update_financial_statements (MODIFIED for TOP 1000 Limit & 5-Year Purge and Stability) ---
def update_financial_statements(tickers_list_override: Optional[List[str]] = None):
    if sql_insert is None:
        logging.error("ETL Job: [update_financial_statements] cannot run because insert statement is not available.")
        return

    # --- [MODIFIED: 5-Year Window & TOP 1000] ---
    job_name = "Job 3: Update Financials (Top 1000, 5-Year Window)" # <<< MODIFIED Job Name
    today = datetime.utcnow().date()
    days_back_5y = 5 * 365
    cutoff_date_5y = today - timedelta(days=days_back_5y)
    # --- [END MODIFIED] ---

    with server.app_context():
        if tickers_list_override is not None and len(tickers_list_override) > 0:
            tickers_to_process = tickers_list_override
            logging.info(f"ETL Job: [{job_name}] Using OVERRIDE list with {len(tickers_to_process)} tickers.")
        else:
            tickers_to_process = ALL_TICKERS_SORTED_BY_MC[:1000] # <<< Set to 1000
            logging.info(f"ETL Job: [{job_name}] Defaulting to Top 1000 Market Cap tickers ({len(tickers_to_process)} total).") # <<< MODIFIED Log

        if not tickers_to_process:
            logging.warning(f"ETL Job: [{job_name}] No tickers to process. Skipping job.")
            logging.info(f"ETL Job: [{job_name}]... COMPLETED with skip logic.")
            return

        # --- [MODIFIED: Query latest financial report date, NOT summary date] ---
        logging.info(f"ETL Job: [{job_name}] Querying DB for latest financial report dates (gap-fill check)...")
        max_date_results = db.session.query(
            FactFinancialStatements.ticker,
            func.max(FactFinancialStatements.report_date) 
        ).filter(
            FactFinancialStatements.ticker.in_(tickers_to_process)
        ).group_by(FactFinancialStatements.ticker).all()
        
        max_dates_map = {ticker: max_date for ticker, max_date in max_date_results}
        # --- [END MODIFIED] ---

    tickers_to_fetch = []
    # --- [NEW: Logic to decide which tickers to fetch based on gap-fill] ---
    QUARTERLY_BUFFER_DAYS = 60 
    
    for ticker in tickers_to_process:
        latest_report_date = max_dates_map.get(ticker)
        
        if latest_report_date:
            days_diff = (today - latest_report_date).days
            if days_diff <= QUARTERLY_BUFFER_DAYS:
                logging.info(f"[{job_name}] Skipping {ticker}: Latest financial report in DB is recent (Date: {latest_report_date}, {days_diff} days ago).")
                continue
        
        # Add to fetch list if no data OR if data is older than the buffer
        tickers_to_fetch.append(ticker)
    # --- [END NEW] ---

    if not tickers_to_fetch:
         logging.info(f"ETL Job: [{job_name}] All {len(tickers_to_process)} tickers are already up-to-date (within {QUARTERLY_BUFFER_DAYS}-day buffer). Nothing to fetch.")
    else:
        logging.info(f"ETL Job: [{job_name}] Fetching financial statements for {len(tickers_to_fetch)}/{len(tickers_to_process)} tickers (gap-fill applied).")
        
        # --- [MODIFIED: Pass the max_dates_map AND 5Y CUTOFF to the retry helper] ---
        ticker_data_tuples = []
        for ticker in tickers_to_fetch:
            ticker_data_tuples.append(
                (ticker, max_dates_map.get(ticker), cutoff_date_5y) # Pass ticker, latest date, AND 5Y cutoff
            )
        
        process_tickers_with_retry(job_name, ticker_data_tuples, process_single_ticker_financials, initial_delay=0.8, max_retries=2)
        # --- [END MODIFIED] ---
    
    # --- [NEW] Purge data older than 5 years ---
    logging.info(f"[{job_name}] Purging old financial data (before {cutoff_date_5y})...")
    try:
        with server.app_context():
            deleted_count = db.session.query(FactFinancialStatements).filter(
                FactFinancialStatements.report_date < cutoff_date_5y
            ).delete(synchronize_session=False)
            db.session.commit()
            logging.info(f"[{job_name}] Purged {deleted_count} old financial records.")
    except Exception as e:
        logging.error(f"[{job_name}] Error during old financial data purge: {e}")
        with server.app_context():
            db.session.rollback()
    # --- [END PURGE] ---

    logging.info(f"ETL Job: [{job_name}]... COMPLETED with skip logic.")


# --- [MODIFIED: Function signature accepts the tuple, and implements cleaning] ---
def process_single_ticker_financials(ticker_data_tuple):
    job_name = "Job 3: Update Financials (Top 1000, 5-Year Window)" # <<< MODIFIED Job Name
    
    # --- [MODIFIED: Unpack 5Y Cutoff] ---
    ticker, latest_date_in_db, cutoff_date_5y = ticker_data_tuple # Unpack the tuple
    
    if sql_insert is None:
        logging.error(f"[{job_name}] cannot run because insert statement is not available.")
        return

    try:
        if latest_date_in_db:
            logging.info(f"[{job_name}] Gap Filling: Latest financial report date in DB for {ticker} is {latest_date_in_db}. Will filter API data.")
        else:
            logging.info(f"[{job_name}] Gap Filling: No financial data in DB for {ticker}. Will fetch all available data (since {cutoff_date_5y}).")

        logging.info(f"[{job_name}] Fetching all available financial statements for {ticker} using yfinance...")
        tkr = yf.Ticker(ticker)
        statements_data = {
            'income_q': tkr.quarterly_financials, 
            'balance_q': tkr.quarterly_balance_sheet, 
            'cashflow_q': tkr.quarterly_cashflow,
            'income_a': tkr.financials,            
            'balance_a': tkr.balance_sheet,        
            'cashflow_a': tkr.cashflow             
        }
        if not any(df is not None and not df.empty for df in statements_data.values()): 
            logging.warning(f"[{job_name}] No financial statement data found via yfinance for {ticker}.")
            time.sleep(0.8)
            return

        all_statements_data = [] 
        for statement_key, df in statements_data.items(): 
            if df is None or df.empty: continue
            if df.index.name is None: df.index.name = 'metric_name'
            
            df_to_process = df.copy() 

            try:
                date_columns = pd.to_datetime(df_to_process.columns, errors='coerce').dropna()
                non_date_column_names = [col for col in df_to_process.columns if col not in date_columns]
                
                columns_to_keep = non_date_column_names[:] 

                # --- [MODIFIED: 5-Year Cutoff Logic] ---
                valid_date_columns = [col for col in date_columns if col.date() >= cutoff_date_5y]

                if latest_date_in_db:
                    new_date_columns = [col for col in valid_date_columns if col.date() > latest_date_in_db]
                    
                    if not new_date_columns:
                        logging.info(f"[{job_name}] Gap Filling: No new report dates found *after* {latest_date_in_db} for {ticker} ({statement_key}).")
                        continue
                        
                    logging.info(f"[{job_name}] Gap Filling: Found {len(new_date_columns)} new report(s) for {ticker} ({statement_key}). Processing them.")
                    columns_to_keep.extend(new_date_columns)
                else:
                    logging.info(f"[{job_name}] Gap Filling: No DB data. Fetching all reports since 5-year cutoff ({cutoff_date_5y}) for {ticker} ({statement_key}).")
                    columns_to_keep.extend(valid_date_columns) 
                
                df_to_process = df_to_process[columns_to_keep]
                # --- [END MODIFIED] ---

            except Exception as date_err:
                logging.warning(f"[{job_name}] Could not handle columns to datetime for {ticker}, {statement_key}: {date_err}. Skipping this statement.")
                continue
            
            if df_to_process.empty or all(col in non_date_column_names for col in df_to_process.columns):
                 logging.info(f"[{job_name}] No date columns left to process for {ticker} ({statement_type}) after filtering. Skipping.")
                 continue

            df_to_melt = df_to_process.reset_index()
            
            # --- [MODIFIED STEP 3: Melt and re-parse dates] ---
            id_vars_to_use = [col for col in non_date_column_names + ['metric_name'] if col in df_to_melt.columns]
            df_long = df_to_melt.melt(id_vars=id_vars_to_use, var_name='report_date', value_name='metric_value')
            
            df_long['report_date'] = pd.to_datetime(df_long['report_date']).dt.date
            # --- [END MODIFIED STEP 3] ---
            
            df_long['ticker'] = ticker
            df_long['statement_type'] = statement_key 
            
            df_long.dropna(subset=['metric_value'], inplace=True)
            df_long['metric_value'] = pd.to_numeric(df_long['metric_value'], errors='coerce').astype(float)
            df_long.dropna(subset=['metric_value'], inplace=True)
            df_long['metric_name'] = df_long['metric_name'].astype(str).str.strip()
            
            df_long = df_long[['ticker', 'report_date', 'statement_type', 'metric_name', 'metric_value']]
            
            all_statements_data.extend(df_long.to_dict('records'))

        if all_statements_data:
            with server.app_context():
                chunk_size = 1000
                constraint_name = '_ticker_date_statement_metric_uc'
                for i in range(0, len(all_statements_data), chunk_size):
                    chunk = all_statements_data[i:i + chunk_size]
                    
                    # --- [FIX: Clean Numpy Types for stability] ---
                    chunk_clean = [_clean_numpy_types(item) for item in chunk] 
                    # --- [END FIX] ---
                    
                    stmt = sql_insert(FactFinancialStatements).values(chunk_clean)
                    set_ = {'metric_value': stmt.excluded.metric_value}
                    try:
                        if db_dialect == 'postgresql': on_conflict_stmt = stmt.on_conflict_do_update(constraint=constraint_name, set_=set_)
                        elif db_dialect == 'sqlite': on_conflict_stmt = stmt.on_conflict_do_update(index_elements=['ticker', 'report_date', 'statement_type', 'metric_name'], set_=set_)
                        else: raise Exception("Unsupported DB dialect for UPSERT.")
                        db.session.execute(on_conflict_stmt)
                        db.session.commit()
                    except Exception as chunk_e:
                        logging.error(f"[{job_name}] Error inserting financial chunk for {ticker}: {chunk_e}")
                        db.session.rollback(); continue
        time.sleep(0.8)
        return True

    except Exception as inner_e:
        logging.error(f"[{job_name}] Error during processing/DB UPSERT for {ticker}: {inner_e}", exc_info=True)
        try:
            with server.app_context(): db.session.rollback()
        except Exception as rb_err: logging.error(f"[{job_name}] Error during rollback for {ticker}: {rb_err}")
        raise inner_e


# --- Job 4: update_news_sentiment (MODIFIED for 30-Day Purge) ---
def update_news_sentiment():
    if sql_insert is None:
        logging.error("ETL Job: [update_news_sentiment] cannot run because insert statement is not available.")
        return

    job_name = "Job 4: Update News/Sentiment (Top 20, 30-Day Purge)"
    tickers_for_etl = ALL_TICKERS_SORTED_BY_MC[:20] 
    companies_list = []
    
    # --- [NEW: 30-Day Cutoff for News Purge] ---
    today = datetime.utcnow().date()
    cutoff_date_news = datetime.utcnow() - timedelta(days=30) 
    # --- [END NEW] ---

    with server.app_context():
        try:
            companies_list = db.session.query(DimCompany.ticker, DimCompany.company_name).filter(DimCompany.ticker.in_(tickers_for_etl), DimCompany.company_name != None).all()
            logging.info(f"[{job_name}] Found {len(companies_list)}/{len(tickers_for_etl)} top tickers to process news for.")
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
                        if published_dt_str.endswith('Z'): published_dt = datetime.fromisoformat(published_dt_str.replace("Z", "+00:00"))
                        else:
                             try: published_dt = datetime.fromisoformat(published_dt_str)
                             except ValueError: logging.warning(f"[{job_name}] Could not parse datetime '{published_dt_str}' for article URL {article.get('url')}. Skipping article."); continue
                    elif isinstance(published_dt_str, datetime): published_dt = published_dt_str
                    else: logging.warning(f"[{job_name}] Invalid publishedAt format for article URL {article.get('url')}. Skipping article."); continue

                    articles_data.append({'ticker': ticker, 'published_at': published_dt, 'title': article.get('title'), 'article_url': article.get('url'), 'source_name': article.get('source', {}).get('name'), 'description': article.get('description'), 'sentiment_label': article.get('sentiment'), 'sentiment_score': article.get('sentiment_score')})
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
                            if db_dialect == 'postgresql': on_conflict_stmt = stmt.on_conflict_do_nothing(constraint='fact_news_sentiment_article_url_key')
                            elif db_dialect == 'sqlite': on_conflict_stmt = stmt.on_conflict_do_nothing(index_elements=['article_url'])
                            else: raise Exception("Unsupported DB dialect for UPSERT with DO NOTHING.")
                            db.session.execute(on_conflict_stmt)
                        except Exception as chunk_e:
                            logging.error(f"[{job_name}] Error inserting news chunk for {ticker}: {chunk_e}")
                            db.session.rollback()
                    db.session.commit()
            return True

        except Exception as inner_e:
            logging.error(f"[{job_name}] Error during processing/DB UPSERT for {ticker} ({company_name}): {inner_e}", exc_info=True)
            with server.app_context(): db.session.rollback()
            raise inner_e

    process_tickers_with_retry(job_name, companies_list, process_single_ticker_news, initial_delay=1.0, retry_delay=90, max_retries=2)
    
    # --- [NEW] Purge data older than 30 days for news ---
    logging.info(f"[{job_name}] Purging old news data (before {cutoff_date_news})...")
    try:
        with server.app_context():
            deleted_count = db.session.query(FactNewsSentiment).filter(
                FactNewsSentiment.published_at < cutoff_date_news
            ).delete(synchronize_session=False)
            db.session.commit()
            logging.info(f"[{job_name}] Purged {deleted_count} old news records.")
    except Exception as e:
        logging.error(f"[{job_name}] Error during old news data purge: {e}")
        with server.app_context():
            db.session.rollback()
    # --- [END PURGE] ---
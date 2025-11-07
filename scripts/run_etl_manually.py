# scripts/run_etl_manually.py (Corrected)
# run_etl_manually.py (Modified for specific Top N limits and Job Selection)
import sys
import os
import time
import logging
import argparse 
# หาที่อยู่ของโฟลเดอร์ปัจจุบัน (scripts/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# หาที่อยู่ของโฟลเดอร์แม่ (project_dash/)
project_root = os.path.dirname(script_dir)
# เพิ่มโฟลเดอร์แม่เข้าไปใน sys.path
sys.path.append(project_root)
from index import server
from app.etl import update_company_summaries, update_daily_prices, update_financial_statements, update_news_sentiment

# **[สำคัญ] ใช้ชื่อ Constant ที่มีอยู่ใน app/constants.py (อาจต้องเปลี่ยนเป็น ALL_TICKERS_SORTED_BY_MC ชั่วคราว)**
from app.constants import ALL_TICKERS_SORTED_BY_GROWTH, INDEX_TICKER_TO_NAME, HISTORICAL_START_DATE 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # --- (1) Argument Parser ---
    parser = argparse.ArgumentParser(description="Run ETL jobs selectively.")
    parser.add_argument('--jobs', nargs='+', type=int, help="List of job numbers to run (e.g., 1 2 3). Runs all if not specified.")
    args = parser.parse_args()
    jobs_to_run = args.jobs
    
    if jobs_to_run:
        logging.info(f"Attempting to run specified ETL jobs: {jobs_to_run}")
    else:
        logging.info("Attempting to run ALL ETL jobs manually...")
    
    start_time_total = time.time()

    # --- [NEW CONSTANT] FIXED CORE TICKERS (NVDA, AAPL, MSFT, GOOGL, META) ---
    FIXED_CORE_TICKERS = ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'META']
    # --- [END NEW CONSTANT] ---

    # --- (2) กำหนดขีดจำกัด (Job Limits) ---
    TOP_100_BASE = ALL_TICKERS_SORTED_BY_GROWTH[:100] # Job 1 & 3
    TOP_50_BASE = ALL_TICKERS_SORTED_BY_GROWTH[:50]   # Job 2
    INDEX_TICKERS = list(INDEX_TICKER_TO_NAME.keys())
    
    # List สำหรับ Job 1 (Top 100 + Fixed Core)
    TICKERS_FOR_JOB_1 = list(set(TOP_100_BASE + FIXED_CORE_TICKERS))
    logging.info(f"Defined list for Job 1: Top 100 Growth + Fixed Core = {len(TICKERS_FOR_JOB_1)} unique tickers.")
    
    # List สำหรับ Job 2 (Top 50 + Indices + Fixed Core)
    TICKERS_FOR_JOB_2 = list(set(TOP_50_BASE + INDEX_TICKERS + FIXED_CORE_TICKERS))
    logging.info(f"Defined list for Job 2: Top 50 Growth + Indices + Fixed Core = {len(TICKERS_FOR_JOB_2)} unique tickers.")
    
    # List สำหรับ Job 3 (Top 100 + Indices + Fixed Core)
    TICKERS_FOR_JOB_3 = list(set(TOP_100_BASE + INDEX_TICKERS + FIXED_CORE_TICKERS))
    logging.info(f"Defined list for Job 3: Top 100 Growth + Indices + Fixed Core = {len(TICKERS_FOR_JOB_3)} unique tickers.")
    
    with server.app_context():

        # --- Job 1: Update Company Summaries (TOP 100 + Fixed Core) ---
        if not jobs_to_run or 1 in jobs_to_run: 
            logging.info(f"--- Starting Manual Run: Job 1 (update_company_summaries for {len(TICKERS_FOR_JOB_1)} tickers) ---")
            start_time_job1 = time.time()
            try:
                # ใช้ TICKERS_FOR_JOB_1 สำหรับ Job 1
                update_company_summaries(TICKERS_FOR_JOB_1) 
                elapsed_job1 = time.time() - start_time_job1
                logging.info(f"--- Finished Manual Run: Job 1 (update_company_summaries) in {elapsed_job1:.2f} seconds ---")
            except Exception as e:
                elapsed_job1 = time.time() - start_time_job1
                logging.error(f"--- Manual Run FAILED: Job 1 (update_company_summaries) after {elapsed_job1:.2f} seconds: {e} ---", exc_info=True)

        # --- Run Job 2 (Price Update for TOP 50 + Indices + Fixed Core) ---
        if not jobs_to_run or 2 in jobs_to_run:
            logging.info(f"--- Starting Manual Run: Job 2 (update_daily_prices for {len(TICKERS_FOR_JOB_2)} tickers) ---")
            start_time_job2 = time.time()
            try:
                # ใช้ TICKERS_FOR_JOB_2 สำหรับ Job 2
                update_daily_prices(tickers_list_override=TICKERS_FOR_JOB_2) 
                elapsed_job2 = time.time() - start_time_job2
                logging.info(f"--- Finished Manual Run: Job 2 (update_daily_prices) in {elapsed_job2:.2f} seconds ---")
            except Exception as e:
                elapsed_job2 = time.time() - start_time_job2
                logging.error(f"--- Manual Run FAILED: Job 2 (update_daily_prices) after {elapsed_job2:.2f} seconds: {e} ---", exc_info=True)

        # --- Run Job 3 (Financial Statements for TOP 100 + Indices + Fixed Core) ---
        if not jobs_to_run or 3 in jobs_to_run: 
            logging.info(f"--- Starting Manual Run: Job 3 (update_financial_statements for {len(TICKERS_FOR_JOB_3)} tickers) ---")
            start_time_job3 = time.time()
            try:
                # ใช้ TICKERS_FOR_JOB_3 สำหรับ Job 3
                update_financial_statements(tickers_list_override=TICKERS_FOR_JOB_3) 
                elapsed_job3 = time.time() - start_time_job3
                logging.info(f"--- Finished Manual Run: Job 3 (update_financial_statements) in {elapsed_job3:.2f} seconds ---")
            except Exception as e:
                elapsed_job3 = time.time() - start_time_job3
                logging.error(f"--- Manual Run FAILED: Job 3 (update_financial_statements) after {elapsed_job3:.2f} seconds: {e} ---", exc_info=True)

        # --- Run Job 4 (News Sentiment for TOP 10 + Fixed Core) ---
        if not jobs_to_run or 4 in jobs_to_run: 
            logging.info("--- Starting Manual Run: Job 4 (update_news_sentiment for TOP 10 + Fixed Core) ---")
            start_time_job4 = time.time()
            try:
                # Job 4 จะใช้ FIXED_CORE_TICKERS ภายใน app/etl.py แล้ว (ไม่จำเป็นต้อง Override)
                update_news_sentiment()
                elapsed_job4 = time.time() - start_time_job4
                logging.info(f"--- Finished Manual Run: Job 4 (update_news_sentiment) in {elapsed_job4:.2f} seconds ---")
            except Exception as e:
                elapsed_job4 = time.time() - start_time_job4
                logging.error(f"--- Manual Run FAILED: Job 4 (update_news_sentiment) after {elapsed_job4:.2f} seconds: {e} ---", exc_info=True)

    elapsed_total = time.time() - start_time_total
    logging.info(f"--- Finished ALL executed Manual ETL jobs in {elapsed_total:.2f} seconds ---")
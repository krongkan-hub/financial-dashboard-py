# run_etl_manually.py (Modified for Top 1000 and Job Selection)
import sys
import os
import time
import logging
import argparse # <<< (1) Import argparse
# หาที่อยู่ของโฟลเดอร์ปัจจุบัน (scripts/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# หาที่อยู่ของโฟลเดอร์แม่ (project_dash/)
project_root = os.path.dirname(script_dir)
# เพิ่มโฟลเดอร์แม่เข้าไปใน sys.path
sys.path.append(project_root)
from index import server
from app.etl import update_company_summaries, update_daily_prices, update_financial_statements, update_news_sentiment
from app.constants import ALL_TICKERS_SORTED_BY_MC, INDEX_TICKER_TO_NAME, HISTORICAL_START_DATE
# --- [END NEW IMPORT] ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # --- (2) เพิ่ม Argument Parser ---
    parser = argparse.ArgumentParser(description="Run ETL jobs selectively.")
    parser.add_argument('--jobs', nargs='+', type=int, help="List of job numbers to run (e.g., 1 2 3). Runs all if not specified.")
    args = parser.parse_args()
    jobs_to_run = args.jobs
    
    if jobs_to_run:
        logging.info(f"Attempting to run specified ETL jobs: {jobs_to_run}")
    else:
        logging.info("Attempting to run ALL ETL jobs manually...")
    # --- [จบส่วนเพิ่ม] ---

    start_time_total = time.time()

    # --- (3) แก้ไขเป็น Top 1000 (ตามไฟล์ที่คุณอัปโหลดมา) ---
    TOP_1500_BASE = ALL_TICKERS_SORTED_BY_MC[:1000] #
    INDEX_TICKERS = list(INDEX_TICKER_TO_NAME.keys())
    # Use set to ensure uniqueness when combining
    TICKERS_FOR_JOB_2_3 = list(set(TOP_1500_BASE + INDEX_TICKERS))
    # (4) แก้ไข Log ให้ถูกต้อง
    logging.info(f"Defined list for Job 2 & 3: Top 1000 + Indices = {len(TICKERS_FOR_JOB_2_3)} unique tickers.")
    # --- [END NEW] ---

    with server.app_context():

        # --- Job 1: Update Company Summaries (Runs for ALL tickers) ---
        if not jobs_to_run or 1 in jobs_to_run: # <--- (5) เพิ่มเงื่อนไข
            logging.info("--- Starting Manual Run: Job 1 (update_company_summaries for ALL tickers) ---")
            start_time_job1 = time.time()
            try:
                # ใช้ ALL_TICKERS_SORTED_BY_MC เพื่อให้ Job 1 อัปเดตข้อมูล Dim และ FactCompanySummary ทั้งหมด
                update_company_summaries(ALL_TICKERS_SORTED_BY_MC) # <<< ใช้ List เต็ม
                elapsed_job1 = time.time() - start_time_job1
                logging.info(f"--- Finished Manual Run: Job 1 (update_company_summaries) in {elapsed_job1:.2f} seconds ---")
            except Exception as e:
                elapsed_job1 = time.time() - start_time_job1
                logging.error(f"--- Manual Run FAILED: Job 1 (update_company_summaries) after {elapsed_job1:.2f} seconds: {e} ---", exc_info=True)
        # --- [END Job 1] ---

        # --- Run Job 2 (Price Update for Top 1000 + Indices) ---
        if not jobs_to_run or 2 in jobs_to_run: # <--- (5) เพิ่มเงื่อนไข
            # (4) แก้ไข Log ให้ถูกต้อง
            logging.info(f"--- Starting Manual Run: Job 2 (update_daily_prices for Top 1000 + Indices) ---")
            start_time_job2 = time.time()
            try:
                # รัน Job 2 โดยส่ง List ที่จำกัดไว้เข้าไป
                update_daily_prices(tickers_list_override=TICKERS_FOR_JOB_2_3) # <<< ใช้ List ที่จำกัด
                elapsed_job2 = time.time() - start_time_job2
                logging.info(f"--- Finished Manual Run: Job 2 (update_daily_prices) in {elapsed_job2:.2f} seconds ---")
            except Exception as e:
                elapsed_job2 = time.time() - start_time_job2
                logging.error(f"--- Manual Run FAILED: Job 2 (update_daily_prices) after {elapsed_job2:.2f} seconds: {e} ---", exc_info=True)
        # --- [END Job 2] ---

        # --- Run Job 3 (Financial Statements for Top 1000) ---
        if not jobs_to_run or 3 in jobs_to_run: # <--- (5) เพิ่มเงื่อนไข
            # (4) แก้ไข Log ให้ถูกต้อง
            logging.info(f"--- Starting Manual Run: Job 3 (update_financial_statements for Top 1000) ---")
            start_time_job3 = time.time()
            try:
                # รัน Job 3 โดยส่ง List ที่จำกัดไว้เข้าไป (Job 3 ไม่รวม Index Tickers อยู่แล้ว)
                # เราใช้ TICKERS_FOR_JOB_2_3 ไปเลยก็ได้ เพราะใน Job 3 มันจะกรอง Index ออกเองถ้าไม่มีใน DimCompany
                update_financial_statements(tickers_list_override=TICKERS_FOR_JOB_2_3) # <<< ใช้ List ที่จำกัด
                elapsed_job3 = time.time() - start_time_job3
                logging.info(f"--- Finished Manual Run: Job 3 (update_financial_statements) in {elapsed_job3:.2f} seconds ---")
            except Exception as e:
                elapsed_job3 = time.time() - start_time_job3
                logging.error(f"--- Manual Run FAILED: Job 3 (update_financial_statements) after {elapsed_job3:.2f} seconds: {e} ---", exc_info=True)
        # --- [END Job 3] ---

        # --- Run Job 4 (News Sentiment for Top 20) ---
        if not jobs_to_run or 4 in jobs_to_run: # <--- (5) เพิ่มเงื่อนไข
            logging.info("--- Starting Manual Run: Job 4 (update_news_sentiment for Top 20) ---")
            start_time_job4 = time.time()
            try:
                # รัน Job 4 ซึ่งจำกัดแค่ 20 ตัวแรกอยู่แล้ว ไม่ต้องส่ง override
                update_news_sentiment()
                elapsed_job4 = time.time() - start_time_job4
                logging.info(f"--- Finished Manual Run: Job 4 (update_news_sentiment) in {elapsed_job4:.2f} seconds ---")
            except Exception as e:
                elapsed_job4 = time.time() - start_time_job4
                logging.error(f"--- Manual Run FAILED: Job 4 (update_news_sentiment) after {elapsed_job4:.2f} seconds: {e} ---", exc_info=True)
        # --- [END Job 4] ---

    elapsed_total = time.time() - start_time_total
    logging.info(f"--- Finished ALL executed Manual ETL jobs in {elapsed_total:.2f} seconds ---")
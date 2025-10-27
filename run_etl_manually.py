# run_etl_manually.py (รันทั้ง Job 1 และ Job 2 - FIXED: เปิดการรัน Job 1)
import logging
import time # Import time เพื่อจับเวลา

from app import server # Import server เพื่อใช้ app_context
# Import job ทั้งสองตัว
from etl import update_company_summaries, update_daily_prices

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    logging.info("Attempting to run ALL ETL jobs manually...")
    start_time_total = time.time()

    with server.app_context():
        
        # --- [FIXED] เปิดการทำงาน Job 1 --- (ต้องรันก่อน Job 2 เพื่อให้มี Tickers ใน DimCompany)
        logging.info("--- Starting Manual Run: Job 1 (update_company_summaries) ---")
        start_time_job1 = time.time()
        try:
            update_company_summaries() 
            elapsed_job1 = time.time() - start_time_job1
            logging.info(f"--- Finished Manual Run: Job 1 (update_company_summaries) in {elapsed_job1:.2f} seconds ---")
        except Exception as e:
            elapsed_job1 = time.time() - start_time_job1
            logging.error(f"--- Manual Run FAILED: Job 1 (update_company_summaries) after {elapsed_job1:.2f} seconds: {e} ---", exc_info=True)
        # --- [END FIXED] ---

        # --- Run Job 2 (Price Update) ---
        logging.info("--- Starting Manual Run: Job 2 (update_daily_prices) ---")
        start_time_job2 = time.time()
        try:
            # รัน Job 2
            # **ตอนนี้ Job 2 จะวน Loop ดึงทีละ Ticker ทำให้ไม่เกิด OOM**
            update_daily_prices(days_back=5*365) 
            elapsed_job2 = time.time() - start_time_job2
            logging.info(f"--- Finished Manual Run: Job 2 (update_daily_prices) in {elapsed_job2:.2f} seconds ---")
        except Exception as e:
            elapsed_job2 = time.time() - start_time_job2
            logging.error(f"--- Manual Run FAILED: Job 2 (update_daily_prices) after {elapsed_job2:.2f} seconds: {e} ---", exc_info=True)
            
    elapsed_total = time.time() - start_time_total
    logging.info(f"--- Finished ALL Manual ETL jobs in {elapsed_total:.2f} seconds ---")
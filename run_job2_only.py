# run_job2_only.py
import logging
import time
from app import server
from etl import update_daily_prices # Import เฉพาะ Job 2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    logging.info("Attempting to run ONLY ETL Job 2 manually...")
    start_time_total = time.time()

    with server.app_context():
        # --- Run Job 2 ---
        logging.info("--- Starting Manual Run: Job 2 (update_daily_prices) ---")
        start_time_job2 = time.time()
        try:
            # รัน Job 2 (ดึงข้อมูลย้อนหลัง 5 ปีตาม default)
            update_daily_prices(days_back=5*365)
            elapsed_job2 = time.time() - start_time_job2
            logging.info(f"--- Finished Manual Run: Job 2 (update_daily_prices) in {elapsed_job2:.2f} seconds ---")
        except Exception as e:
            elapsed_job2 = time.time() - start_time_job2
            logging.error(f"--- Manual Run FAILED: Job 2 (update_daily_prices) after {elapsed_job2:.2f} seconds: {e} ---", exc_info=True)

    elapsed_total = time.time() - start_time_total
    logging.info(f"--- Finished Manual ETL Job 2 run in {elapsed_total:.2f} seconds ---")
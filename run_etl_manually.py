# run_etl_manually.py (รันทั้ง Job 1 และ Job 2 - FIXED: เปิดการรัน Job 1 และ Resume Index)
import logging
import time 

from app import server 
# Import job ทั้งสองตัว
from etl import update_company_summaries, update_daily_prices 
# --- [NEW IMPORT] ---
from constants import ALL_TICKERS_SORTED_BY_MC
# --- [END NEW IMPORT] ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    logging.info("Attempting to run ALL ETL jobs manually...")
    start_time_total = time.time()

    # --- [CONFIGURATION FOR RESUME] ---
    # *********** ปรับค่านี้เพื่อรันต่อ ***********
    # Ticker ลำดับที่ 397 (PAG) คือ Index 396 ในรายการ 0-based
    # *** ตั้งค่าเป็น 0 เพื่อรันใหม่ทั้งหมด ***
    START_INDEX = 396 
    # *********** ปรับค่านี้เพื่อรันต่อ ***********
    # --- [END CONFIGURATION] ---

    with server.app_context():
        
        # --- [FIXED] เปิดการทำงาน Job 1 --- 
        logging.info(f"--- Starting Manual Run: Job 1 (update_company_summaries) from Index {START_INDEX} ---")
        start_time_job1 = time.time()
        try:
            # --- [MODIFICATION: Pass the sliced list] ---
            if START_INDEX > 0:
                 tickers_to_run = ALL_TICKERS_SORTED_BY_MC[START_INDEX:]
                 logging.info(f"Resuming Job 1 with {len(tickers_to_run)} tickers starting from {tickers_to_run[0]}")
            else:
                 tickers_to_run = ALL_TICKERS_SORTED_BY_MC
            
            # ส่งรายการ Ticker ที่เหลือไปให้ Job 1
            update_company_summaries(tickers_to_run) 
            # --- [END MODIFICATION] ---
            
            elapsed_job1 = time.time() - start_time_job1
            logging.info(f"--- Finished Manual Run: Job 1 (update_company_summaries) in {elapsed_job1:.2f} seconds ---")
        except Exception as e:
            elapsed_job1 = time.time() - start_time_job1
            logging.error(f"--- Manual Run FAILED: Job 1 (update_company_summaries) after {elapsed_job1:.2f} seconds: {e} ---", exc_info=True)

        # --- Run Job 2 (Price Update) ---
        logging.info("--- Starting Manual Run: Job 2 (update_daily_prices) ---")
        start_time_job2 = time.time()
        try:
            # รัน Job 2 จะเกิดขึ้นเมื่อ Job 1 เสร็จสมบูรณ์
            update_daily_prices(days_back=5*365) 
            elapsed_job2 = time.time() - start_time_job2
            logging.info(f"--- Finished Manual Run: Job 2 (update_daily_prices) in {elapsed_job2:.2f} seconds ---")
        except Exception as e:
            elapsed_job2 = time.time() - start_time_job2
            logging.error(f"--- Manual Run FAILED: Job 2 (update_daily_prices) after {elapsed_job2:.2f} seconds: {e} ---", exc_info=True)
            
    elapsed_total = time.time() - start_time_total
    logging.info(f"--- Finished ALL Manual ETL jobs in {elapsed_total:.2f} seconds ---")
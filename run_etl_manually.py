# run_etl_manually.py (รัน Job 2 เท่านั้น - FIXED: Comment Job 1 ออก)
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
    logging.info("Attempting to run ALL ETL jobs manually (Job 1 is commented out)...")
    start_time_total = time.time()

    with server.app_context():
        
        # --- [MODIFICATION: COMMENTED OUT] Job 1: Update Company Summaries ---
        # logging.info("--- Starting Manual Run: Job 1 (update_company_summaries) ---")
        # start_time_job1 = time.time()
        # try:
        #     # ถ้า Job 1 เสร็จสมบูรณ์แล้ว ให้ Comment ส่วนนี้ออก
        #     update_company_summaries(ALL_TICKERS_SORTED_BY_MC) 
        #     elapsed_job1 = time.time() - start_time_job1
        #     logging.info(f"--- Finished Manual Run: Job 1 (update_company_summaries) in {elapsed_job1:.2f} seconds ---")
        # except Exception as e:
        #     elapsed_job1 = time.time() - start_time_job1
        #     logging.error(f"--- Manual Run FAILED: Job 1 (update_company_summaries) after {elapsed_job1:.2f} seconds: {e} ---", exc_info=True)
        # --- [END MODIFICATION] ---

        # --- Run Job 2 (Price Update) ---
        logging.info("--- Starting Manual Run: Job 2 (update_daily_prices) ---")
        start_time_job2 = time.time()
        try:
            # รัน Job 2
            update_daily_prices(days_back=5*365) 
            elapsed_job2 = time.time() - start_time_job2
            logging.info(f"--- Finished Manual Run: Job 2 (update_daily_prices) in {elapsed_job2:.2f} seconds ---")
        except Exception as e:
            elapsed_job2 = time.time() - start_time_job2
            logging.error(f"--- Manual Run FAILED: Job 2 (update_daily_prices) after {elapsed_job2:.2f} seconds: {e} ---", exc_info=True)
            
    elapsed_total = time.time() - start_time_total
    logging.info(f"--- Finished ALL Manual ETL jobs in {elapsed_total:.2f} seconds ---")
# run_etl_manually.py
import time
import logging

from app import server 
# Import job ทั้งสองตัว
from etl import update_company_summaries, update_daily_prices, update_financial_statements, update_news_sentiment
# --- [NEW IMPORT] ---
from constants import ALL_TICKERS_SORTED_BY_MC, INDEX_TICKER_TO_NAME
# --- [END NEW IMPORT] ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    logging.info("Attempting to run ALL ETL jobs manually (Job 1 is commented out)...")
    start_time_total = time.time()

    with server.app_context():
        
        # --- Job 1: Update Company Summaries (COMMENTED OUT) ---
        logging.info("--- Starting Manual Run: Job 1 (update_company_summaries) ---")
        start_time_job1 = time.time()
        try:
            # ใช้ ALL_TICKERS_SORTED_BY_MC เพื่อให้ Job 1 อัปเดตข้อมูล Dim และ FactCompanySummary
            update_company_summaries(ALL_TICKERS_SORTED_BY_MC) 
            elapsed_job1 = time.time() - start_time_job1
            logging.info(f"--- Finished Manual Run: Job 1 (update_company_summaries) in {elapsed_job1:.2f} seconds ---")
        except Exception as e:
            elapsed_job1 = time.time() - start_time_job1
            logging.error(f"--- Manual Run FAILED: Job 1 (update_company_summaries) after {elapsed_job1:.2f} seconds: {e} ---", exc_info=True)
        # --- [END Job 1] ---

        # --- Run Job 2 (Price Update) ---
        logging.info("--- Starting Manual Run: Job 2 (update_daily_prices) ---")
        start_time_job2 = time.time()
        try:
            # รัน Job 2 เพื่อดึงข้อมูลราคาย้อนหลัง 5 ปี
            update_daily_prices(days_back=5*365) 
            elapsed_job2 = time.time() - start_time_job2
            logging.info(f"--- Finished Manual Run: Job 2 (update_daily_prices) in {elapsed_job2:.2f} seconds ---")
        except Exception as e:
            elapsed_job2 = time.time() - start_time_job2
            logging.error(f"--- Manual Run FAILED: Job 2 (update_daily_prices) after {elapsed_job2:.2f} seconds: {e} ---", exc_info=True)
            
        # # --- [NEW] Run Job 3 (Financial Statements)
        # logging.info("--- Starting Manual Run: Job 3 (update_financial_statements) ---")
        # start_time_job3 = time.time()
        # try:
        #     # รัน Job 3 ซึ่งจะดึงข้อมูลงบการเงินและใช้ Skip logic ที่เพิ่งเพิ่ม
        #     update_financial_statements() 
        #     elapsed_job3 = time.time() - start_time_job3
        #     logging.info(f"--- Finished Manual Run: Job 3 (update_financial_statements) in {elapsed_job3:.2f} seconds ---")
        # except Exception as e:
        #     elapsed_job3 = time.time() - start_time_job3
        #     logging.error(f"--- Manual Run FAILED: Job 3 (update_financial_statements) after {elapsed_job3:.2f} seconds: {e} ---", exc_info=True)
        # # --- [END Job 3] ---
        
        # # --- [NEW] Run Job 4 (News Sentiment)
        # logging.info("--- Starting Manual Run: Job 4 (update_news_sentiment) ---")
        # start_time_job4 = time.time()
        # try:
        #     # รัน Job 4 ซึ่งจะดึงข้อมูลข่าวและ sentiment สำหรับ 20 Ticker แรก
        #     update_news_sentiment() 
        #     elapsed_job4 = time.time() - start_time_job4
        #     logging.info(f"--- Finished Manual Run: Job 4 (update_news_sentiment) in {elapsed_job4:.2f} seconds ---")
        # except Exception as e:
        #     elapsed_job4 = time.time() - start_time_job4
        #     logging.error(f"--- Manual Run FAILED: Job 4 (update_news_sentiment) after {elapsed_job4:.2f} seconds: {e} ---", exc_info=True)
        # # --- [END Job 4] ---
            
    elapsed_total = time.time() - start_time_total
    logging.info(f"--- Finished ALL Manual ETL jobs in {elapsed_total:.2f} seconds ---")
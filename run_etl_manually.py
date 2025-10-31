# run_etl_manually.py (Modified to limit Job 2 & 3 to Top 500)
import time
import logging

from app import server
# Import job ทั้งสองตัว
from etl import update_company_summaries, update_daily_prices, update_financial_statements, update_news_sentiment
# --- [NEW IMPORT] ---
from constants import ALL_TICKERS_SORTED_BY_MC, INDEX_TICKER_TO_NAME, HISTORICAL_START_DATE
# --- [END NEW IMPORT] ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    logging.info("Attempting to run ETL jobs manually...")
    start_time_total = time.time()

    # --- [NEW] Define Top 500 Tickers (including Index Tickers) ---
    TOP_1500_BASE = ALL_TICKERS_SORTED_BY_MC[:1500]
    INDEX_TICKERS = list(INDEX_TICKER_TO_NAME.keys())
    # Use set to ensure uniqueness when combining
    TICKERS_FOR_JOB_2_3 = list(set(TOP_1500_BASE + INDEX_TICKERS))
    logging.info(f"Defined list for Job 2 & 3: Top 500 + Indices = {len(TICKERS_FOR_JOB_2_3)} unique tickers.")
    # --- [END NEW] ---

    with server.app_context():

        # # --- Job 1: Update Company Summaries (Runs for ALL tickers) ---
        # logging.info("--- Starting Manual Run: Job 1 (update_company_summaries for ALL tickers) ---")
        # start_time_job1 = time.time()
        # try:
        #     # ใช้ ALL_TICKERS_SORTED_BY_MC เพื่อให้ Job 1 อัปเดตข้อมูล Dim และ FactCompanySummary ทั้งหมด
        #     update_company_summaries(ALL_TICKERS_SORTED_BY_MC) # <<< ใช้ List เต็ม
        #     elapsed_job1 = time.time() - start_time_job1
        #     logging.info(f"--- Finished Manual Run: Job 1 (update_company_summaries) in {elapsed_job1:.2f} seconds ---")
        # except Exception as e:
        #     elapsed_job1 = time.time() - start_time_job1
        #     logging.error(f"--- Manual Run FAILED: Job 1 (update_company_summaries) after {elapsed_job1:.2f} seconds: {e} ---", exc_info=True)
        # # --- [END Job 1] ---

        # --- Run Job 2 (Price Update for Top 500 + Indices) ---
        logging.info(f"--- Starting Manual Run: Job 2 (update_daily_prices for Top 500 + Indices) ---")
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

        # # --- Run Job 3 (Financial Statements for Top 500) ---
        # logging.info(f"--- Starting Manual Run: Job 3 (update_financial_statements for Top 500) ---")
        # start_time_job3 = time.time()
        # try:
        #     # รัน Job 3 โดยส่ง List ที่จำกัดไว้เข้าไป (Job 3 ไม่รวม Index Tickers อยู่แล้ว)
        #     # เราใช้ TICKERS_FOR_JOB_2_3 ไปเลยก็ได้ เพราะใน Job 3 มันจะกรอง Index ออกเองถ้าไม่มีใน DimCompany
        #     update_financial_statements(tickers_list_override=TICKERS_FOR_JOB_2_3) # <<< ใช้ List ที่จำกัด
        #     elapsed_job3 = time.time() - start_time_job3
        #     logging.info(f"--- Finished Manual Run: Job 3 (update_financial_statements) in {elapsed_job3:.2f} seconds ---")
        # except Exception as e:
        #     elapsed_job3 = time.time() - start_time_job3
        #     logging.error(f"--- Manual Run FAILED: Job 3 (update_financial_statements) after {elapsed_job3:.2f} seconds: {e} ---", exc_info=True)
        # # --- [END Job 3] ---

        # # --- [KEEP COMMENTED] Run Job 4 (News Sentiment for Top 20) ---
        # logging.info("--- Starting Manual Run: Job 4 (update_news_sentiment for Top 20) ---")
        # start_time_job4 = time.time()
        # try:
        #     # รัน Job 4 ซึ่งจำกัดแค่ 20 ตัวแรกอยู่แล้ว ไม่ต้องส่ง override
        #     update_news_sentiment()
        #     elapsed_job4 = time.time() - start_time_job4
        #     logging.info(f"--- Finished Manual Run: Job 4 (update_news_sentiment) in {elapsed_job4:.2f} seconds ---")
        # except Exception as e:
        #     elapsed_job4 = time.time() - start_time_job4
        #     logging.error(f"--- Manual Run FAILED: Job 4 (update_news_sentiment) after {elapsed_job4:.2f} seconds: {e} ---", exc_info=True)
        # # --- [END Job 4] ---

    elapsed_total = time.time() - start_time_total
    logging.info(f"--- Finished ALL executed Manual ETL jobs in {elapsed_total:.2f} seconds ---")
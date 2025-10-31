# # clear_all_etl_data.py

# from app import db, server, DimCompany, FactCompanySummary, FactDailyPrices, FactFinancialStatements, FactNewsSentiment
# import logging

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def clear_all_etl_tables():
#     """
#     ลบข้อมูลทั้งหมดในตาราง DimCompany, FactCompanySummary, FactDailyPrices, 
#     FactFinancialStatements, และ FactNewsSentiment เพื่อเตรียมรัน ETL ใหม่ทั้งหมด.
#     """
#     with server.app_context():
#         try:
#             logging.info("--- Starting deletion of ALL ETL data ---")
            
#             # ลบข้อมูลใน Fact Tables ก่อน (ข้อมูลที่อ้างอิงถึง DimCompany)
            
#             # Job 2: FactDailyPrices
#             deleted_prices = db.session.query(FactDailyPrices).delete()
#             logging.info(f"Deleted {deleted_prices} records from FactDailyPrices (Job 2).")

#             # Job 3: FactFinancialStatements
#             deleted_financials = db.session.query(FactFinancialStatements).delete()
#             logging.info(f"Deleted {deleted_financials} records from FactFinancialStatements (Job 3).")
            
#             # Job 4: FactNewsSentiment
#             deleted_news = db.session.query(FactNewsSentiment).delete()
#             logging.info(f"Deleted {deleted_news} records from FactNewsSentiment (Job 4).")

#             # Job 1: FactCompanySummary
#             deleted_summaries = db.session.query(FactCompanySummary).delete()
#             logging.info(f"Deleted {deleted_summaries} records from FactCompanySummary (Job 1).")
            
#             # Job 1: DimCompany (ตารางมิติ)
#             deleted_dim = db.session.query(DimCompany).delete()
#             logging.info(f"Deleted {deleted_dim} records from DimCompany (Job 1).")
            
#             db.session.commit()
#             logging.info("✅ Database commit successful. All ETL data tables are now empty.")
            
#         except Exception as e:
#             logging.error(f"An error occurred during database operation: {e}")
#             db.session.rollback()

# if __name__ == '__main__':
#     clear_all_etl_tables()

# clear_job2_data.py
import logging
import time

# นำเข้าสิ่งที่จำเป็นจากโปรเจกต์ของคุณ
# FactDailyPrices คือโมเดลตารางสำหรับ Job 2
from app import db, server, FactDailyPrices

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clear_job2_prices_data():
    """ฟังก์ชันสำหรับลบข้อมูลทั้งหมดจากตาราง FactDailyPrices (Job 2)."""
    job_name = "Data Clear: FactDailyPrices (Job 2)"
    logging.info(f"--- Starting {job_name} ---")
    start_time = time.time()

    # ใช้ server.app_context() เพื่อให้สามารถเข้าถึงฐานข้อมูลได้
    with server.app_context():
        try:
            logging.info("Executing DELETE command for FactDailyPrices...")
            
            # ใช้ delete() เพื่อลบทุกแถวในตาราง FactDailyPrices
            # 'fetch' คือการกำหนดให้ SQLAlchemy นับจำนวนแถวที่ถูกลบ
            num_rows_deleted = db.session.query(FactDailyPrices).delete(synchronize_session='fetch')
            
            # ยืนยันการลบ (commit)
            db.session.commit()
            
            elapsed_time = time.time() - start_time
            logging.info(f"--- Finished {job_name} in {elapsed_time:.2f} seconds ---")
            logging.info(f"✅ Successfully deleted {num_rows_deleted} rows from FactDailyPrices.")

        except Exception as e:
            # ยกเลิก (rollback) หากเกิดข้อผิดพลาดในการลบ
            db.session.rollback()
            logging.error(f"--- FAILED {job_name} ---")
            logging.error(f"❌ Error during deletion: {e}", exc_info=True)

if __name__ == "__main__":
    clear_job2_prices_data()
# clear_all_etl_data.py

from app import db, server, DimCompany, FactCompanySummary, FactDailyPrices, FactFinancialStatements, FactNewsSentiment
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clear_all_etl_tables():
    """
    ลบข้อมูลทั้งหมดในตาราง DimCompany, FactCompanySummary, FactDailyPrices, 
    FactFinancialStatements, และ FactNewsSentiment เพื่อเตรียมรัน ETL ใหม่ทั้งหมด.
    """
    with server.app_context():
        try:
            logging.info("--- Starting deletion of ALL ETL data ---")
            
            # ลบข้อมูลใน Fact Tables ก่อน (ข้อมูลที่อ้างอิงถึง DimCompany)
            
            # Job 2: FactDailyPrices
            deleted_prices = db.session.query(FactDailyPrices).delete()
            logging.info(f"Deleted {deleted_prices} records from FactDailyPrices (Job 2).")

            # Job 3: FactFinancialStatements
            deleted_financials = db.session.query(FactFinancialStatements).delete()
            logging.info(f"Deleted {deleted_financials} records from FactFinancialStatements (Job 3).")
            
            # Job 4: FactNewsSentiment
            deleted_news = db.session.query(FactNewsSentiment).delete()
            logging.info(f"Deleted {deleted_news} records from FactNewsSentiment (Job 4).")

            # Job 1: FactCompanySummary
            deleted_summaries = db.session.query(FactCompanySummary).delete()
            logging.info(f"Deleted {deleted_summaries} records from FactCompanySummary (Job 1).")
            
            # Job 1: DimCompany (ตารางมิติ)
            deleted_dim = db.session.query(DimCompany).delete()
            logging.info(f"Deleted {deleted_dim} records from DimCompany (Job 1).")
            
            db.session.commit()
            logging.info("✅ Database commit successful. All ETL data tables are now empty.")
            
        except Exception as e:
            logging.error(f"An error occurred during database operation: {e}")
            db.session.rollback()

if __name__ == '__main__':
    clear_all_etl_tables()
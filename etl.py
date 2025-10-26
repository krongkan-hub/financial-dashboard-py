# etl.py
import logging
import pandas as pd
from datetime import datetime

# Import สิ่งที่จำเป็นจากโปรเจกต์ของเรา
from app import db, server, DimCompany, FactCompanySummary
from data_handler import get_competitor_data
from constants import ALL_TICKERS_SORTED_BY_MC

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def update_company_summaries():
    """
    ETL Job 1: ดึงข้อมูลสรุปบริษัททั้งหมด
    (ข้อมูลจาก get_competitor_data)
    """
    # --- [สำคัญมาก!] ---
    # งานที่ทำงานใน Background Thread ต้องใช้ app_context
    # เพื่อให้สามารถเชื่อมต่อกับฐานข้อมูล (db) ได้
    with server.app_context():
        logging.info("Starting ETL Job: [update_company_summaries]...")
        
        # 1. Extract: ดึงข้อมูลสดจาก API
        #    (เราใช้ tuple ตามที่ data_handler.py ต้องการ)
        try:
            tickers_tuple = tuple(ALL_TICKERS_SORTED_BY_MC)
            df = get_competitor_data(tickers_tuple)
            
            if df.empty:
                logging.warning("ETL Job: get_competitor_data returned an empty DataFrame. Skipping update.")
                return

        except Exception as e:
            logging.error(f"ETL Job: Failed during data extraction (get_competitor_data): {e}", exc_info=True)
            return

        # 2. Transform & 3. Load
        #    วนลูปข้อมูลที่ได้มา (DataFrame) เพื่อบันทึกลง DB
        today = datetime.utcnow().date()
        processed_count = 0
        
        for _, row in df.iterrows():
            try:
                # --- 3.1: Load/Update DimCompany (ตารางมิติ) ---
                # (เราใช้ db.session.merge() ซึ่งทำหน้าที่เป็น "UPSERT")
                # (ถ้า Ticker นี้ยังไม่มี -> INSERT, ถ้ามีแล้ว -> UPDATE)
                company_info = DimCompany(
                    ticker=row['Ticker'],
                    company_name=row.get('longName'), # .get() เพื่อป้องกัน Error ถ้าไม่มี
                    logo_url=row.get('logo_url'),
                    sector=row.get('sector')
                )
                db.session.merge(company_info)

                # --- 3.2: Load/Update FactCompanySummary (ตารางข้อเท็จจริง) ---
                # (เราใช้ merge เช่นกัน มันจะเช็ค UniqueConstraint ที่เราสร้างไว้)
                # (ถ้า Ticker นี้ + วันที่วันนี้ ยังไม่มี -> INSERT, ถ้ามีแล้ว -> UPDATE)
                summary = FactCompanySummary(
                    ticker=row['Ticker'],
                    date_updated=today,
                    
                    # --- [สำคัญ] แมปชื่อคอลัมน์จาก DataFrame ไปยัง Model ---
                    price=row.get('Price'),
                    market_cap=row.get('Market Cap'),
                    beta=row.get('Beta'),
                    pe_ratio=row.get('P/E'),
                    pb_ratio=row.get('P/B'),
                    ev_ebitda=row.get('EV/EBITDA'),
                    revenue_growth_yoy=row.get('Revenue Growth (YoY)'),
                    revenue_cagr_3y=row.get('Revenue CAGR (3Y)'),
                    net_income_growth_yoy=row.get('Net Income Growth (YoY)'),
                    roe=row.get('ROE'),
                    de_ratio=row.get('D/E Ratio'),
                    operating_margin=row.get('Operating Margin'),
                    cash_conversion=row.get('Cash Conversion')
                )
                db.session.merge(summary)
                processed_count += 1
            
            except Exception as e:
                logging.error(f"ETL Job: Failed to merge data for ticker {row.get('Ticker')}: {e}")
                db.session.rollback() # ย้อนกลับเฉพาะ Ticker ที่มีปัญหา
                continue # ไปทำ Ticker ตัวต่อไป

        # --- 3.3: Commit การเปลี่ยนแปลงทั้งหมด ---
        try:
            db.session.commit()
            logging.info(f"ETL Job: [update_company_summaries]... COMPLETED. Processed {processed_count} tickers.")
        except Exception as e:
            logging.error(f"ETL Job: Failed during final commit: {e}", exc_info=True)
            db.session.rollback()

def update_daily_prices():
    """
    ETL Job 2: ดึงข้อมูลราคาย้อนหลัง (ยังไม่ทำในขั้นตอนนี้)
    (ในอนาคต เราจะเขียน Logic ที่นี่เพื่อดึง yf.history() มาเติมตาราง FactDailyPrices)
    """
    logging.info("ETL Job: [update_daily_prices]... (Not Implemented Yet)")
    pass
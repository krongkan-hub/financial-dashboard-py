# check_db_data.py
import logging
import pandas as pd
from app import db, server, FactFinancialStatements, DimCompany
from sqlalchemy import func, distinct

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("===== Starting Database Diagnostic Script =====")

try:
    with server.app_context():
        logging.info("Connecting to database...")
        
        # 1. ตรวจสอบจำนวนข้อมูลทั้งหมด
        total_count = db.session.query(func.count(FactFinancialStatements.ticker)).scalar()
        logging.info(f"1. [TOTAL COUNT] 'FactFinancialStatements' table has: {total_count} rows")

        if total_count == 0:
            logging.error("Database table 'FactFinancialStatements' is empty. This is the problem.")
            logging.error("Please run your ETL process (e.g., 'python run_etl_manually.py') to populate the database.")
        else:
            # 2. ตรวจสอบช่วงวันที่
            min_date = db.session.query(func.min(FactFinancialStatements.report_date)).scalar()
            max_date = db.session.query(func.max(FactFinancialStatements.report_date)).scalar()
            logging.info(f"2. [DATE RANGE] Data ranges from {min_date} to {max_date}")

            if max_date and max_date.year < 2010:
                logging.warning(f"  -> WARNING: Your latest data ({max_date}) is older than the script's start_year (2010).")

            # 3. ตรวจสอบ 'statement_type' ที่มีอยู่จริง
            distinct_statements = db.session.query(distinct(FactFinancialStatements.statement_type)).all()
            statement_types = [s[0] for s in distinct_statements]
            logging.info(f"3. [STATEMENT TYPES] Found types: {statement_types}")

            if 'income' not in statement_types and 'balance' not in statement_types:
                logging.warning(f"  -> WARNING: Could not find 'income' or 'balance'. Found {statement_types} instead. (Case sensitive!)")

            # 4. ตรวจสอบ 'metric_name' ที่มีอยู่จริง (สำคัญที่สุด)
            logging.info("4. [METRIC NAMES] Fetching first 50 distinct metric_names...")
            distinct_metrics = db.session.query(distinct(FactFinancialStatements.metric_name)).limit(50).all()
            metric_names = [m[0] for m in distinct_metrics]
            
            if not metric_names:
                logging.warning("  -> Found 0 distinct metric names.")
            else:
                logging.info("  -> Found the following metric names (sample):")
                for name in metric_names:
                    print(f"     - {name}")

            # 5. ลอง Query ที่ล้มเหลวโดยตรง (แต่ใช้ List ที่เราแก้)
            logging.info("5. [TEST QUERY] Trying the ML script's query logic...")
            required_metrics = [
                'Accumulated Deficit', 'Accumulated_Deficit',
                'Current Ratio', 'Current_Ratio',
                'Total Equity', 'Stockholders Equity', 'Total Stockholders Equity'
            ]
            
            test_query = db.session.query(FactFinancialStatements.metric_name) \
                .filter(FactFinancialStatements.metric_name.in_(required_metrics)) \
                .filter(FactFinancialStatements.statement_type.in_(['income', 'balance', 'cashflow'])) \
                .filter(FactFinancialStatements.report_date >= '2010-01-01')
            
            found_metrics = test_query.distinct().all()
            
            if not found_metrics:
                logging.warning("  -> TEST QUERY FAILED: Found 0 rows matching all criteria.")
                logging.warning("  -> This confirms the 'No financial statement data found' error.")
            else:
                logging.info(f"  -> TEST QUERY SUCCESS: Found matching metrics: {[m[0] for m in found_metrics]}")

except Exception as e:
    logging.error(f"An error occurred during database connection or query: {e}", exc_info=True)

logging.info("===== Database Diagnostic Script Finished =====")
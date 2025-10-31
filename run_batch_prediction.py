import pandas as pd
import joblib
import logging
import sys
import datetime
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# --- (USER REQUEST 2) Import Feature Logic ---
# Import ลอจิกการสร้าง Feature และ "สัญญา" 22 features จากไฟล์ใหม่
try:
    from ml_risk_features import engineer_features_for_prediction, ML_RISK_BASE_FEATURES
except ImportError:
    print("FATAL: ไม่พบไฟล์ ml_risk_features.py กรุณาสร้างไฟล์ตามคำแนะนำ")
    sys.exit(1)

# --- 1. Configuration ---

# --- (USER REQUEST 1) Load Secrets from .env file ---
load_dotenv()
DB_URL = os.getenv('SQLALCHEMY_DATABASE_URI')
if not DB_URL:
    print("FATAL: ไม่ได้ตั้งค่า SQLALCHEMY_DATABASE_URI ในไฟล์ .env")
    sys.exit(1)
# ---------------------------------------------------

# ที่อยู่ของ Artifacts (ตรงกับใน Log ของคุณ)
ARTIFACT_PATHS = {
    'MODEL': 'models/trained_risk_model_xgb.joblib',
    'IMPUTER': 'models/imputer.joblib',
    'SCALER': 'models/scaler.joblib'
}

# --- 2. Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# --- 3. Main Execution ---
def run_prediction_pipeline():
    log.info("===== Starting Batch Risk Prediction Pipeline =====")
    start_time = datetime.datetime.now()
    
    try:
        # --- Step 1: Load Artifacts & Connect DB ---
        log.info("1. Loading model, imputer, and scaler...")
        model = joblib.load(ARTIFACT_PATHS['MODEL'])
        imputer = joblib.load(ARTIFACT_PATHS['IMPUTER'])
        scaler = joblib.load(ARTIFACT_PATHS['SCALER'])
        
        log.info(f"   Connecting to database (using config)...")
        engine = create_engine(DB_URL)

        # --- Step 2: Fetch Latest Data ---
        log.info("2. Fetching latest financial data for all companies...")
        # (ฝึก SQL): ใช้ DISTINCT ON (ticker) เพื่อเอาข้อมูลงบล่าสุดของแต่ละบริษัท
        sql_query = text("""
            SELECT DISTINCT ON (ticker) *
            FROM fact_financial_statements
            ORDER BY ticker, report_date DESC;
        """)
        
        with engine.connect() as conn:
            raw_data_df = pd.read_sql(sql_query, conn)
        
        if raw_data_df.empty:
            log.warning("No data fetched from fact_financial_statements. Stopping.")
            return

        log.info(f"   Fetched {len(raw_data_df)} unique company records.")

        # --- Step 3: Engineer Features (Using imported logic) ---
        log.info("3. Engineering features for prediction...")
        # (USER REQUEST 2) เรียกใช้ฟังก์ชันที่ Import มา
        # X_raw_features จะมี 22 คอลัมน์ตาม "สัญญา"
        X_raw_features, tickers = engineer_features_for_prediction(raw_data_df)

        # --- Step 4: Preprocess (Impute & Scale) ---
        log.info("4. Applying Imputer and Scaler...")
        # (ฝึก ML): ใช้ .transform() เท่านั้น!
        X_imputed_array = imputer.transform(X_raw_features)
        
        # --- (NEW) Re-build 41 Feature Names Dynamically ---
        # สร้างชื่อ 41 features (22 base + 19 indicators)
        # โดยอิงจาก "imputer" ที่โหลดมา เพื่อให้ตรงกับตอนเทรน 100%
        base_features = ML_RISK_BASE_FEATURES
        indicator_indices = imputer.indicator_.features_
        indicator_names = [f'Missing_{base_features[i]}' for i in indicator_indices]
        
        final_feature_names = base_features + indicator_names
        log.info(f"   Reconstructed {len(final_feature_names)} feature names (Base + Indicators).")
        # ----------------------------------------------------

        # สร้าง DataFrame ที่มี 41 features ก่อน Scale
        X_imputed_df = pd.DataFrame(
            X_imputed_array, 
            columns=final_feature_names, 
            index=X_raw_features.index
        )
        
        # Scale ทั้ง 41 features
        X_scaled_array = scaler.transform(X_imputed_df)
        log.info("   Preprocessing complete.")

        # --- Step 5: Predict Probabilities ---
        log.info("5. Predicting default probabilities...")
        # [:, 1] คือการดึงความน่าจะเป็นของ Class 1 (Default)
        probabilities = model.predict_proba(X_scaled_array)[:, 1]
        log.info("   Prediction complete.")

        # --- Step 6: Prepare & Update Database ---
        log.info("6. Preparing results for database update...")
        df_results = pd.DataFrame({
            'ticker_key': tickers,
            'new_prob': probabilities
        })

        # (ฝึก SQL): ใช้ Temp Table เพื่อ Update อย่างรวดเร็ว (วิธีแบบ Quant)
        log.info(f"   Writing {len(df_results)} predictions to temp table 'temp_risk_predictions'...")
        with engine.begin() as conn:
            # 1. ส่ง DataFrame เข้าตารางชั่วคราว
            df_results.to_sql(
                'temp_risk_predictions',
                conn,
                if_exists='replace',
                index=False
            )
            
            log.info("7. Updating main table 'fact_company_summary'...")
            # 2. (ฝึก SQL): รัน UPDATE โดย JOIN จากตารางชั่วคราว
            update_sql = text("""
                UPDATE fact_company_summary fcs
                SET predicted_default_prob = trp.new_prob
                FROM temp_risk_predictions trp
                WHERE fcs.ticker = trp.ticker_key;
            """)
            result = conn.execute(update_sql)
            
            log.info(f"   Successfully updated {result.rowcount} rows in fact_company_summary.")

            # 3. ลบตารางชั่วคราว
            conn.execute(text("DROP TABLE IF EXISTS temp_risk_predictions;"))
            log.info("   Temp table dropped.")

    except Exception as e:
        log.critical(f"An error occurred during the prediction pipeline: {e}", exc_info=True)
    
    finally:
        end_time = datetime.datetime.now()
        log.info(f"===== Pipeline Finished in {end_time - start_time} =====")

if __name__ == "__main__":
    # ติดตั้ง library ที่จำเป็น
    try:
        import dotenv
    except ImportError:
        print("Warning: 'python-dotenv' not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
        
    run_prediction_pipeline()
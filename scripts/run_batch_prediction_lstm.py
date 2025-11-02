# scripts/run_batch_prediction_lstm.py
# สคริปต์สำหรับทำนายความเสี่ยงด้วยโมเดล LSTM ที่เทรนแล้ว

import pandas as pd
import joblib
import logging
import sys
import datetime
import os
import numpy as np

# --- [NEU] Import สำหรับ Deep Learning ---
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
# --- [END NEU] ---

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# เพิ่ม Path ไปยัง Root Directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Logic การสร้าง Feature จาก app/ml/
try:
    from app.ml.ml_risk_features import engineer_features_for_prediction, ML_RISK_BASE_FEATURES
except ImportError:
    print("FATAL: ไม่พบไฟล์ ml_risk_features.py กรุณาสร้างไฟล์ตามคำแนะนำ")
    sys.exit(1)

# --- 1. Configuration ---
load_dotenv()
DB_URL = os.getenv('SQLALCHEMY_DATABASE_URI')
if not DB_URL:
    print("FATAL: ไม่ได้ตั้งค่า SQLALCHEMY_DATABASE_URI ในไฟล์ .env")
    sys.exit(1)

# --- [MODIFIED] ชี้ไปที่ Artifacts ของ LSTM ---
ARTIFACT_PATHS = {
    'MODEL': 'models/trained_risk_model_lstm.keras',
    'IMPUTER': 'models/imputer_lstm.joblib',
    'SCALER': 'models/scaler_lstm.joblib'
}
SEQUENCE_LENGTH = 12 # ต้องตรงกับตอนเทรน (ใน run_ml_risk.py)
# --- [END MODIFIED] ---

# --- 2. Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# --- 3. Helper Functions ---

def load_artifacts():
    """โหลด Model, Imputer, Scaler"""
    try:
        log.info("1. Loading model, imputer, and scaler...")
        model = load_model(ARTIFACT_PATHS['MODEL']) # Keras load
        imputer = joblib.load(ARTIFACT_PATHS['IMPUTER'])
        scaler = joblib.load(ARTIFACT_PATHS['SCALER'])
        log.info("   Artifacts loaded successfully.")
        return model, imputer, scaler
    except Exception as e:
        log.critical(f"Failed to load artifacts: {e}", exc_info=True)
        return None, None, None

def fetch_data(engine):
    """
    [MODIFIED] ดึงข้อมูลงบการเงิน 12 ไตรมาสล่าสุด (หรือน้อยกว่า) 
    สำหรับ *ทุก* บริษัท เพื่อสร้าง Sequences
    """
    try:
        log.info("2. Fetching latest 12 quarters of financial data for all companies...")
        
        # (ฝึก SQL): ใช้ Window Function (ROW_NUMBER) เพื่อดึง 12 แถวล่าสุดของแต่ละ ticker
        sql_query = text(f"""
            WITH RankedReports AS (
                SELECT 
                    *, 
                    ROW_NUMBER() OVER(
                        PARTITION BY ticker 
                        ORDER BY report_date DESC
                    ) as rn
                FROM fact_financial_statements
            )
            SELECT * FROM RankedReports 
            WHERE rn <= {SEQUENCE_LENGTH}
            ORDER BY ticker, report_date ASC; 
        """) # ORDER BY ASC เพื่อให้พร้อมสร้าง Sequence
        
        with engine.connect() as conn:
            raw_data_df = pd.read_sql(sql_query, conn)
        
        if raw_data_df.empty:
            log.warning("No data fetched from fact_financial_statements. Stopping.")
            return pd.DataFrame()

        log.info(f"   Fetched {len(raw_data_df)} records (max {SEQUENCE_LENGTH} per ticker).")
        return raw_data_df
        
    except Exception as e:
        log.error(f"Failed to fetch data: {e}", exc_info=True)
        return pd.DataFrame()

def preprocess_and_predict(raw_data_df, model, imputer, scaler):
    """
    [MODIFIED] สร้าง Features, Impute/Scale, สร้าง Sequences, 
    และทำนายผล (Predict)
    """
    try:
        log.info("3. Engineering features for all fetched quarters...")
        # 1. สร้าง 22 Raw Features (จะใช้เวลาสักครู่)
        X_raw_features, _ = engineer_features_for_prediction(raw_data_df)

        # 2. Impute (จัดการคอลัมน์ NaN ทั้งหมดก่อน)
        log.info("4. Applying Imputer and Scaler...")
        all_nan_cols = X_raw_features.columns[X_raw_features.isnull().all()].tolist()
        if all_nan_cols:
            log.warning(f"Forcing imputation for all-NaN features by filling with 0: {all_nan_cols}")
            X_raw_features[all_nan_cols] = X_raw_features[all_nan_cols].fillna(0)  
            
        X_imputed_array = imputer.transform(X_raw_features)
        
        # 3. สร้างชื่อ Features ทั้งหมด (รวม Indicators)
        base_features = ML_RISK_BASE_FEATURES
        indicator_indices = imputer.indicator_.features_
        indicator_names = [f'Missing_{base_features[i]}' for i in indicator_indices]
        final_feature_names = base_features + indicator_names
        
        X_imputed_df = pd.DataFrame(
            X_imputed_array, 
            columns=final_feature_names, 
            index=X_raw_features.index
        )
        
        # 4. Scale
        X_scaled = scaler.transform(X_imputed_df)
        X_scaled_df = pd.DataFrame(X_scaled, columns=final_feature_names, index=X_raw_features.index)
        
        # 5. สร้าง Sequences (จัดกลุ่มตาม Ticker)
        log.info("5. Creating sequences for prediction...")
        X_scaled_df['ticker'] = raw_data_df['ticker']
        
        grouped = X_scaled_df.groupby('ticker')
        X_batch_list = []
        tickers_for_update = []
        
        for ticker, group in grouped:
            # ไม่ต้อง sort_values(by='report_date') เพราะเราทำใน SQL Query แล้ว
            sequence_data = group[final_feature_names].values
            X_batch_list.append(sequence_data)
            tickers_for_update.append(ticker)
            
        # 6. Padding (เติม 0 ด้านหน้าหากข้อมูลไม่ถึง 12 ไตรมาส)
        X_padded_batch = pad_sequences(
            X_batch_list, 
            maxlen=SEQUENCE_LENGTH, 
            dtype='float32', 
            padding='pre', 
            truncating='pre'
        )
        
        log.info(f"   Batch created with shape: {X_padded_batch.shape}")
        
        # 7. Predict
        log.info("6. Predicting default probabilities...")
        probabilities = model.predict(X_padded_batch).ravel()
        log.info("   Prediction complete.")

        # 8. สร้าง DataFrame ผลลัพธ์
        df_results = pd.DataFrame({
            'ticker_key': tickers_for_update,
            'new_prob': probabilities
        })
        return df_results

    except Exception as e:
        log.error(f"Failed during preprocessing or prediction: {e}", exc_info=True)
        return pd.DataFrame()

def update_database(engine, df_results):
    """
    [UNCHANGED] อัปเดต Database โดยใช้ Temp Table
    (FIXED: 2025-11-02 - แก้ไข SQL syntax สำหรับ PostgreSQL)
    """
    if df_results.empty:
        log.warning("No results to update in database.")
        return

    try:
        log.info(f"7. Writing {len(df_results)} predictions to temp table 'temp_risk_predictions_lstm'...")
        with engine.begin() as conn:
            # 1. ส่ง DataFrame เข้าตารางชั่วคราว
            df_results.to_sql(
                'temp_risk_predictions_lstm',
                conn,
                if_exists='replace',
                index=False
            )
            
            log.info("8. Updating main table 'fact_company_summary'...")
            
            # --- [FIXED SQL SYNTAX] ---
            # ย้าย logic การ JOIN ทั้งหมดมาไว้ใน WHERE clause 
            # ซึ่งเป็น syntax ที่ PostgreSQL รองรับ
            update_sql = text("""
                WITH latest_summary AS (
                    SELECT id, 
                           ticker, 
                           ROW_NUMBER() OVER(
                               PARTITION BY ticker 
                               ORDER BY date_updated DESC
                           ) as rn
                    FROM fact_company_summary
                    WHERE ticker IN (SELECT DISTINCT ticker_key FROM temp_risk_predictions_lstm)
                )
                UPDATE fact_company_summary fcs
                SET predicted_default_prob = trp.new_prob
                FROM temp_risk_predictions_lstm trp, latest_summary ls
                WHERE fcs.id = ls.id           -- Join condition 1
                  AND fcs.ticker = trp.ticker_key -- Join condition 2
                  AND ls.rn = 1;                  -- Filter condition
            """)
            # --- [END FIXED SQL SYNTAX] ---
            
            result = conn.execute(update_sql)
            log.info(f"   Successfully updated {result.rowcount} rows in fact_company_summary.")

            # 3. ลบตารางชั่วคราว
            conn.execute(text("DROP TABLE IF EXISTS temp_risk_predictions_lstm;"))
            log.info("   Temp table dropped.")
            
    except Exception as e:
        log.error(f"Failed to update database: {e}", exc_info=True)

# --- 4. Main Execution ---
def run_prediction_pipeline():
    log.info("===== Starting Batch Risk Prediction Pipeline (LSTM) =====")
    start_time = datetime.datetime.now()
    
    model, imputer, scaler = load_artifacts()
    
    if model and imputer and scaler:
        engine = create_engine(DB_URL)
        raw_data_df = fetch_data(engine)
        
        if not raw_data_df.empty:
            df_results = preprocess_and_predict(raw_data_df, model, imputer, scaler)
            
            if not df_results.empty:
                update_database(engine, df_results)
            else:
                log.error("Prediction resulted in empty dataframe. No updates performed.")
        else:
            log.error("Data fetching failed. No updates performed.")
    else:
        log.critical("Could not load all artifacts. Aborting pipeline.")
    
    end_time = datetime.datetime.now()
    log.info(f"===== Pipeline Finished in {end_time - start_time} =====")

if __name__ == "__main__":
    run_prediction_pipeline()
# run_ml_cluster.py (เวอร์ชัน Hybrid + New Features + Adjusted Ratio)

import logging
import pandas as pd
import numpy as np
from datetime import date

# --- Machine Learning ---
# ถ้ายังไม่มี ต้อง pip install scikit-learn kneed matplotlib ก่อนนะครับ
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from kneed import KneeLocator # ใช้หา "ข้อศอก" อัตโนมัติ

# --- Database Interaction ---
from app import db, server, FactCompanySummary
from sqlalchemy import func, update

# ตั้งค่า Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
K_RANGE = range(3, 21) # กำหนดช่วง k
MIN_K = min(K_RANGE)

# --- [ปรับปรุง] คอลัมน์ (Features) ที่จะใช้ ---
# (เพิ่ม ebitda_margin, cash_conversion, fcf_sales_ratio และจะเพิ่ม log_market_cap ทีหลัง)
FEATURES_FOR_CLUSTERING = [
    'pe_ratio', 'pb_ratio', 'ev_ebitda', # Valuation
    'revenue_growth_yoy', 'net_income_growth_yoy', # Growth
    'roe', 'operating_margin', 'ebitda_margin', # Profitability
    'de_ratio', # Leverage
    'beta', # Risk
    'cash_conversion', # Quality
]

# --- Sanity Check Configuration ---
MIN_CLUSTER_SIZE = 10         # กลุ่มต้องมีหุ้นอย่างน้อยกี่ตัว
MAX_CLUSTER_SIZE_RATIO = 0.35 # --- [ปรับปรุง] ลดเกณฑ์ลงเหลือ 35% ---

# --- ฟังก์ชันดึงข้อมูล ---
def get_latest_financial_data() -> pd.DataFrame:
    """ดึงข้อมูลล่าสุดจาก FactCompanySummary สำหรับทุก Ticker"""
    logging.info("Querying latest financial data from FactCompanySummary...")
    try:
        with server.app_context():
            latest_date = db.session.query(func.max(FactCompanySummary.date_updated)).scalar()
            if not latest_date:
                logging.error("No data found in FactCompanySummary.")
                return pd.DataFrame()
            logging.info(f"Latest data date: {latest_date}")
            # --- [ปรับปรุง] Query ทุกคอลัมน์ที่ต้องการ รวมถึง market_cap ---
            query = db.session.query(FactCompanySummary).filter(
                FactCompanySummary.date_updated == latest_date
            )
            df = pd.read_sql(query.statement, db.engine)
            logging.info(f"Successfully queried {len(df)} records.")
            return df
    except Exception as e:
        logging.error(f"Error querying data: {e}", exc_info=True)
        return pd.DataFrame()

# --- [ปรับปรุง] ฟังก์ชันเตรียมข้อมูล ---
def preprocess_data(df: pd.DataFrame) -> (pd.DataFrame, list): # <<< คืนค่า list ของ features ที่ใช้ด้วย
    """เตรียมข้อมูลให้พร้อมสำหรับ K-Means (ใช้ Features ใหม่)"""
    logging.info("Preprocessing data...")
    if df.empty: return pd.DataFrame(), []

    # 1. เลือก Features + ticker + market_cap
    required_db_cols = ['ticker'] + FEATURES_FOR_CLUSTERING # Features ที่ต้องมีใน DB
    if 'market_cap' in df.columns:
        cols_to_select = required_db_cols + ['market_cap']
        has_market_cap = True
    else:
        logging.warning("Market Cap column not found in data.")
        cols_to_select = required_db_cols
        has_market_cap = False

    # เช็คว่ามีคอลัมน์ Feature ใหม่ครบหรือไม่
    missing_features = [f for f in FEATURES_FOR_CLUSTERING if f not in df.columns]
    if missing_features:
        logging.error(f"Missing required features from DB query: {missing_features}. Clustering cannot proceed.")
        return pd.DataFrame(), []

    df_features = df[cols_to_select].copy()

    # กรองแถวที่ค่าส่วนใหญ่เป็น NaN ออก (ใช้ FEATURES_FOR_CLUSTERING เป็นเกณฑ์)
    df_features.dropna(thresh=len(FEATURES_FOR_CLUSTERING) * 0.6, subset=FEATURES_FOR_CLUSTERING, inplace=True)

    if df_features.empty:
        logging.warning("No rows remaining after initial NaN drop.")
        return pd.DataFrame(), []

    tickers = df_features['ticker']
    numeric_features_base = df_features[FEATURES_FOR_CLUSTERING] # Features หลักจาก DB

    # 2. คำนวณ Log Market Cap (ถ้ามี)
    final_feature_list = FEATURES_FOR_CLUSTERING[:] # Copy list ไว้ก่อน
    if has_market_cap:
        # ใช้ np.log1p เพื่อจัดการค่า market_cap = 0 (ถ้ามี) ให้กลายเป็น log(1)=0 แทนที่จะเป็น -inf
        log_market_cap = np.log1p(df_features['market_cap'].clip(lower=0))
        # ตรวจสอบค่า inf หรือ NaN ที่อาจเกิดจากการคำนวณ log
        log_market_cap.replace([np.inf, -np.inf], np.nan, inplace=True)
        # รวม log_market_cap เข้ากับ features หลัก (สำคัญ: ต้อง reset_index ก่อนเพื่อให้ index ตรงกัน)
        numeric_features_full = pd.concat(
            [numeric_features_base.reset_index(drop=True),
             log_market_cap.rename('log_market_cap').reset_index(drop=True)],
            axis=1
        )
        final_feature_list.append('log_market_cap') # เพิ่มชื่อ feature ใหม่
    else:
        numeric_features_full = numeric_features_base
        # final_feature_list ไม่ต้องแก้

    # 3. จัดการค่าที่ "สุดโต่ง" (Outliers) - ใช้ clip กับ numeric_features_full
    for col in final_feature_list: # วนลูปตาม features สุดท้ายทั้งหมด
         if col in numeric_features_full.columns: # เช็คเผื่อกรณีไม่มี market_cap
            q_low = numeric_features_full[col].quantile(0.01)
            q_high = numeric_features_full[col].quantile(0.99)
            # ใช้ .loc เพื่อหลีกเลี่ยง SettingWithCopyWarning
            numeric_features_full.loc[:, col] = numeric_features_full[col].clip(lower=q_low, upper=q_high)

    # 4. เติมค่าว่าง (Impute Missing Values) - ใช้ Median กับ numeric_features_full
    imputer = SimpleImputer(strategy='median')
    # ต้องแน่ใจว่า input ไม่มีค่า inf อีก (clip ควรจัดการแล้ว แต่เช็คอีกรอบ)
    numeric_features_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_features_imputed = imputer.fit_transform(numeric_features_full)
    df_imputed = pd.DataFrame(numeric_features_imputed, columns=final_feature_list, index=numeric_features_full.index)

    # 5. ปรับสเกล (Scale Data) กับ df_imputed
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_imputed)
    df_scaled = pd.DataFrame(scaled_features, columns=final_feature_list, index=df_imputed.index)

    # 6. รวม ticker กลับเข้าไป (ใช้ index เดิม)
    # ต้อง reset_index ของ tickers ด้วยเพื่อให้ตรงกับ df_scaled
    df_processed = pd.concat([tickers.reset_index(drop=True), df_scaled], axis=1)

    logging.info(f"Data preprocessing complete. {len(df_processed)} stocks remaining. Using features: {final_feature_list}")
    # คืนค่า DataFrame ที่ Process แล้ว และ List ของ Features ที่ใช้จริง
    return df_processed, final_feature_list # <<< แก้ไข

# --- [ปรับปรุง] ฟังก์ชันหา k ที่เหมาะสม (รับ features ที่ใช้จริง) ---
def find_optimal_k(df_scaled: pd.DataFrame, actual_features: list) -> int: # <<< รับ actual_features
    """หาจำนวน Cluster (k) ที่เหมาะสมโดยใช้ Elbow Method"""
    logging.info("Finding optimal k using Elbow Method...")
    if df_scaled.empty or len(df_scaled) < max(K_RANGE) or not actual_features: # <<< เช็ค actual_features
        logging.warning("Not enough data or features to reliably determine optimal k. Defaulting to MIN_K.")
        return MIN_K

    data_for_elbow = df_scaled[actual_features] # <<< ใช้ actual_features
    inertia_values = []

    for k in K_RANGE:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(data_for_elbow)
        inertia_values.append(kmeans.inertia_)

    try:
        kl = KneeLocator(list(K_RANGE), inertia_values, curve='convex', direction='decreasing')
        optimal_k = kl.elbow
        if optimal_k is None:
            logging.warning("Could not automatically detect elbow point. Defaulting to MIN_K.")
            optimal_k = MIN_K
        else:
            logging.info(f"Optimal k detected by Elbow Method: {optimal_k}")
    except Exception as e:
        logging.warning(f"Error during KneeLocator: {e}. Defaulting to MIN_K.")
        optimal_k = MIN_K

    return optimal_k

# --- [ปรับปรุง] ฟังก์ชันจัดกลุ่ม (รับ features ที่ใช้จริง) ---
def perform_clustering(df_scaled: pd.DataFrame, n_clusters: int, actual_features: list) -> pd.DataFrame: # <<< รับ actual_features
    """ใช้ K-Means จัดกลุ่มข้อมูลด้วยจำนวน Cluster และ Features ที่ระบุ"""
    logging.info(f"Performing K-Means clustering with k={n_clusters}...")
    if df_scaled.empty or len(df_scaled) < n_clusters or not actual_features: # <<< เช็ค actual_features
        logging.warning("Not enough data or features for clustering. Skipping.")
        return pd.DataFrame()

    data_for_clustering = df_scaled[actual_features] # <<< ใช้ actual_features
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(data_for_clustering)
    cluster_labels = kmeans.labels_

    df_results = pd.DataFrame({
        'ticker': df_scaled['ticker'],
        'peer_cluster_id': cluster_labels
    })
    logging.info("K-Means clustering complete.")
    return df_results

# --- ฟังก์ชันตรวจสอบความสมเหตุสมผล (เหมือนเดิม) ---
def is_clustering_reasonable(df_results: pd.DataFrame, total_stocks: int) -> bool:
    """ตรวจสอบว่าผลลัพธ์การจัดกลุ่มสมเหตุสมผลหรือไม่"""
    if df_results.empty: return False
    cluster_counts = df_results['peer_cluster_id'].value_counts()
    logging.info(f"Checking reasonableness of clustering results (k={len(cluster_counts)}):")
    logging.info(f"Cluster distribution:\n{cluster_counts.sort_index()}")
    min_size = cluster_counts.min()
    if min_size < MIN_CLUSTER_SIZE:
        logging.warning(f"Sanity Check FAILED: Smallest cluster size ({min_size}) < minimum required ({MIN_CLUSTER_SIZE}).")
        return False
    max_size = cluster_counts.max()
    max_ratio = max_size / total_stocks
    if max_ratio > MAX_CLUSTER_SIZE_RATIO:
        logging.warning(f"Sanity Check FAILED: Largest cluster size ({max_size}, {max_ratio:.1%}) > maximum ratio ({MAX_CLUSTER_SIZE_RATIO:.1%}).")
        return False
    logging.info("Sanity Check PASSED: Cluster sizes are reasonable.")
    return True

# --- ฟังก์ชันบันทึกผล (เหมือนเดิม) ---
def update_cluster_ids_in_db(df_results: pd.DataFrame):
    """อัปเดตค่า peer_cluster_id กลับลงฐานข้อมูล"""
    if df_results.empty:
        logging.warning("No clustering results to update in the database.")
        return
    logging.info(f"Updating {len(df_results)} cluster IDs in the database...")
    update_count = 0; error_count = 0
    try:
        with server.app_context():
            latest_date = db.session.query(func.max(FactCompanySummary.date_updated)).scalar()
            if not latest_date: logging.error("Cannot determine latest date for update."); return

            logging.info(f"Clearing previous cluster IDs for date {latest_date}...")
            clear_stmt = update(FactCompanySummary).where(FactCompanySummary.date_updated == latest_date).values(peer_cluster_id=None)
            db.session.execute(clear_stmt)

            for index, row in df_results.iterrows():
                try:
                    stmt = update(FactCompanySummary).where(FactCompanySummary.ticker == row['ticker']).where(FactCompanySummary.date_updated == latest_date).values(peer_cluster_id=int(row['peer_cluster_id']))
                    db.session.execute(stmt)
                    update_count += 1
                except Exception as e_row:
                    logging.error(f"Error updating cluster ID for {row['ticker']}: {e_row}")
                    error_count += 1; db.session.rollback(); continue

            if error_count == 0:
                db.session.commit()
                logging.info(f"Successfully updated {update_count} cluster IDs.")
            else:
                logging.warning(f"Finished updating with {error_count} errors. Rolling back all updates for this run.")
                db.session.rollback()
    except Exception as e:
        logging.error(f"Critical error during database update: {e}", exc_info=True)
        with server.app_context(): db.session.rollback()

# --- [ปรับปรุง] ส่วนหลัก (Main Execution) ---
if __name__ == "__main__":
    logging.info("Starting ML Clustering Script (Hybrid Approach + New Features)...")

    # 1. ดึงข้อมูล
    df_raw = get_latest_financial_data()

    if not df_raw.empty:
        # 2. เตรียมข้อมูล (รับ list features ที่ใช้จริงกลับมาด้วย)
        df_processed, actual_features_used = preprocess_data(df_raw) # <<< แก้ไข

        # --- [ปรับปรุง] เช็คว่ามี features เหลือให้ใช้หรือไม่ ---
        if not df_processed.empty and actual_features_used:
            total_stocks_processed = len(df_processed)

            # 3. หา k ที่เหมาะสมทางสถิติ (ส่ง features ที่ใช้จริงไปด้วย)
            current_k = find_optimal_k(df_processed, actual_features_used) # <<< แก้ไข

            # 4. วนลูปหา k ที่สมเหตุสมผล
            final_results = pd.DataFrame()
            while current_k >= MIN_K:
                # 4.1 จัดกลุ่มด้วย k ปัจจุบัน (ส่ง features ที่ใช้จริงไปด้วย)
                df_cluster_results = perform_clustering(df_processed, current_k, actual_features_used) # <<< แก้ไข

                if df_cluster_results.empty:
                    logging.warning(f"Clustering failed for k={current_k}. Trying k={current_k-1}...")
                    current_k -= 1
                    continue

                # 4.2 ตรวจสอบความสมเหตุสมผล (เหมือนเดิม)
                if is_clustering_reasonable(df_cluster_results, total_stocks_processed):
                    logging.info(f"Found reasonable clustering with k={current_k}.")
                    final_results = df_cluster_results
                    break
                else:
                    logging.warning(f"Clustering with k={current_k} was not reasonable. Trying k={current_k-1}...")
                    current_k -= 1
            else:
                logging.error(f"Could not find a reasonable k value even down to {MIN_K}. Clustering failed.")

            # 5. บันทึกผลลัพธ์สุดท้าย (เหมือนเดิม)
            if not final_results.empty:
                update_cluster_ids_in_db(final_results)
            else:
                logging.warning("No final clustering results to update in the database.")
        # --- [จบการปรับปรุง] ---
        elif df_processed.empty:
             logging.warning("Data preprocessing resulted in empty data.")
        else: # กรณี actual_features_used เป็น list ว่าง
             logging.error("No valid features remaining after preprocessing. Clustering cannot proceed.")
    else:
        logging.warning("Failed to retrieve data from database.")

    logging.info("ML Clustering Script finished.")
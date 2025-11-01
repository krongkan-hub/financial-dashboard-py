# run_ml_cluster.py (เวอร์ชันปรับปรุง Logic การหา k)

import sys
import os
# หาที่อยู่ของโฟลเดอร์ปัจจุบัน (scripts/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# หาที่อยู่ของโฟลเดอร์แม่ (project_dash/)
project_root = os.path.dirname(script_dir)
# เพิ่มโฟลเดอร์แม่เข้าไปใน sys.path
sys.path.append(project_root)
import logging
import pandas as pd
import numpy as np
from datetime import date

# --- Machine Learning ---
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from kneed import KneeLocator # ใช้หา "ข้อศอก" อัตโนมัติ

# --- Database Interaction ---
from app import db, server
from app.models import FactCompanySummary
from sqlalchemy import func, update

# ตั้งค่า Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- [ปรับปรุง] Configuration ---
K_RANGE = range(10, 40) # <--- ปรับช่วง k เป็น 5 ถึง 30
MIN_K = min(K_RANGE)
MAX_K = max(K_RANGE)
MAX_ITERATIONS = 100 # <--- เพิ่ม: จำกัดจำนวนครั้งในการวน Loop หา k

# --- คอลัมน์ (Features) ที่จะใช้ (เหมือนเดิม) ---
FEATURES_FOR_CLUSTERING = [
    'pe_ratio', 'pb_ratio', 'ev_ebitda', # Valuation
    'revenue_growth_yoy', 'net_income_growth_yoy', # Growth
    'roe', 'operating_margin', 'ebitda_margin', # Profitability
    'de_ratio', # Leverage
    'beta', # Risk
    'cash_conversion', # Quality
]

# --- Sanity Check Configuration (เหมือนเดิม) ---
MIN_CLUSTER_SIZE = 10         # กลุ่มต้องมีหุ้นอย่างน้อยกี่ตัว
MAX_CLUSTER_SIZE_RATIO = 0.35 # ลดเกณฑ์ลงเหลือ 35%

# --- ฟังก์ชันดึงข้อมูล (เหมือนเดิม) ---
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

# --- ฟังก์ชันเตรียมข้อมูล (เหมือนเดิม) ---
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
        log_market_cap = np.log1p(df_features['market_cap'].clip(lower=0))
        log_market_cap.replace([np.inf, -np.inf], np.nan, inplace=True)
        numeric_features_full = pd.concat(
            [numeric_features_base.reset_index(drop=True),
             log_market_cap.rename('log_market_cap').reset_index(drop=True)],
            axis=1
        )
        final_feature_list.append('log_market_cap')
    else:
        numeric_features_full = numeric_features_base

    # 3. จัดการค่าที่ "สุดโต่ง" (Outliers)
    for col in final_feature_list:
         if col in numeric_features_full.columns:
            q_low = numeric_features_full[col].quantile(0.01)
            q_high = numeric_features_full[col].quantile(0.99)
            numeric_features_full.loc[:, col] = numeric_features_full[col].clip(lower=q_low, upper=q_high)

    # 4. เติมค่าว่าง (Impute Missing Values)
    imputer = SimpleImputer(strategy='median')
    numeric_features_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_features_imputed = imputer.fit_transform(numeric_features_full)
    df_imputed = pd.DataFrame(numeric_features_imputed, columns=final_feature_list, index=numeric_features_full.index)

    # 5. ปรับสเกล (Scale Data)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_imputed)
    df_scaled = pd.DataFrame(scaled_features, columns=final_feature_list, index=df_imputed.index)

    # 6. รวม ticker กลับเข้าไป
    df_processed = pd.concat([tickers.reset_index(drop=True), df_scaled], axis=1)

    logging.info(f"Data preprocessing complete. {len(df_processed)} stocks remaining. Using features: {final_feature_list}")
    return df_processed, final_feature_list

# --- ฟังก์ชันหา k ที่เหมาะสม (เหมือนเดิม) ---
def find_optimal_k(df_scaled: pd.DataFrame, actual_features: list) -> int:
    """หาจำนวน Cluster (k) ที่เหมาะสมโดยใช้ Elbow Method"""
    logging.info("Finding optimal k using Elbow Method...")
    if df_scaled.empty or len(df_scaled) < max(K_RANGE) or not actual_features:
        logging.warning(f"Not enough data or features to reliably determine optimal k. Defaulting to MIN_K={MIN_K}.")
        return MIN_K

    # --- [ปรับปรุง] ใช้ช่วง K_RANGE ใหม่ (5-30) ---
    k_values_to_test = list(K_RANGE)
    # Ensure we don't test more k than available data points
    if len(df_scaled) < max(k_values_to_test):
        k_values_to_test = list(range(MIN_K, len(df_scaled)))
        if not k_values_to_test:
            logging.warning("Less data points than MIN_K. Defaulting to MIN_K.")
            return MIN_K
        logging.warning(f"Number of data points ({len(df_scaled)}) is less than MAX_K ({MAX_K}). Testing k up to {k_values_to_test[-1]}.")

    data_for_elbow = df_scaled[actual_features]
    inertia_values = []

    for k in k_values_to_test:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(data_for_elbow)
            inertia_values.append(kmeans.inertia_)
        except Exception as e:
            logging.error(f"Error during KMeans fit for k={k}: {e}. Stopping Elbow Method early.")
            # If error occurs, only use inertia values calculated so far
            k_values_to_test = k_values_to_test[:len(inertia_values)]
            break # Exit the loop early

    if len(inertia_values) < 2: # Need at least 2 points to find elbow
         logging.warning(f"Could not calculate inertia for enough k values. Defaulting to MIN_K={MIN_K}.")
         return MIN_K

    try:
        # --- [ปรับปรุง] ใช้ k_values_to_test ที่อาจสั้นลง ---
        kl = KneeLocator(k_values_to_test, inertia_values, curve='convex', direction='decreasing')
        optimal_k = kl.elbow
        if optimal_k is None:
            logging.warning(f"Could not automatically detect elbow point. Defaulting to MIN_K={MIN_K}.")
            optimal_k = MIN_K
        else:
            # --- [ปรับปรุง] Clamp ค่า k ให้อยู่ในช่วง MIN_K ถึง MAX_K ---
            optimal_k = max(MIN_K, min(MAX_K, optimal_k))
            logging.info(f"Optimal k detected by Elbow Method (Clamped): {optimal_k}")
    except Exception as e:
        logging.warning(f"Error during KneeLocator: {e}. Defaulting to MIN_K={MIN_K}.")
        optimal_k = MIN_K

    return optimal_k

# --- ฟังก์ชันจัดกลุ่ม (เหมือนเดิม) ---
def perform_clustering(df_scaled: pd.DataFrame, n_clusters: int, actual_features: list) -> pd.DataFrame:
    """ใช้ K-Means จัดกลุ่มข้อมูลด้วยจำนวน Cluster และ Features ที่ระบุ"""
    logging.info(f"Performing K-Means clustering with k={n_clusters}...")
    if df_scaled.empty or len(df_scaled) < n_clusters or not actual_features:
        logging.warning("Not enough data or features for clustering. Skipping.")
        return pd.DataFrame()

    data_for_clustering = df_scaled[actual_features]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(data_for_clustering)
    cluster_labels = kmeans.labels_

    df_results = pd.DataFrame({
        'ticker': df_scaled['ticker'],
        'peer_cluster_id': cluster_labels
    })
    logging.info("K-Means clustering complete.")
    return df_results

# --- [ปรับปรุง] ฟังก์ชันตรวจสอบความสมเหตุสมผล (คืนค่า Flag แยก) ---
def check_clustering_reasonableness(df_results: pd.DataFrame, total_stocks: int) -> (bool, bool, bool):
    """
    ตรวจสอบว่าผลลัพธ์การจัดกลุ่มสมเหตุสมผลหรือไม่
    Returns: (passes_all, fails_min_size, fails_max_size)
    """
    if df_results.empty: return False, False, False

    cluster_counts = df_results['peer_cluster_id'].value_counts()
    k = len(cluster_counts)
    logging.info(f"Checking reasonableness of clustering results (k={k}):")
    logging.info(f"Cluster distribution:\n{cluster_counts.sort_index()}")

    fails_min_size = False
    min_size = cluster_counts.min()
    if min_size < MIN_CLUSTER_SIZE:
        logging.warning(f"Sanity Check FAILED: Smallest cluster size ({min_size}) < minimum required ({MIN_CLUSTER_SIZE}).")
        fails_min_size = True

    fails_max_size = False
    max_size = cluster_counts.max()
    max_ratio = max_size / total_stocks
    if max_ratio > MAX_CLUSTER_SIZE_RATIO:
        logging.warning(f"Sanity Check FAILED: Largest cluster size ({max_size}, {max_ratio:.1%}) > maximum ratio ({MAX_CLUSTER_SIZE_RATIO:.1%}).")
        fails_max_size = True

    passes_all = not fails_min_size and not fails_max_size
    if passes_all:
        logging.info("Sanity Check PASSED: Cluster sizes are reasonable.")

    return passes_all, fails_min_size, fails_max_size

# --- ฟังก์ชันบันทึกผล (เหมือนเดิม) ---
def update_cluster_ids_in_db(df_results: pd.DataFrame):
    """อัปเดตค่า peer_cluster_id กลับลงฐานข้อมูล โดยอัปเดตแถวที่มี date_updated ล่าสุดสำหรับแต่ละ Ticker"""
    if df_results.empty:
        logging.warning("No clustering results to update in the database.")
        return
    
    # 1. เตรียมข้อมูลที่จะอัปเดตเป็น Dict of Dicts (Ticker -> Cluster ID)
    # ใช้ .apply(int) เพื่อให้แน่ใจว่าเป็น Integer (จาก numpy.int64)
    ticker_to_cluster = df_results.set_index('ticker')['peer_cluster_id'].apply(int).to_dict()
    
    logging.info(f"Updating {len(ticker_to_cluster)} cluster IDs in the database...")
    update_count = 0
    
    try:
        with server.app_context():
            
            # 2. สร้าง Subquery เพื่อหาแถวที่ล่าสุด (MAX(date_updated)) สำหรับ Ticker ที่ต้องการอัปเดต
            # Query หา Max Date สำหรับ Tickers ที่มีผลลัพธ์ Clustering
            latest_date_subquery = db.session.query(
                FactCompanySummary.ticker,
                func.max(FactCompanySummary.date_updated).label('max_date')
            ).filter(
                FactCompanySummary.ticker.in_(list(ticker_to_cluster.keys()))
            ).group_by(FactCompanySummary.ticker).subquery()
            
            # ดึง Max Date สำหรับ Tickers ทั้งหมดในครั้งเดียว
            max_dates_result = db.session.query(
                latest_date_subquery.c.ticker,
                latest_date_subquery.c.max_date
            ).all()
            
            ticker_to_max_date = {t: d for t, d in max_dates_result}

            # 3. วนลูปและรัน UPDATE โดยเจาะจง Ticker และ MAX(date_updated) ของ Ticker นั้น
            for ticker, cluster_id in ticker_to_cluster.items():
                latest_date = ticker_to_max_date.get(ticker)
                
                if latest_date is None:
                    logging.warning(f"Could not find latest update date for {ticker}. Skipping update.")
                    continue
                
                # สร้างคำสั่ง UPDATE โดยเจาะจงที่ Ticker และ MAX(date_updated) ของ Ticker นั้น
                stmt = update(FactCompanySummary).where(
                    FactCompanySummary.ticker == ticker
                ).where(
                    FactCompanySummary.date_updated == latest_date
                ).values(
                    peer_cluster_id=cluster_id
                )
                db.session.execute(stmt)
                update_count += 1
            
            # 4. Commit ครั้งเดียว เพื่อให้ Transaction รวดเร็วที่สุด
            db.session.commit()
            logging.info(f"Successfully updated {update_count} cluster IDs on their respective latest dates.")
            
    except Exception as e:
        logging.error(f"Critical error during database update: {e}", exc_info=True)
        with server.app_context(): 
            db.session.rollback()

# --- [ปรับปรุง] ส่วนหลัก (Main Execution) ---
if __name__ == "__main__":
    logging.info("Starting ML Clustering Script (Hybrid Approach + New Features)...")

    # 1. ดึงข้อมูล
    df_raw = get_latest_financial_data()

    if not df_raw.empty:
        # 2. เตรียมข้อมูล
        df_processed, actual_features_used = preprocess_data(df_raw)

        if not df_processed.empty and actual_features_used:
            total_stocks_processed = len(df_processed)

            # 3. หา k เริ่มต้นทางสถิติ
            k_start = find_optimal_k(df_processed, actual_features_used)
            # --- [เพิ่ม] ตรวจสอบ k_start ให้อยู่ในช่วง 5-30 ---
            k_start = max(MIN_K, min(MAX_K, k_start))
            logging.info(f"Starting k search from k_start = {k_start}")

            # 4. --- [ปรับปรุง] วนลูปค้นหา k ที่เหมาะสม ---
            current_k = k_start
            tried_k = set()
            iteration = 0
            found_k = None
            final_results = pd.DataFrame()

            while iteration < MAX_ITERATIONS:
                # --- เงื่อนไขการหยุด Loop ---
                if current_k < MIN_K or current_k > MAX_K:
                    logging.warning(f"k={current_k} is outside the allowed range ({MIN_K}-{MAX_K}). Stopping search.")
                    break
                if current_k in tried_k:
                    logging.warning(f"Already tried k={current_k}. Stopping search to prevent infinite loop.")
                    break

                logging.info(f"--- Iteration {iteration + 1}/{MAX_ITERATIONS}: Trying k={current_k} ---")
                tried_k.add(current_k)

                # 4.1 จัดกลุ่มด้วย k ปัจจุบัน
                df_cluster_results = perform_clustering(df_processed, current_k, actual_features_used)

                if df_cluster_results.empty:
                    logging.warning(f"Clustering failed unexpectedly for k={current_k}.")
                    # ลองลด k ถ้าเกิด error (อาจจะลอง strategy อื่นก็ได้)
                    next_k = current_k - 1
                else:
                    # 4.2 ตรวจสอบความสมเหตุสมผล
                    passes_all, fails_min_size, fails_max_size = check_clustering_reasonableness(
                        df_cluster_results, total_stocks_processed
                    )

                    # 4.3 ตัดสินใจ k ต่อไปตาม Logic ใหม่
                    if passes_all:
                        logging.info(f"Found reasonable clustering with k={current_k}.")
                        found_k = current_k
                        final_results = df_cluster_results
                        break # เจอแล้ว หยุด Loop
                    elif fails_min_size: # ให้ความสำคัญกับ Min Size ก่อน (ไม่ว่า Max Size จะ Fail ด้วยหรือไม่)
                        logging.warning(f"k={current_k} failed MIN size check. Trying k={current_k - 1}...")
                        next_k = current_k - 1
                    elif fails_max_size: # กรณี Fail เฉพาะ Max Size
                        logging.warning(f"k={current_k} failed MAX size check. Trying k={current_k + 1}...")
                        next_k = current_k + 1
                    else:
                         # Should not happen if logic is correct
                         logging.error(f"Unexpected state for k={current_k}. Stopping search.")
                         break

                current_k = next_k
                iteration += 1
            # --- จบ Loop ค้นหา k ---

            # 5. สรุปผลและบันทึก
            if found_k is not None and not final_results.empty:
                logging.info(f"Successfully found a reasonable k = {found_k}. Updating database.")
                update_cluster_ids_in_db(final_results)
            else:
                if iteration >= MAX_ITERATIONS:
                    logging.error(f"Could not find a reasonable k value within {MAX_ITERATIONS} iterations. Clustering failed.")
                else:
                    logging.error(f"Stopped searching for k prematurely (k={current_k}). Clustering failed.")
                logging.warning("No final clustering results to update in the database.")

        elif df_processed.empty:
             logging.warning("Data preprocessing resulted in empty data.")
        else:
             logging.error("No valid features remaining after preprocessing. Clustering cannot proceed.")
    else:
        logging.warning("Failed to retrieve data from database.")

    logging.info("ML Clustering Script finished.")
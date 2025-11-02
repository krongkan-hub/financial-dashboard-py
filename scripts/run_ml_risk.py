# scripts/run_ml_risk.py
# (ดัดแปลงจาก run_ml_risk.py เพื่อใช้โมเดลอนุกรมเวลา LSTM/Keras)
# (FIXED: 2025-11-02 - แก้ไข ValueError: Shape mismatch (42 vs 43))

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import os
import joblib 

# --- [NEU] Import สำหรับ Deep Learning ---
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import class_weight
# --- [END NEU] ---

from app.data_handler import get_ml_risk_raw_data
from app import db, server
from app.models import FactFinancialStatements, FactCompanySummary

# --- [NEU] Import Logic การสร้าง Feature ดิบ ---
# เราจะใช้ Logic นี้ในการสร้าง Feature ดิบ 22 ตัวสำหรับ *ทุกไตรมาส*
try:
    from app.ml.ml_risk_features import engineer_features_for_prediction, ML_RISK_BASE_FEATURES
except ImportError:
    print("FATAL: ไม่พบไฟล์ ml_risk_features.py กรุณาสร้างไฟล์ตามคำแนะนำ")
    sys.exit(1)

# --- Imports จาก run_ml_risk.py เดิม (ที่ยังต้องใช้) ---
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve
)
# --- (ลบ XGBClassifier, RandomizedSearchCV, SHAP ออก) ---

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- [NEU] LSTM Configuration ---
SEQUENCE_LENGTH = 12 # ดูข้อมูลงบการเงินย้อนหลัง 12 ไตรมาส (3 ปี)
NUM_BASE_FEATURES = len(ML_RISK_BASE_FEATURES) # = 22 features
# --- [END NEU] ---

# Constants for Target Variable Definition (เหมือนเดิม)
DE_RATIO_THRESHOLD = 7.0 
ICR_THRESHOLD = 1.0
CONSECUTIVE_LOSS_PERIODS = 8 
RATING_MAP = {
    'AAA': 1, 'AA+': 2, 'AA': 3, 'AA-': 4, 'A+': 5, 'A': 6, 'A-': 7,
    'BBB+': 8, 'BBB': 9, 'BBB-': 10, 'BB+': 11, 'BB': 12, 'BB-': 13,
    'B+': 14, 'B': 15, 'B-': 16, 'CCC+': 17, 'CCC': 18, 'CCC-': 19,
    'CC': 20, 'C': 21, 'D': 22, 'NR': 99, 'N/A': 99, np.nan: 99, 'None': 99, '': 99
}

# Model Saving Configuration (เปลี่ยนชื่อไฟล์โมเดล)
MODEL_DIR = "models"
MODEL_FILENAME = "trained_risk_model_lstm.keras" # <-- [NEU] ใช้ .keras
SCALER_FILENAME = "scaler_lstm.joblib" # <-- [NEU] สร้าง Scaler ใหม่
IMPUTER_FILENAME = "imputer_lstm.joblib" # <-- [NEU] สร้าง Imputer ใหม่

os.makedirs(MODEL_DIR, exist_ok=True)

# --- Helper Functions (จาก run_ml_risk.py) ---
def get_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None 

# --- [UNCHANGED] ฟังก์ชันสร้าง Y (Target Variable) ---
# ฟังก์ชันนี้ยังคงทำงานได้ดีเหมือนเดิม
def create_target_variable(df_ffs, df_fcs):
    """Creates the binary target variable (Y) based on defined V2.0 risk criteria..."""
    logging.info("2. Creating Target Variable (Y) using V2.0 Solvency and Rating Downgrade criteria...")
    if df_ffs.empty:
        logging.warning("Financial statement data is empty, cannot create target variable.")
        return pd.DataFrame()

    df = df_ffs.copy()
    df = df.sort_values(by=['ticker', 'report_date']).reset_index(drop=True)

    # --- V2.0 Target 1: Y_Rating (Downgrade in next 4 quarters) ---
    if 'credit_rating' in df.columns:
        df['rating_score'] = df['credit_rating'].astype(str).str.upper().map(RATING_MAP).fillna(99)
        df['future_rating_score'] = df.groupby('ticker')['rating_score'].shift(-4)
        rule_rating_downgrade = (df['future_rating_score'] > df['rating_score']) & (df['rating_score'] < 99)
        df['Y_Rating_raw'] = rule_rating_downgrade.astype(int)
    else:
        logging.warning("Missing 'credit_rating' column. Y_Rating cannot be calculated.")
        df['Y_Rating_raw'] = 0

    # --- V2.0 Target 2: Y_Solvency (Accumulated Deficit > 0 AND Current Ratio < 1.0) ---
    accumulated_deficit_col = get_column(df, ['Accumulated Deficit', 'Accumulated_Deficit'])
    current_ratio_col = get_column(df, ['Current Ratio', 'Current_Ratio'])
    
    if accumulated_deficit_col and current_ratio_col:
        rule_solvency = (df[accumulated_deficit_col] > 0) & (df[current_ratio_col] < 1.0)
        df['Y_Solvency_raw'] = rule_solvency.astype(int)
    else:
        logging.warning("Missing 'Accumulated Deficit' or 'Current Ratio' columns. Y_Solvency cannot be calculated.")
        df['Y_Solvency_raw'] = 0

    # --- Final Target (Y) using OR logic ---
    df['Y_raw_v2'] = ((df['Y_Rating_raw'] == 1) | (df['Y_Solvency_raw'] == 1)).astype(int)

    lag_periods = 4
    df['Y'] = df.groupby('ticker')['Y_raw_v2'].shift(-lag_periods) 
    logging.info(f"Target variable Y (V2.0) created with a lag of {lag_periods} quarters.")

    df_final = df.dropna(subset=['Y']).copy()
    df_final['Y'] = df_final['Y'].astype(int)

    logging.info(f"Created Y for {len(df_final)} data points.")
    logging.info(f"Distribution of Y: \n{df_final['Y'].value_counts(normalize=True)}")

    # ส่งคืน DataFrame ทั้งหมดที่มี Y เพื่อใช้ในการสร้าง Features ต่อไป
    return df_final


# --- [MODIFIED] ฟังก์ชันสร้าง Feature (X) ---
def engineer_features_and_sequences(df_with_y):
    """
    สร้าง 22 Base Features, Impute/Scale, 
    และสร้างข้อมูลอนุกรมเวลา (Sequences) สำหรับ LSTM
    """
    logging.info("3. Engineering Features (X) for Sequential Model...")
    if df_with_y.empty or 'Y' not in df_with_y.columns:
        logging.warning("Input DataFrame is empty or missing Y column.")
        return np.array([]), np.array([]), pd.DataFrame(), []

    # 1. สร้าง 22 Raw Features สำหรับ *ทุกไตรมาส*
    # เราใช้ df_with_y ทั้งหมด ซึ่งมีข้อมูลดิบทางการเงินที่ดึงมาจาก data_handler
    #
    # **สำคัญ**: `engineer_features_for_prediction` คืนค่า (X_features, tickers_list)
    # แต่เราต้องการ index เดิมเพื่อ join กลับเข้าไป
    df_with_y_indexed = df_with_y.set_index(['ticker', 'report_date'])
    X_raw_features_all_quarters, _ = engineer_features_for_prediction(df_with_y)
    
    # นำ index เดิมกลับมา
    X_raw_features_all_quarters.index = df_with_y.index
    
    # 2. Impute และ Scale ข้อมูล
    # **สำคัญ**: เราต้องสร้าง Imputer/Scaler *ใหม่*
    
    X_to_impute = X_raw_features_all_quarters[ML_RISK_BASE_FEATURES].copy()
    y = df_with_y['Y'].copy()
    ids = df_with_y[['ticker', 'report_date']].copy()
    
    # --- Imputation (สร้าง Imputer ใหม่) ---
    
    # --- [FIX for ValueError (2025-11-02)] ---
    # (โค้ด 5 บรรทัดนี้คัดลอกจาก run_ml_risk.py เดิม)
    all_nan_cols = X_to_impute.columns[X_to_impute.isnull().all()].tolist()
    if all_nan_cols:
        logging.warning(f"Forcing imputation for all-NaN features by filling with 0: {all_nan_cols}")
        X_to_impute[all_nan_cols] = X_to_impute[all_nan_cols].fillna(0)  
    # --- [END FIX] ---

    imputer = SimpleImputer(strategy='median', add_indicator=True)
    X_imputed_array = imputer.fit_transform(X_to_impute)
    
    # ดึงชื่อ Features ที่สร้างขึ้น (22 base + N indicators)
    feature_names_out = list(ML_RISK_BASE_FEATURES)
    missing_features_indices = imputer.indicator_.features_
    indicator_feature_names = [f'Missing_{feature_names_out[i]}' for i in missing_features_indices]
    final_feature_names = feature_names_out + indicator_feature_names
    
    X_imputed_df = pd.DataFrame(X_imputed_array, columns=final_feature_names, index=X_to_impute.index)
    
    # --- Scaling (สร้าง Scaler ใหม่) ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=final_feature_names, index=X_imputed_df.index)
    
    logging.info(f"Engineered {len(final_feature_names)} features (Base + Indicators) for {len(X_scaled_df)} quarters.")
    
    # --- บันทึก Artifacts ใหม่ ---
    joblib.dump(imputer, os.path.join(MODEL_DIR, IMPUTER_FILENAME))
    joblib.dump(scaler, os.path.join(MODEL_DIR, SCALER_FILENAME))
    logging.info(f"New LSTM Imputer saved to {os.path.join(MODEL_DIR, IMPUTER_FILENAME)}")
    logging.info(f"New LSTM Scaler saved to {os.path.join(MODEL_DIR, SCALER_FILENAME)}")

    # 3. สร้าง Sequences (X_seq, y_seq)
    logging.info(f"Creating sequences with max length {SEQUENCE_LENGTH}...")
    data_for_seq = pd.concat([ids, X_scaled_df, y], axis=1)
    
    X_list = [] # List of sequences (each sequence is a numpy array)
    y_list = [] # List of labels
    ids_list = [] # List of (ticker, report_date) for each label
    
    grouped = data_for_seq.groupby('ticker')
    
    for ticker, group in grouped:
        features = group[final_feature_names].values
        labels = group['Y'].values
        dates = group['report_date'].values
        
        # วนลูปทุกจุดข้อมูลใน group
        for i in range(len(group)):
            # จุดเริ่มต้นของ sequence (ย้อนหลังไม่เกิน SEQUENCE_LENGTH และไม่ต่ำกว่า 0)
            start_idx = max(0, i - SEQUENCE_LENGTH + 1)
            # ดึง sequence (อาจจะสั้นกว่า SEQUENCE_LENGTH ในช่วงแรกๆ)
            seq = features[start_idx : i + 1] 
            
            X_list.append(seq)
            y_list.append(labels[i])
            ids_list.append((ticker, dates[i]))

    if not X_list:
        logging.error("No sequences could be created.")
        return np.array([]), np.array([]), pd.DataFrame(), []

    # --- [NEU] Padding ---
    # `pad_sequences` จะเติม 0 (padding) ด้านหน้า (pre)
    # ให้ทุก sequence มีความยาวเท่ากับ SEQUENCE_LENGTH
    logging.info(f"Padding {len(X_list)} sequences...")
    X_padded = pad_sequences(X_list, maxlen=SEQUENCE_LENGTH, dtype='float32', padding='pre', truncating='pre')
    
    y_seq = np.array(y_list)
    ids_seq = pd.DataFrame(ids_list, columns=['ticker', 'report_date'])
    
    logging.info(f"Sequence creation complete. X shape: {X_padded.shape}, y shape: {y_seq.shape}")
    
    return X_padded, y_seq, ids_seq, final_feature_names

# --- [UNCHANGED] ฟังก์ชันแบ่งข้อมูล ---
# ฟังก์ชันนี้ใช้ได้เลย เพราะเราส่ง ids_seq ที่มี 'report_date' ไปให้
def split_data(X_seq, y_seq, ids_seq, feature_names):
    """Splits sequenced data into train, validation, and test sets based on time."""
    logging.info("4. Splitting sequenced data into Train, Validation, Test sets (Time-Based)...")
    
    TEST_RATIO = 0.20  
    VAL_RATIO = 0.10  
    MIN_TRAIN_SAMPLES = 50 

    # สร้าง DataFrame ชั่วคราวเพื่อ sort ตามเวลา
    ids_seq['y_temp'] = y_seq
    ids_seq = ids_seq.sort_values(by='report_date').reset_index() # .reset_index() เพื่อเอา original index (0 to N-1)
    
    total_samples = len(ids_seq)
    test_size = int(total_samples * TEST_RATIO)
    val_size = int(total_samples * VAL_RATIO)
    
    if total_samples - test_size - val_size < MIN_TRAIN_SAMPLES or total_samples < 100:
        raise ValueError("Not enough sequenced samples to perform time-based split.")
    
    train_end_index = total_samples - test_size - val_size
    val_end_index = total_samples - test_size
    
    # ดึง original indices (จาก .reset_index())
    train_indices = ids_seq.iloc[:train_end_index]['index'].values
    val_indices = ids_seq.iloc[train_end_index:val_end_index]['index'].values
    test_indices = ids_seq.iloc[val_end_index:]['index'].values
    
    # ใช้ indices เพื่อแบ่ง X_seq และ y_seq
    X_train, y_train = X_seq[train_indices], y_seq[train_indices]
    X_val, y_val = X_seq[val_indices], y_seq[val_indices]
    X_test, y_test = X_seq[test_indices], y_seq[test_indices]
    
    train_ids = ids_seq.iloc[:train_end_index]
    val_ids = ids_seq.iloc[train_end_index:val_end_index]
    test_ids = ids_seq.iloc[val_end_index:]

    logging.info(f"Data split (Sequenced):")
    logging.info(f"  Train: {len(X_train)} samples (Dates: {train_ids['report_date'].min().date()} - {train_ids['report_date'].max().date()})")
    logging.info(f"  Validation: {len(X_val)} samples (Dates: {val_ids['report_date'].min().date()} - {val_ids['report_date'].max().date()})")
    logging.info(f"  Test: {len(X_test)} samples (Dates: {test_ids['report_date'].min().date()} - {test_ids['report_date'].max().date()})")

    logging.info(f"  Train Y distribution:\n{pd.Series(y_train).value_counts(normalize=True)}")
    logging.info(f"  Validation Y distribution:\n{pd.Series(y_val).value_counts(normalize=True)}")
    logging.info(f"  Test Y distribution:\n{pd.Series(y_test).value_counts(normalize=True)}")

    return X_train, y_train, X_val, y_val, X_test, y_test

# --- [NEU] ฟังก์ชันสร้างโมเดล Keras ---
def build_model(input_shape):
    """สร้างโมเดล Sequential LSTM/GRU ของ Keras"""
    model = Sequential()
    
    # Input Shape = (SEQUENCE_LENGTH, num_features)
    # (เช่น (12, 41))
    
    # ใช้ Bidirectional GRU (เร็วกว่า LSTM และมักให้ผลดี)
    model.add(Bidirectional(GRU(64, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    
    model.add(Bidirectional(GRU(32, return_sequences=False))) # False เพราะเป็น Layer สุดท้ายก่อน Dense
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.4))
    
    model.add(Dense(1, activation='sigmoid')) # Output Layer
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[
            'accuracy', 
            tf.keras.metrics.Precision(name='precision'), 
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model

# --- [MODIFIED] ฟังก์ชันเทรนโมเดล ---
def train_model(X_train, y_train, X_val, y_val, num_features):
    """เทรนโมเดล Keras (LSTM/GRU)"""
    logging.info("5. Training Deep Learning (LSTM/GRU) model...")
    
    # 1. คำนวณ Class Weights (วิธีของ Keras)
    count_neg = np.sum(y_train == 0)
    count_pos = np.sum(y_train == 1)
    
    if count_pos == 0:
        logging.error("No positive samples (Y=1) in training data. Cannot train.")
        return None
        
    scale_pos_weight_value = count_neg / count_pos
    logging.info(f"  Calculated scale_pos_weight: {scale_pos_weight_value:.2f}")

    # Keras ใช้ class_weight dictionary
    class_weights = {0: 1.0, 1: scale_pos_weight_value}
    
    # 2. สร้างโมเดล
    model = build_model(input_shape=(SEQUENCE_LENGTH, num_features))
    model.summary()
    
    # 3. สร้าง Callbacks
    early_stopping = EarlyStopping(
        monitor='val_recall', # เน้น Recall บน Validation Set
        mode='max',
        patience=15, 
        verbose=1,
        restore_best_weights=True # คืนค่าน้ำหนักที่ดีที่สุด
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # 4. เทรนโมเดล
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100, # ตั้งไว้สูงๆ แล้วให้ EarlyStopping หยุด
        batch_size=64,
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    logging.info("Model training complete.")
    return model

# --- [UNCHANGED] ฟังก์ชันหา Threshold ---
# ฟังก์ชันนี้ใช้ได้เลย เพราะรับ y_pred_proba
def optimize_threshold(model, X_val, y_val, beta=2.0):
    """
    (FIX 3) Calculates the optimal threshold from the Validation set by explicitly maximizing F-beta score (F2-Score)
    with a stricter MIN_PRECISION constraint.
    """
    if X_val.shape[0] == 0 or y_val.shape[0] == 0:
        logging.warning("Validation set is empty, skipping threshold optimization.")
        return 0.5, 0.0, 0.0 

    logging.info("Optimizing Classification Threshold using Validation Set...")
    y_pred_proba = model.predict(X_val).ravel() # <-- .ravel()
    
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
    
    # Calculate F-beta scores (F2-Score) for all thresholds
    fbeta_scores = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-6)
    
    # Constraint: Only consider thresholds where Precision is at least 0.30
    MIN_PRECISION = 0.30 
    constrained_indices = np.where(precision[:-1] >= MIN_PRECISION)[0]
    
    if len(constrained_indices) > 0:
        optimal_idx = constrained_indices[np.argmax(fbeta_scores[constrained_indices])]
    else:
        # Fallback to pure F2-Score maximization
        optimal_idx = np.argmax(fbeta_scores[:-1])
        logging.warning(f"  Cannot meet minimum Precision target of {MIN_PRECISION:.2f}. Falling back to pure F{beta}-Score maximization.")

    optimal_threshold = thresholds[optimal_idx]
    optimal_precision = precision[optimal_idx]
    optimal_recall = recall[optimal_idx]
    optimal_fbeta = fbeta_scores[optimal_idx]

    logging.info(f"  Optimization Goal: Maximize F{beta}-Score (Min Precision {MIN_PRECISION:.2f}).")
    logging.info(f"  Optimal Threshold found: {optimal_threshold:.4f}")
    logging.info(f"  Resulting Precision: {optimal_precision:.4f}")
    logging.info(f"  Resulting Recall: {optimal_recall:.4f}")
    logging.info(f"  Resulting F{beta}-Score: {optimal_fbeta:.4f}")
    
    return optimal_threshold, optimal_recall, optimal_precision

# --- [UNCHANGED] ฟังก์ชันประเมินผล ---
def evaluate_model(model, X_test, y_test, optimal_threshold=0.5):
    """Evaluates the model on the test set and prints metrics."""
    logging.info(f"6. Evaluating model on Test Set (Threshold {optimal_threshold:.4f})...")
    y_pred_proba = model.predict(X_test).ravel() # <-- .ravel()
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred) 
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    if len(np.unique(y_test)) == 2:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    else:
        roc_auc = np.nan
        logging.warning("Cannot calculate AUC: Only one class present in y_test.")

    cm = confusion_matrix(y_test, y_pred)
    logging.info(f"--- Evaluation Metrics (Threshold {optimal_threshold:.4f}) ---")
    logging.info(f"Accuracy:  {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall:    {recall:.4f}")
    logging.info(f"F1-Score:  {f1:.4f}")
    logging.info(f"AUC-ROC:   {roc_auc:.4f}" if not np.isnan(roc_auc) else "AUC-ROC:   N/A")
    logging.info("--- Confusion Matrix ---")
    logging.info(f"\n{cm}")
    logging.info("--- Classification Report ---")
    logging.info(f"\n{classification_report(y_test, y_pred, zero_division=0)}")
    logging.info("--- End Evaluation ---")

# --- [MODIFIED] ฟังก์ชันบันทึกโมเดล ---
def save_model(model):
    """Saves the trained Keras model."""
    logging.info("7. Saving the trained model...")
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    try:
        model.save(model_path) # <-- [NEU] ใช้ Keras save
        logging.info(f"Model successfully saved to {model_path}")
    except Exception as e:
        logging.error(f"Error saving Keras model: {e}", exc_info=True)


# --- [MODIFIED] Main Execution ---
if __name__ == "__main__":
    logging.info("===== Starting ML Risk Prediction Script (LSTM Sequential Model) =====")
    start_time = datetime.now()

    # 1. Fetch Data (เหมือนเดิม)
    logging.info("1. Fetching comprehensive ML raw data from database starting from year 2010...")
    df_raw_data = get_ml_risk_raw_data(start_year=2010) 

    if df_raw_data.empty:
        logging.error("Stopping script because no financial data could be fetched.")
    else:
        # 2. Create Target Variable (เหมือนเดิม)
        df_financials_reset = df_raw_data.reset_index()
        df_with_y = create_target_variable(df_financials_reset, pd.DataFrame()) 

        if not df_with_y.empty:
            # 3. [NEU] Engineer Features and Create Sequences (With FIX)
            X_seq, y_seq, ids_seq, feature_names_final = engineer_features_and_sequences(df_with_y)
            
            if X_seq.shape[0] > 0:
                num_features = X_seq.shape[2] # (Num Samples, 12, 41) -> 41
            else:
                num_features = 0

            if num_features > 0:
                # 4. Split Data (เหมือนเดิม, แต่ใช้ X_seq, y_seq)
                try:
                    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X_seq, y_seq, ids_seq, feature_names_final)

                    # 5. [NEU] Train Keras Model
                    trained_model = train_model(X_train, y_train, X_val, y_val, num_features)

                    if trained_model:
                        # 6. Optimize Threshold (เหมือนเดิม)
                        optimized_threshold, _, _ = optimize_threshold(trained_model, X_val, y_val, beta=2.0)
                        
                        # 7. Evaluate Model (เหมือนเดิม)
                        evaluate_model(trained_model, X_test, y_test, optimized_threshold)

                        # 8. Save Model (เวอร์ชัน Keras)
                        save_model(trained_model)
                    
                    else:
                        logging.error("Model training failed (e.g., no positive samples).")

                except ValueError as ve:
                    logging.error(f"Error during data splitting or processing: {ve}")
                except Exception as e:
                    logging.error(f"An unexpected error occurred during ML pipeline: {e}", exc_info=True)
            else:
                logging.error("Stopping script because feature/sequence engineering resulted in empty data.")
        else:
            logging.error("Stopping script because target variable creation failed or resulted in empty data.")

    end_time = datetime.now()
    logging.info(f"===== ML Risk (LSTM) Script Finished in {end_time - start_time} =====")
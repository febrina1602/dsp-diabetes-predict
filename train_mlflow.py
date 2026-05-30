import pandas as pd
import numpy as np
import warnings
import joblib
import os
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# INISIALISASI DAGSHUB & MLFLOW
dagshub.init(repo_owner='febrina1602', repo_name='dsp-diabetes-predict', mlflow=True)

mlflow.set_experiment("Diabetes_Prediction_RandomForest")

def load_and_preprocess_data():
    print("Memuat dan memproses data...")
    df = pd.read_csv('Datasetfordiabetesresearch.csv', encoding='latin1')
    df.replace(' ', np.nan, inplace=True)

    # Hapus missing values > 30%
    missing_percent = df.isnull().sum() / len(df) * 100
    cols_to_drop = missing_percent[missing_percent > 30].index.tolist()
    df_cleaned = df.drop(columns=cols_to_drop)

    # Mencegah Data Leakage & ID
    cols_to_remove = ['NO', 'Data', 'VAR00001'] 
    leakage_cols = ['PGclass6', 'DM', 'FPGover7', 'P2PGover11', 'PG2h', 'FPG', 'HbA1c']
    cols_to_drop_final = [col for col in cols_to_remove + leakage_cols if col in df_cleaned.columns]
    df_cleaned = df_cleaned.drop(columns=cols_to_drop_final)

    # Imputasi
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df_cleaned.select_dtypes(exclude=['number']).columns.tolist()

    if 'PGclass3' in numeric_cols: numeric_cols.remove('PGclass3')
    if 'PGclass3' in categorical_cols: categorical_cols.remove('PGclass3')

    for col in numeric_cols:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

    for col in categorical_cols:
        if not df_cleaned[col].mode().empty:
             df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
        else:
             df_cleaned[col] = df_cleaned[col].fillna('Unknown')

    df_cleaned = df_cleaned.dropna(subset=['PGclass3'])

    # Encoding
    le = LabelEncoder()
    for col in df_cleaned.select_dtypes(include=['object']).columns:
        df_cleaned[col] = df_cleaned[col].astype(str)
        df_cleaned[col] = le.fit_transform(df_cleaned[col])

    # Memisahkan target
    X = df_cleaned.drop('PGclass3', axis=1)
    y = df_cleaned['PGclass3']

    return X, y

if __name__ == "__main__":
    # Load data
    X, y = load_and_preprocess_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Menerapkan SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # 14 Fitur Pilihan
    top_14_features = [
        'ISIGutt', 'Homa¦Â', 'A1cover6575', 'ThalussemiaAND', 'GA', 
        'INS2h', 'HomaIR', 'Bpdia', 'CP2h', 'Bpsys', 'BP13085', 
        'FIB4n', 'WHO1999hbp', 'GGT'
    ]

    # Potong dataset menjadi 14 fitur
    X_train_final = X_train_smote[top_14_features]
    X_test_final = X_test[top_14_features]

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_final), columns=top_14_features)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_final), columns=top_14_features)

    # --- MULAI MLFLOW TRACKING ---
    with mlflow.start_run(run_name="RandomForest_14_Features"):
        print("Melatih model Random Forest...")
        
        n_estimators = 100
        random_state = 42
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("features_count", len(top_14_features))
        mlflow.log_param("smote_applied", True)

        # Latih Model
        rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        rf_model.fit(X_train_scaled, y_train_smote)

        # Evaluasi Prediksi
        y_pred = rf_model.predict(X_test_scaled)
        
        # MENGHITUNG MULTIPLE METRICS (weighted untuk dataset multiclass)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        print("\n--- HASIL EVALUASI MODEL ---")
        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {prec:.4f}")
        print(f"Recall    : {rec:.4f}")
        print(f"F1-Score  : {f1:.4f}")
        print("----------------------------\n")

        mlflow.sklearn.log_model(rf_model, "random_forest_model")

        joblib.dump(scaler, 'scaler_14features.pkl')
        joblib.dump(top_14_features, 'top_14_features_list.pkl')
        joblib.dump(rf_model, 'rf_diabetes_14features.pkl')

        mlflow.log_artifact('scaler_14features.pkl', artifact_path="preprocessing_assets")
        mlflow.log_artifact('top_14_features_list.pkl', artifact_path="preprocessing_assets")
        mlflow.log_artifact('rf_diabetes_14features.pkl', artifact_path="model_assets")

        print("Run MLflow selesai! Silakan cek dashboard DagsHub.")
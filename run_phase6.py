import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models import prepare_features_tier2, train_tier2_model, evaluate_and_plot_tier2

if __name__ == "__main__":
    processed_path = "data/processed/cleaned_unsw_nb15.parquet"
    if not os.path.exists(processed_path):
        raise FileNotFoundError("Run Phase 4 first!")
        
    print("[INFO] Loading cleaned dataset for Tier 2...")
    df = pd.read_parquet(processed_path)
    
    print("[INFO] Preparing features for Tier 2 (Attack traffic only)...")
    X, y, le = prepare_features_tier2(df)
    
    print("[INFO] Splitting dataset into Train (80%) and Test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # تدريب النموذج
    model = train_tier2_model(X_train, y_train)
    
    # تقييم النموذج
    evaluate_and_plot_tier2(model, X_test, y_test, le)
    
    # حفظ النموذج والـ Label Encoder
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "tier2_xgboost.joblib")
    encoder_path = os.path.join(models_dir, "label_encoder.joblib")
    
    joblib.dump(model, model_path)
    joblib.dump(le, encoder_path)
    
    print(f"\n[SUCCESS] Tier 2 model saved successfully to: {model_path}")
    print(f"[SUCCESS] Label Encoder saved to: {encoder_path}")
    
    print("\n==========================================")
    print("           PHASE 6 COMPLETE               ")
    print("==========================================")
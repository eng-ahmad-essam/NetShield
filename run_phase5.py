import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models import prepare_features_tier1, train_tier1_model, evaluate_and_plot_tier1

if __name__ == "__main__":
    processed_path = "data/processed/cleaned_unsw_nb15.parquet"
    if not os.path.exists(processed_path):
        raise FileNotFoundError("Run Phase 4 first to generate the cleaned dataset!")
        
    print("[INFO] Loading cleaned dataset...")
    df = pd.read_parquet(processed_path)
    
    print("[INFO] Preparing features for Tier 1...")
    X, y = prepare_features_tier1(df)
    
    print("[INFO] Splitting dataset into Train (80%) and Test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # تدريب النموذج
    model = train_tier1_model(X_train, y_train)
    
    # تقييم النموذج
    evaluate_and_plot_tier1(model, X_test, y_test)
    
    # حفظ النموذج المدرب
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "tier1_xgboost.joblib")
    joblib.dump(model, model_path)
    print(f"\n[SUCCESS] Tier 1 model saved successfully to: {model_path}")
    
    print("\n==========================================")
    print("           PHASE 5 COMPLETE               ")
    print("==========================================")
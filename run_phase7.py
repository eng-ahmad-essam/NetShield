import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from src.pipeline import NetShieldPipeline

if __name__ == "__main__":
    processed_path = "data/processed/cleaned_unsw_nb15.parquet"
    if not os.path.exists(processed_path):
        raise FileNotFoundError("Run Phase 4 first!")
        
    print("[INFO] Loading cleaned test dataset...")
    df = pd.read_parquet(processed_path)
    
    # اختيار عينة اختبار (20% من البيانات)
    test_df = df.sample(frac=0.2, random_state=42).reset_index(drop=True)
    
    # تشغيل الـ Pipeline
    pipeline = NetShieldPipeline()
    print("[PIPELINE] Running Hierarchical Inference on Test Sample...")
    
    final_preds, t1_preds, t1_probas = pipeline.predict(test_df)
    
    # تجهيز True Labels لمقارنة النتائج النهائية
    true_cats = test_df['attack_cat'].values
    
    print("\n==========================================")
    print("      NETSHIELD END-TO-END EVALUATION     ")
    print("==========================================")
    
    overall_acc = accuracy_score(true_cats, final_preds)
    print(f"Hierarchical System Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)\n")
    
    report = classification_report(true_cats, final_preds)
    print("Detailed Final Classification Report:\n", report)
    
    output_dir = "results/metrics"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "end_to_end_report.txt"), "w") as f:
        f.write(f"Hierarchical System Overall Accuracy: {overall_acc:.4f}\n\n" + report)
        
    print("\n==========================================")
    print("           PHASE 7 COMPLETE               ")
    print("==========================================")
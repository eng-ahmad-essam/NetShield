import os
from src.data_loader import load_full_dataset, clean_and_preprocess

if __name__ == "__main__":
    df = load_full_dataset("data/raw")
    
    print("\n--- STARTING DATA CLEANING & PREPROCESSING ---")
    clean_df = clean_and_preprocess(df)
    
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cleaned_unsw_nb15.parquet")
    
    print(f"[INFO] Saving cleaned dataset to {output_path}...")
    # حفظ الملف بصيغة Parquet السريعة والمضغوطة
    clean_df.to_parquet(output_path, index=False)
    
    print("\n==========================================")
    print("           PHASE 4 COMPLETE               ")
    print("==========================================")
    print(f"Final Cleaned Dataset Shape: {clean_df.shape}")
    print("\nUpdated Attack Categories:")
    print(clean_df['attack_cat'].value_counts())
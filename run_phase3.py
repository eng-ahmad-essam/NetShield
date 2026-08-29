import pandas as pd
from src.data_loader import load_full_dataset
from src.visualization import plot_class_distributions

if __name__ == "__main__":
    df = load_full_dataset("data/raw")
    
    print("\n==========================================")
    print("           EDA ANALYSIS REPORT            ")
    print("==========================================")
    
    # 1. Label Distribution (Tier 1)
    print("\n--- 1. TIER 1: BINARY LABEL DISTRIBUTION ---")
    tier1_counts = df['label'].value_counts()
    tier1_percents = df['label'].value_counts(normalize=True) * 100
    t1_summary = pd.DataFrame({'Counts': tier1_counts, 'Percentage (%)': tier1_percents.round(2)})
    t1_summary.index = t1_summary.index.map({0: 'Normal (0)', 1: 'Attack (1)'})
    print(t1_summary)
    
    # 2. Attack Category Distribution (Tier 2)
    print("\n--- 2. TIER 2: MULTICLASS ATTACK CATEGORIES ---")
    attack_df = df[df['label'] == 1]
    # Clean string spaces if any
    clean_cats = attack_df['attack_cat'].astype(str).str.strip().replace({'nan': 'Unlabeled/Missing'})
    t2_counts = clean_cats.value_counts()
    t2_percents = (clean_cats.value_counts(normalize=True) * 100).round(2)
    t2_summary = pd.DataFrame({'Counts': t2_counts, 'Percentage (%)': t2_percents})
    print(t2_summary)
    
    # 3. Categorical vs Numerical Features
    print("\n--- 3. FEATURE TYPES SUMMARY ---")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df.select_dtypes(include=['int8', 'int16', 'int32', 'int64', 'float32', 'float64']).columns.tolist()
    print(f"Categorical Features ({len(cat_cols)}): {cat_cols}")
    print(f"Numerical Features ({len(num_cols)}): {len(num_cols)} columns")
    
    # 4. Missing Values Summary
    print("\n--- 4. MISSING VALUES SUMMARY ---")
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        print(pd.DataFrame({'Missing Count': null_cols, 'Percentage (%)': (null_cols / len(df) * 100).round(2)}))
    else:
        print("No missing values found.")
        
    # 5. Duplicate Rows Count
    print("\n--- 5. DUPLICATE ROWS ANALYSIS ---")
    duplicates = df.duplicated().sum()
    print(f"Total Duplicate Rows: {duplicates:,} ({(duplicates/len(df)*100):.2f}%)")
    
    # 6. Generate Plots
    print("\n--- 6. GENERATING VISUALIZATIONS ---")
    plot_class_distributions(df)
    print("\n==========================================")
    print("           PHASE 3 COMPLETE               ")
    print("==========================================")
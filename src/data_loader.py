import os
import pandas as pd
import numpy as np

def load_feature_names(features_path):
    try:
        df_feat = pd.read_csv(features_path, encoding='utf-8')
    except UnicodeDecodeError:
        df_feat = pd.read_csv(features_path, encoding='ISO-8859-1')
        
    df_feat.columns = [col.strip().lower() for col in df_feat.columns]
    name_col = [c for c in df_feat.columns if 'name' in c][0]
    return df_feat[name_col].str.strip().tolist()

def optimize_dtypes(df):
    """تقليل استهلاك الذاكرة عبر تحسين أنواع البيانات"""
    categorical_cols = ['srcip', 'dstip', 'proto', 'state', 'service', 'attack_cat']
    
    for col in df.columns:
        if col in categorical_cols:
            continue
            
        df[col] = pd.to_numeric(df[col], errors='coerce')
        c_min = df[col].min()
        c_max = df[col].max()
        
        if pd.isna(c_min) or pd.isna(c_max):
            continue
            
        col_type = df[col].dtype
        
        if np.issubdtype(col_type, np.integer):
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        elif np.issubdtype(col_type, np.floating):
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
                
    return df

def load_full_dataset(raw_dir="data/raw"):
    """دمج كل الملفات في DataFrame واحد محين"""
    files = [f"NUSW-NB15_{i}.csv" for i in range(1, 5)]
    features_file = os.path.join(raw_dir, "NUSW-NB15_features.csv")
    
    feature_names = load_feature_names(features_file)
    
    df_list = []
    print("[INFO] Loading raw datasets...")
    for file_name in files:
        file_path = os.path.join(raw_dir, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, header=None, names=feature_names, low_memory=False)
            df_list.append(df)
            
    full_df = pd.concat(df_list, ignore_index=True)
    full_df.columns = full_df.columns.str.strip().str.lower()
    full_df = optimize_dtypes(full_df)
    return full_df

def clean_and_preprocess(df):
    """تنظيف وتجهيز البيانات بالكامل"""
    print("[CLEANING] Removing duplicate rows...")
    df = df.drop_duplicates().reset_index(drop=True)
    
    print("[CLEANING] Cleaning attack categories...")
    # توحيد النصوص وحذف المسافات الزائدة ودمج Backdoors مع Backdoor
    if 'attack_cat' in df.columns:
        df['attack_cat'] = df['attack_cat'].astype(str).str.strip()
        df['attack_cat'] = df['attack_cat'].replace({
            'Backdoors': 'Backdoor',
            'nan': 'Normal',
            '': 'Normal'
        })
        # لو الحركة العادية label = 0 يبقى الفئة Normal
        df.loc[df['label'] == 0, 'attack_cat'] = 'Normal'

    print("[CLEANING] Imputing missing values...")
    # ملء الـ Nulls في الأعمدة الرقمية بـ 0
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    
    # حذف أعمدة الـ IP لأنها بتسبب Overfitting وغير تعميمية
    drop_ips = ['srcip', 'dstip', 'sport', 'dsport']
    df = df.drop(columns=[col for col in drop_ips if col in df.columns])
    
    return df
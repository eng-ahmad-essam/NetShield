import os
import joblib
import pandas as pd
import numpy as np

class NetShieldPipeline:
    def __init__(self, tier1_path="models/tier1_xgboost.joblib", 
                 tier2_path="models/tier2_xgboost.joblib", 
                 encoder_path="models/label_encoder.joblib"):
        
        print("[PIPELINE] Loading NetShield Trained Models...")
        self.tier1_model = joblib.load(tier1_path)
        self.tier2_model = joblib.load(tier2_path)
        self.label_encoder = joblib.load(encoder_path)
        
    def _prepare_features(self, df):
        """تحويل البيانات للتنسيق المتوافق مع النماذج"""
        X = df.drop(columns=['label', 'attack_cat'], errors='ignore').copy()
        
        # تحويل الأعمدة النصية
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            X[col] = X[col].astype('category').cat.codes
            
        return X

    def predict(self, df):
        """التنبؤ الهرمي الكامل (Tier 1 -> Tier 2)"""
        X = self._prepare_features(df)
        
        # 1. Tier 1 Prediction (Binary: 0=Normal, 1=Attack)
        t1_preds = self.tier1_model.predict(X)
        t1_probas = self.tier1_model.predict_proba(X)[:, 1]
        
        final_predictions = np.array(['Normal'] * len(df), dtype=object)
        
        # 2. Tier 2 Prediction (حركة الهجمات فقط)
        attack_indices = np.where(t1_preds == 1)[0]
        
        if len(attack_indices) > 0:
            X_attack = X.iloc[attack_indices]
            t2_preds_encoded = self.tier2_model.predict(X_attack)
            t2_preds_labels = self.label_encoder.inverse_transform(t2_preds_encoded)
            
            final_predictions[attack_indices] = t2_preds_labels
            
        return final_predictions, t1_preds, t1_probas
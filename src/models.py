import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report, 
                             confusion_matrix)

def prepare_features_tier1(df):
    """تجهيز البيانات لـ Tier 1 (حذف Target والأعمدة غير المطلوبة وتحويل Categorical)"""
    X = df.drop(columns=['label', 'attack_cat'], errors='ignore')
    y = df['label']
    
    # تحويل الأعمدة النصية المتبقية إلى Category Codes لتناسب XGBoost
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        X[col] = X[col].astype('category').cat.codes
        
    return X, y

def train_tier1_model(X_train, y_train):
    """تدريب نموذج XGBoost لـ Tier 1"""
    print("[TRAINING] Training Tier 1 XGBoost Binary Classifier...")
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        tree_method='hist' # لتسريع التدريب على البيانات الضخمة
    )
    model.fit(X_train, y_train)
    return model

def evaluate_and_plot_tier1(model, X_test, y_test, output_dir="results"):
    """تقييم النموذج وحفظ النتائج والـ Confusion Matrix"""
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    
    print("[EVALUATION] Generating predictions...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    metrics_str = f"""==========================================
TIER 1 BINARY CLASSIFICATION METRICS
==========================================
Accuracy  : {acc:.4f} ({acc*100:.2f}%)
Precision : {prec:.4f}
Recall    : {rec:.4f}
F1-Score  : {f1:.4f}
ROC-AUC   : {auc:.4f}
==========================================
"""
    print(metrics_str)
    
    # حفظ التقرير في ملف نصي
    with open(os.path.join(output_dir, "metrics/tier1_report.txt"), "w") as f:
        f.write(metrics_str + "\n\n" + classification_report(y_test, y_pred))
        
    # رسم Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Attack'], 
                yticklabels=['Normal', 'Attack'])
    plt.title('Tier 1: Confusion Matrix (Normal vs Attack)', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, "figures/tier1_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[INFO] Confusion Matrix saved to: {cm_path}")


# ==========================================
# TIER 2: MULTICLASS MODEL FUNCTIONS
# ==========================================
from sklearn.preprocessing import LabelEncoder

def prepare_features_tier2(df):
    """تجهيز البيانات لـ Tier 2 (على بيانات الهجمات فقط)"""
    # نأخذ حركة الهجمات فقط (label == 1)
    attack_df = df[df['label'] == 1].copy()
    
    # حذف الفئات غير المعروفة أو الـ Normal
    attack_df = attack_df[attack_df['attack_cat'] != 'Normal']
    
    X = attack_df.drop(columns=['label', 'attack_cat'], errors='ignore')
    
    # تحويل الأعمدة النصية في X إلى Category Codes
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        X[col] = X[col].astype('category').cat.codes
        
    # Label Encoding للـ Target (attack_cat)
    le = LabelEncoder()
    y = le.fit_transform(attack_df['attack_cat'])
    
    return X, y, le

def train_tier2_model(X_train, y_train):
    """تدريب نموذج XGBoost Multiclass لـ Tier 2"""
    print("[TRAINING] Training Tier 2 XGBoost Multiclass Classifier...")
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        objective='multi:softprob'
    )
    model.fit(X_train, y_train)
    return model

def evaluate_and_plot_tier2(model, X_test, y_test, label_encoder, output_dir="results"):
    """تقييم Tier 2 وحفظ النتائج والـ Confusion Matrix"""
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    
    print("[EVALUATION] Generating predictions for Tier 2...")
    y_pred = model.predict(X_test)
    
    classes = label_encoder.classes_
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    metrics_str = f"""==========================================
TIER 2 MULTICLASS CLASSIFICATION METRICS
==========================================
Overall Accuracy : {acc:.4f} ({acc*100:.2f}%)
Macro F1-Score   : {f1_macro:.4f}
Weighted F1-Score: {f1_weighted:.4f}
==========================================
"""
    print(metrics_str)
    
    # حفظ التقرير في ملف نصي
    report = classification_report(y_test, y_pred, target_names=classes)
    print("Detailed Classification Report:\n", report)
    
    with open(os.path.join(output_dir, "metrics/tier2_report.txt"), "w") as f:
        f.write(metrics_str + "\n\n" + report)
        
    # رسم Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Tier 2: Confusion Matrix (Attack Categories)', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Category')
    plt.ylabel('True Category')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, "figures/tier2_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[INFO] Tier 2 Confusion Matrix saved to: {cm_path}")
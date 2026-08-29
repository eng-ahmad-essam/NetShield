import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distributions(df, output_dir="results/figures"):
    """رسم وحفظ توزيعات الفئات لـ Tier 1 و Tier 2"""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. Tier 1 Distribution (Binary: Label)
    plt.figure(figsize=(7, 5))
    ax1 = sns.countplot(data=df, x='label', palette='Set2')
    plt.title('Tier 1: Target Distribution (0 = Normal, 1 = Attack)', fontsize=12, fontweight='bold')
    plt.xlabel('Traffic Label', fontsize=10)
    plt.ylabel('Total Packets', fontsize=10)
    
    # Add count labels on top of bars
    for p in ax1.patches:
        ax1.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)
        
    plt.tight_layout()
    tier1_path = os.path.join(output_dir, "tier1_distribution.png")
    plt.savefig(tier1_path, dpi=300)
    plt.close()
    
    # 2. Tier 2 Distribution (Multiclass: attack_cat)
    attack_df = df[df['label'] == 1].copy()
    # Handle NaN or trailing spaces in attack categories
    attack_df['attack_cat'] = attack_df['attack_cat'].fillna('Normal/Unlabeled').str.strip()
    
    plt.figure(figsize=(12, 6))
    order = attack_df['attack_cat'].value_counts().index
    ax2 = sns.countplot(data=attack_df, y='attack_cat', order=order, palette='viridis')
    plt.title('Tier 2: Attack Categories Distribution (Attack Traffic Only)', fontsize=12, fontweight='bold')
    plt.xlabel('Total Packets', fontsize=10)
    plt.ylabel('Attack Category', fontsize=10)
    
    for p in ax2.patches:
        ax2.annotate(f'{int(p.get_width()):,}', (p.get_width() + 5000, p.get_y() + p.get_height() / 2.),
                    ha='center', va='center', xytext=(5, 0), textcoords='offset points', fontsize=8)
        
    plt.tight_layout()
    tier2_path = os.path.join(output_dir, "tier2_distribution.png")
    plt.savefig(tier2_path, dpi=300)
    plt.close()
    
    print(f"[INFO] Distributions saved to:\n - {tier1_path}\n - {tier2_path}")
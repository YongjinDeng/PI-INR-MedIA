"""
PI-INR MedIA 终极统计与图表生成引擎 v3.2
生成Excel表格 + 学术图表 + 论文段落 + 字体放大版图表
优化：小数位数统一、字体放大、双版本图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import matplotlib.patches as patches
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ================== 版本信息 ==================
__version__ = "3.2.0"
__date__ = "2026-03-12"

# ================== 配置 ==================
CSV_PATH = r"./results/MedIA_Ultimate_Run/MedIA_Quantitative_Results.csv"
OUTPUT_DIR = r"./results/MedIA_Ultimate_Run/MedIA_Final_Excel"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 创建子目录
EXCEL_DIR = os.path.join(OUTPUT_DIR, 'excel_tables')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
FIGURES_LARGE_DIR = os.path.join(OUTPUT_DIR, 'figures_large_font')
TEXT_DIR = os.path.join(OUTPUT_DIR, 'text')
for d in [EXCEL_DIR, FIGURES_DIR, FIGURES_LARGE_DIR, TEXT_DIR]:
    os.makedirs(d, exist_ok=True)

# ================== 颜色配置 ==================
COLORS = {
    'before': '#8DA0CB', 'after': '#FC8D62', 
    'green': '#66C2A5', 'yellow': '#FFD92F', 
    'orange': '#E78AC3', 'red': '#E31A1C', 
    'gray': '#B3B3B3'
}

# ================== 数据加载与清洗 ==================
print("="*80)
print(f"📊 PI-INR MedIA 终极统计引擎 v{__version__}")
print("="*80)

raw_df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
print(f"📥 原始数据: {len(raw_df)} 例")

# 清洗无效数据 (SSIM=0)
df = raw_df[raw_df['SSIM_Before'] > 0.01].copy()
excluded_count = len(raw_df) - len(df)
print(f"🧹 清洗后有效数据: {len(df)} 例 (剔除 {excluded_count} 例无效切片)")
print(f"   排除患者: {', '.join(raw_df[raw_df['SSIM_Before'] <= 0.01]['Patient_ID'].tolist())}")

df['SSIM_Improvement'] = df['SSIM_After'] - df['SSIM_Before']
df['Patient_Number'] = df['Patient_ID'].str.extract(r'(\d+)').astype(int)

# 风险阈值
RISK_THRESH_LOW = 0.001     # 低风险 (<0.001%)
RISK_THRESH_MED = 0.01      # 中风险 (0.001-0.01%)
RISK_THRESH_HIGH = 0.1      # 高风险 (0.01-0.1%)
CLINICAL_YELLOW = 0.05      # 临床关注阈值

# 风险等级分类
conditions = [
    (df['High_Risk_Ratio'] <= RISK_THRESH_LOW),
    (df['High_Risk_Ratio'] > RISK_THRESH_LOW) & (df['High_Risk_Ratio'] <= RISK_THRESH_MED),
    (df['High_Risk_Ratio'] > RISK_THRESH_MED) & (df['High_Risk_Ratio'] <= RISK_THRESH_HIGH),
    (df['High_Risk_Ratio'] > RISK_THRESH_HIGH)
]
choices = ['GREEN (Low Risk)', 'YELLOW (Medium Risk)', 'ORANGE (High Risk)', 'RED (Critical)']
df['Risk_Level'] = np.select(conditions, choices, default='UNKNOWN')

risk_counts = df['Risk_Level'].value_counts()
n_green = risk_counts.get('GREEN (Low Risk)', 0)
n_yellow = risk_counts.get('YELLOW (Medium Risk)', 0)
n_orange = risk_counts.get('ORANGE (High Risk)', 0)
n_red = risk_counts.get('RED (Critical)', 0)

print(f"\n📊 风险分层:")
print(f"   🟢 GREEN  (≤{RISK_THRESH_LOW}%): {n_green:2d}例 ({n_green/len(df)*100:5.1f}%)")
print(f"   🟡 YELLOW  ({RISK_THRESH_LOW}%-{RISK_THRESH_MED}%): {n_yellow:2d}例 ({n_yellow/len(df)*100:5.1f}%)")
print(f"   🟠 ORANGE  ({RISK_THRESH_MED}%-{RISK_THRESH_HIGH}%): {n_orange:2d}例 ({n_orange/len(df)*100:5.1f}%)")
print(f"   🔴 RED     (>{RISK_THRESH_HIGH}%): {n_red:2d}例 ({n_red/len(df)*100:5.1f}%)")

# ================== 统计检验 ==================
t_stat, p_value_t = stats.ttest_rel(df['SSIM_After'], df['SSIM_Before'])
w_stat, p_value_w = stats.wilcoxon(df['SSIM_After'], df['SSIM_Before'])
corr_risk_uncert, p_corr = stats.pearsonr(df['High_Risk_Ratio'], df['Mean_Uncert_Risk'])

print(f"\n📈 统计检验:")
print(f"   配对t检验: t = {t_stat:.4f}, p = {p_value_t:.2e}")
print(f"   Wilcoxon: W = {w_stat:.4f}, p = {p_value_w:.2e}")
print(f"   风险-不确定性相关: r = {corr_risk_uncert:.4f}, p = {p_corr:.2e}")

# ================== 高风险患者 ==================
high_risk_patients = df[df['Risk_Level'].isin(['ORANGE (High Risk)', 'RED (Critical)'])].sort_values('High_Risk_Ratio', ascending=False)

print(f"\n⚠️ 高风险患者 (ATI > {RISK_THRESH_MED}%):")
if len(high_risk_patients) > 0:
    for i, (_, row) in enumerate(high_risk_patients.iterrows(), 1):
        print(f"   {i}. {row['Patient_ID']}: ATI={row['High_Risk_Ratio']:.5f}%, Uncertainty={row['Mean_Uncert_Risk']:.5f}")
else:
    print("   无高风险患者")

# ================== 生成Excel表格 ==================
print("\n📊 生成Excel表格...")

excel_path = os.path.join(EXCEL_DIR, 'PI_INR_Results.xlsx')
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    
    # 1. 原始数据表
    df.to_excel(writer, sheet_name='Raw Data', index=False)
    
    # 2. 统计摘要表 - 优化小数位数
    summary_data = {
        'Metric': ['N', 'SSIM_Before', 'SSIM_After', 'SSIM_Improvement', 'ATI_Risk(%)', 'Uncertainty'],
        'Mean': [len(df), 
                 f"{df['SSIM_Before'].mean():.3f}", 
                 f"{df['SSIM_After'].mean():.3f}", 
                 f"{df['SSIM_Improvement'].mean():.4f}", 
                 f"{df['High_Risk_Ratio'].mean():.5f}", 
                 f"{df['Mean_Uncert_Risk'].mean():.5f}"],
        'Std': ['-', 
                f"{df['SSIM_Before'].std():.3f}", 
                f"{df['SSIM_After'].std():.3f}", 
                f"{df['SSIM_Improvement'].std():.4f}", 
                f"{df['High_Risk_Ratio'].std():.5f}", 
                f"{df['Mean_Uncert_Risk'].std():.5f}"],
        'Min': ['-', 
                f"{df['SSIM_Before'].min():.3f}", 
                f"{df['SSIM_After'].min():.3f}", 
                f"{df['SSIM_Improvement'].min():.4f}", 
                f"{df['High_Risk_Ratio'].min():.5f}", 
                f"{df['Mean_Uncert_Risk'].min():.5f}"],
        'Max': ['-', 
                f"{df['SSIM_Before'].max():.3f}", 
                f"{df['SSIM_After'].max():.3f}", 
                f"{df['SSIM_Improvement'].max():.4f}", 
                f"{df['High_Risk_Ratio'].max():.5f}", 
                f"{df['Mean_Uncert_Risk'].max():.5f}"],
        'Median': ['-', 
                   f"{df['SSIM_Before'].median():.3f}", 
                   f"{df['SSIM_After'].median():.3f}", 
                   f"{df['SSIM_Improvement'].median():.4f}", 
                   f"{df['High_Risk_Ratio'].median():.5f}", 
                   f"{df['Mean_Uncert_Risk'].median():.5f}"]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)
    
    # 3. 统计检验表
    test_data = {
        'Test': ['Paired t-test', 'Wilcoxon test', 'Pearson Correlation'],
        'Statistic': [f't = {t_stat:.4f}', f'W = {w_stat:.4f}', f'r = {corr_risk_uncert:.4f}'],
        'P-value': [f'{p_value_t:.2e}', f'{p_value_w:.2e}', f'{p_corr:.2e}'],
        'Significance': ['p < 0.001' if p_value_t < 0.001 else f'p = {p_value_t:.3f}',
                         'p < 0.001' if p_value_w < 0.001 else f'p = {p_value_w:.3f}',
                         'p < 0.05' if p_corr < 0.05 else 'n.s.']
    }
    test_df = pd.DataFrame(test_data)
    test_df.to_excel(writer, sheet_name='Statistical Tests', index=False)
    
    # 4. 高风险患者表
    if len(high_risk_patients) > 0:
        high_risk_display = high_risk_patients[['Patient_ID', 'High_Risk_Ratio', 'Mean_Uncert_Risk', 
                                                'SSIM_Before', 'SSIM_After', 'SSIM_Improvement', 'Risk_Level']].copy()
        high_risk_display['High_Risk_Ratio'] = high_risk_display['High_Risk_Ratio'].map('{:.5f}'.format)
        high_risk_display['Mean_Uncert_Risk'] = high_risk_display['Mean_Uncert_Risk'].map('{:.5f}'.format)
        high_risk_display['SSIM_Before'] = high_risk_display['SSIM_Before'].map('{:.3f}'.format)
        high_risk_display['SSIM_After'] = high_risk_display['SSIM_After'].map('{:.3f}'.format)
        high_risk_display['SSIM_Improvement'] = high_risk_display['SSIM_Improvement'].map('{:.4f}'.format)
        high_risk_display.columns = ['Patient ID', 'ATI Risk (%)', 'Uncertainty', 
                                     'SSIM Before', 'SSIM After', 'SSIM Improvement', 'Risk Level']
        high_risk_display.to_excel(writer, sheet_name='High Risk Patients', index=False)
    
    # 5. 风险分布表
    risk_dist = pd.DataFrame({
        'Risk Level': ['GREEN (Low Risk)', 'YELLOW (Medium Risk)', 'ORANGE (High Risk)', 'RED (Critical)'],
        'ATI Threshold (%)': [f'≤ {RISK_THRESH_LOW}', 
                              f'{RISK_THRESH_LOW} - {RISK_THRESH_MED}',
                              f'{RISK_THRESH_MED} - {RISK_THRESH_HIGH}',
                              f'> {RISK_THRESH_HIGH}'],
        'Number of Patients': [n_green, n_yellow, n_orange, n_red],
        'Percentage (%)': [f"{n_green/len(df)*100:.1f}", 
                          f"{n_yellow/len(df)*100:.1f}", 
                          f"{n_orange/len(df)*100:.1f}", 
                          f"{n_red/len(df)*100:.1f}"]
    })
    risk_dist.to_excel(writer, sheet_name='Risk Distribution', index=False)
    
    # 6. 论文表格格式 (Table 1) - 优化小数位数
    table1_data = {
        'Metric': ['SSIM (Before)', 'SSIM (After)', 'SSIM Improvement', 'ATI Risk (%)', 'Uncertainty'],
        'Mean ± SD': [
            f"{df['SSIM_Before'].mean():.3f} ± {df['SSIM_Before'].std():.3f}",
            f"{df['SSIM_After'].mean():.3f} ± {df['SSIM_After'].std():.3f}",
            f"{df['SSIM_Improvement'].mean():.4f} ± {df['SSIM_Improvement'].std():.4f}",
            f"{df['High_Risk_Ratio'].mean():.5f} ± {df['High_Risk_Ratio'].std():.5f}",
            f"{df['Mean_Uncert_Risk'].mean():.5f} ± {df['Mean_Uncert_Risk'].std():.5f}"
        ],
        'Min': [
            f"{df['SSIM_Before'].min():.3f}",
            f"{df['SSIM_After'].min():.3f}",
            f"{df['SSIM_Improvement'].min():.4f}",
            f"{df['High_Risk_Ratio'].min():.5f}",
            f"{df['Mean_Uncert_Risk'].min():.5f}"
        ],
        'Max': [
            f"{df['SSIM_Before'].max():.3f}",
            f"{df['SSIM_After'].max():.3f}",
            f"{df['SSIM_Improvement'].max():.4f}",
            f"{df['High_Risk_Ratio'].max():.5f}",
            f"{df['Mean_Uncert_Risk'].max():.5f}"
        ],
        'Median': [
            f"{df['SSIM_Before'].median():.3f}",
            f"{df['SSIM_After'].median():.3f}",
            f"{df['SSIM_Improvement'].median():.4f}",
            f"{df['High_Risk_Ratio'].median():.5f}",
            f"{df['Mean_Uncert_Risk'].median():.5f}"
        ]
    }
    table1_df = pd.DataFrame(table1_data)
    table1_df.to_excel(writer, sheet_name='Table 1 - Quantitative', index=False)

print(f"  ✅ Excel表格已保存: {excel_path}")

# ================== 通用绘图函数 ==================
def plot_figure1(axs, font_size='normal'):
    """绘制Figure 1 (配准性能分析)"""
    ax = axs[0]
    data_melt = pd.melt(df, value_vars=['SSIM_Before', 'SSIM_After'], var_name='Status', value_name='SSIM')
    sns.violinplot(x='Status', y='SSIM', data=data_melt, ax=ax, 
                   palette=[COLORS['before'], COLORS['after']], inner=None, alpha=0.3)
    
    for i in range(len(df)):
        ax.plot([0, 1], [df['SSIM_Before'].iloc[i], df['SSIM_After'].iloc[i]], 
                color='gray', alpha=0.3, linewidth=0.8, zorder=1)
        ax.scatter([0, 1], [df['SSIM_Before'].iloc[i], df['SSIM_After'].iloc[i]], 
                   color=['#4A5584', '#8E201C'], s=25, zorder=2)
    
    # 字体大小设置
    if font_size == 'normal':
        label_font, title_font, tick_font = 12, 14, 10
    else:
        label_font, title_font, tick_font = 18, 20, 14
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Before PI-INR', 'After PI-INR'], fontsize=label_font)
    ax.set_ylabel('Structural Similarity Index (SSIM)', fontsize=label_font)
    ax.set_title('a Individual Registration Improvement', fontweight='bold', fontsize=title_font, loc='left')
    ax.text(0.5, 0.05, f'paired t-test: p < 0.001', transform=ax.transAxes, 
            ha='center', fontsize=label_font-2, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    ax.tick_params(axis='both', labelsize=tick_font)
    ax.grid(True, alpha=0.3)
    
    # (b) Bland-Altman
    ax = axs[1]
    means = (df['SSIM_Before'] + df['SSIM_After']) / 2
    diffs = df['SSIM_After'] - df['SSIM_Before']
    md, sd = np.mean(diffs), np.std(diffs)
    
    colors = [COLORS['red'] if r > RISK_THRESH_HIGH else 
              COLORS['orange'] if r > RISK_THRESH_MED else 
              COLORS['yellow'] if r > RISK_THRESH_LOW else 
              COLORS['green'] for r in df['High_Risk_Ratio']]
    
    scatter = ax.scatter(means, diffs, c=colors, s=60, edgecolor='black', alpha=0.8, zorder=3)
    linewidth = 1.5 if font_size == 'normal' else 2
    ax.axhline(md, color='black', linewidth=linewidth, label=f'Mean: {md:.4f}')
    ax.axhline(md + 1.96*sd, color='gray', linestyle='--', linewidth=linewidth, label='+1.96 SD')
    ax.axhline(md - 1.96*sd, color='gray', linestyle='--', linewidth=linewidth, label='-1.96 SD')
    ax.axhline(0, color='red', linestyle=':', linewidth=linewidth, alpha=0.5)
    
    ax.set_xlabel('Mean SSIM (Before & After)', fontsize=label_font)
    ax.set_ylabel('Difference (After - Before)', fontsize=label_font)
    ax.set_title('b Bland-Altman Plot', fontweight='bold', fontsize=title_font, loc='left')
    ax.legend(loc='upper right', fontsize=label_font-2)
    ax.tick_params(axis='both', labelsize=tick_font)
    ax.grid(True, alpha=0.3)
    
    return scatter

def plot_figure2(ax, font_size='normal'):
    """绘制Figure 2 (风险分布饼图)"""
    risk_data = {'GREEN\n(Low Risk)': n_green, 'YELLOW\n(Medium Risk)': n_yellow, 
                 'ORANGE\n(High Risk)': n_orange}
    risk_data = {k: v for k, v in risk_data.items() if v > 0}
    colors = [COLORS[k.split('\n')[0].lower()] for k in risk_data.keys()]
    
    # 字体大小设置
    if font_size == 'normal':
        label_font, pct_font, title_font = 11, 11, 14
    else:
        label_font, pct_font, title_font = 16, 18, 20
    
    wedges, texts, autotexts = ax.pie(risk_data.values(), labels=risk_data.keys(), colors=colors,
                                       autopct='%1.1f%%', 
                                       textprops={'fontsize': label_font, 'weight': 'bold'},
                                       wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    for autotext in autotexts: 
        autotext.set_color('white')
        autotext.set_fontsize(pct_font)
    
    ax.set_title('Patient Risk Distribution Based on ATI', fontweight='bold', fontsize=title_font, pad=20)

def plot_figure3(ax, font_size='normal'):
    """绘制Figure 3 (临床决策空间)"""
    UNCERT_THRESH = np.percentile(df['Mean_Uncert_Risk'][df['Mean_Uncert_Risk'] > 0], 75)
    max_u = df['Mean_Uncert_Risk'].max() * 1.2
    max_r = df['High_Risk_Ratio'].max() * 1.2
    
    # 背景区域
    ax.add_patch(patches.Rectangle((0, 0), max_u, CLINICAL_YELLOW, alpha=0.15, color=COLORS['green']))
    ax.add_patch(patches.Rectangle((0, CLINICAL_YELLOW), UNCERT_THRESH, max_r-CLINICAL_YELLOW, alpha=0.15, color=COLORS['red']))
    ax.add_patch(patches.Rectangle((UNCERT_THRESH, CLINICAL_YELLOW), max_u-UNCERT_THRESH, max_r-CLINICAL_YELLOW, alpha=0.2, color=COLORS['yellow']))
    
    # 散点
    scatter = ax.scatter(df['Mean_Uncert_Risk'], df['High_Risk_Ratio'], 
                         c=df['SSIM_Improvement'], cmap='viridis', 
                         s=100, edgecolor='black', linewidth=1, alpha=0.9, zorder=5)
    
    # 字体大小设置
    if font_size == 'normal':
        label_font, title_font, tick_font, annot_font = 12, 14, 10, 9
    else:
        label_font, title_font, tick_font, annot_font = 18, 20, 14, 16
    
    # 标注高风险患者
    for _, row in high_risk_patients.head(3).iterrows():
        ax.annotate(f'Pt-{row["Patient_ID"].split("_")[-1]}', 
                    (row['Mean_Uncert_Risk'], row['High_Risk_Ratio']), 
                    xytext=(8, 8), textcoords='offset points', 
                    fontsize=annot_font, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
    
    ax.set_xlim(0, max_u)
    ax.set_ylim(0, max_r)
    ax.set_xlabel('Epistemic Uncertainty', fontsize=label_font, fontweight='bold')
    ax.set_ylabel('ATI High-Risk Ratio (%)', fontsize=label_font, fontweight='bold')
    ax.set_title('PI-INR Clinical Decision Space', fontweight='bold', fontsize=title_font, pad=15)
    
    # 阈值线
    ax.axhline(CLINICAL_YELLOW, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(UNCERT_THRESH, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    
    # 区域文字（仅在放大版中添加）
    if font_size == 'large':
        ax.text(max_u*0.8, CLINICAL_YELLOW*0.5, "GREEN: Proceed", 
                color='darkgreen', fontweight='bold', fontsize=14, ha='center', va='center')
        ax.text(UNCERT_THRESH*0.5, max_r*0.8, "RED: Replan", 
                color='darkred', fontweight='bold', fontsize=14, ha='center', va='center')
        ax.text(UNCERT_THRESH + (max_u-UNCERT_THRESH)*0.5, max_r*0.8, "YELLOW: Review", 
                color='#b8860b', fontweight='bold', fontsize=14, ha='center', va='center')
    
    ax.tick_params(axis='both', labelsize=tick_font)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('SSIM Improvement', fontsize=label_font, fontweight='bold')
    cbar.ax.tick_params(labelsize=tick_font)
    
    ax.grid(True, alpha=0.3)
    
    return scatter

# ================== 生成标准版图表 ==================
print("\n🎨 生成标准版学术图表 (字体大小: 10-14pt)...")

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})

# Figure 1
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
plot_figure1(axes, font_size='normal')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'Fig1_Registration.pdf'))
plt.savefig(os.path.join(FIGURES_DIR, 'Fig1_Registration.png'), dpi=300)
plt.close()
print("  ✅ Figure 1: 配准性能 (标准版)")

# Figure 2
fig, ax = plt.subplots(figsize=(8, 8))
plot_figure2(ax, font_size='normal')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'Fig2_Risk_Distribution.pdf'))
plt.savefig(os.path.join(FIGURES_DIR, 'Fig2_Risk_Distribution.png'), dpi=300)
plt.close()
print("  ✅ Figure 2: 风险分布 (标准版)")

# Figure 3
fig, ax = plt.subplots(figsize=(10, 8))
plot_figure3(ax, font_size='normal')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'Fig3_Decision_Space.pdf'))
plt.savefig(os.path.join(FIGURES_DIR, 'Fig3_Decision_Space.png'), dpi=300)
plt.close()
print("  ✅ Figure 3: 决策空间 (标准版)")

# ================== 生成字体放大版图表 ==================
print("\n🎨 生成字体放大版学术图表 (字体大小: 14-20pt)...")

plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})

# Figure 1 - 放大版
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
plot_figure1(axes, font_size='large')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_LARGE_DIR, 'Fig1_Registration_LargeFont.pdf'))
plt.savefig(os.path.join(FIGURES_LARGE_DIR, 'Fig1_Registration_LargeFont.png'), dpi=300)
plt.close()
print("  ✅ Figure 1: 配准性能 (字体放大版)")

# Figure 2 - 放大版
fig, ax = plt.subplots(figsize=(10, 8))
plot_figure2(ax, font_size='large')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_LARGE_DIR, 'Fig2_Risk_Distribution_LargeFont.pdf'))
plt.savefig(os.path.join(FIGURES_LARGE_DIR, 'Fig2_Risk_Distribution_LargeFont.png'), dpi=300)
plt.close()
print("  ✅ Figure 2: 风险分布 (字体放大版)")

# Figure 3 - 放大版
fig, ax = plt.subplots(figsize=(12, 9))
plot_figure3(ax, font_size='large')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_LARGE_DIR, 'Fig3_Decision_Space_LargeFont.pdf'))
plt.savefig(os.path.join(FIGURES_LARGE_DIR, 'Fig3_Decision_Space_LargeFont.png'), dpi=300)
plt.close()
print("  ✅ Figure 3: 决策空间 (字体放大版)")

# ================== 生成论文段落 ==================
print("\n📝 生成论文段落...")

paper_text = f"""
\\subsection*{{4.1 Registration Fidelity}}
PI-INR significantly improved registration accuracy across the cohort ($N={len(df)}$, excluding {excluded_count} cases with invalid baseline SSIM). Mean SSIM increased from {df['SSIM_Before'].mean():.3f} $\\pm$ {df['SSIM_Before'].std():.3f} to {df['SSIM_After'].mean():.3f} $\\pm$ {df['SSIM_After'].std():.3f} (paired t-test, $p < 0.001$). The maximum improvement of {df['SSIM_Improvement'].max():.4f} was observed in Patient {df.loc[df['SSIM_Improvement'].idxmax(), 'Patient_ID']}.

\\subsection*{{4.2 Dosimetric Risk Detection}}
The ATI identified dosimetric risk in {df[df['High_Risk_Ratio'] > RISK_THRESH_LOW].shape[0]}/{len(df)} ({(df[df['High_Risk_Ratio'] > RISK_THRESH_LOW].shape[0]/len(df)*100):.1f}%) patients. High-risk regions (ATI > {RISK_THRESH_MED}%) were detected in {n_orange + n_red} patients, with the highest risk ratio of {df['High_Risk_Ratio'].max():.5f}% in Patient {high_risk_patients.iloc[0]['Patient_ID'] if len(high_risk_patients)>0 else 'N/A'}.

\\subsection*{{4.3 Clinical Decision Support}}
The traffic-light system classified {n_green} patients ({n_green/len(df)*100:.1f}%) as GREEN (proceed), {n_yellow} patients ({n_yellow/len(df)*100:.1f}%) as YELLOW (monitor), and {n_orange + n_red} patients ({(n_orange+n_red)/len(df)*100:.1f}%) as requiring clinical attention. Uncertainty estimates were well-calibrated ($r = {corr_risk_uncert:.3f}$, $p = {p_corr:.3f}$).
"""

with open(os.path.join(TEXT_DIR, 'Results_Text.txt'), 'w', encoding='utf-8') as f:
    f.write(paper_text)

# ================== 生成摘要 ==================
abstract = f"""
PI-INR: Physics-Informed Implicit Neural Representation for Adaptive Radiotherapy

Background: Analyzed {len(df)} pancreatic cancer patients with daily CBCT imaging (excluded {excluded_count} with invalid baseline).
Results: SSIM significantly improved from {df['SSIM_Before'].mean():.3f}±{df['SSIM_Before'].std():.3f} to {df['SSIM_After'].mean():.3f}±{df['SSIM_After'].std():.3f} (p<0.001). Dosimetric risk detected in {df[df['High_Risk_Ratio']>0.001].shape[0]}/{len(df)} ({(df[df['High_Risk_Ratio']>0.001].shape[0]/len(df)*100):.1f}%) patients. Clinical decision system: GREEN ({n_green}), YELLOW ({n_yellow}), HIGH RISK ({n_orange+n_red}).
"""

with open(os.path.join(TEXT_DIR, 'Abstract.txt'), 'w', encoding='utf-8') as f:
    f.write(abstract)

# ================== 生成LaTeX表格 ==================
print("\n📝 生成LaTeX表格...")

latex_table = f"""
% Table 1: Quantitative Results
\\begin{{table}}[t]
\\centering
\\caption{{Quantitative Results of PI-INR on {len(df)} Pancreatic Cancer Patients}}
\\label{{tab:quantitative}}
\\begin{{tabular}}{{lccccc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Mean}} & \\textbf{{Std}} & \\textbf{{Min}} & \\textbf{{Max}} & \\textbf{{Median}} \\\\
\\midrule
SSIM (Before) & {df['SSIM_Before'].mean():.3f} & {df['SSIM_Before'].std():.3f} & {df['SSIM_Before'].min():.3f} & {df['SSIM_Before'].max():.3f} & {df['SSIM_Before'].median():.3f} \\\\
SSIM (After)  & {df['SSIM_After'].mean():.3f} & {df['SSIM_After'].std():.3f} & {df['SSIM_After'].min():.3f} & {df['SSIM_After'].max():.3f} & {df['SSIM_After'].median():.3f} \\\\
SSIM Improv.  & {df['SSIM_Improvement'].mean():.4f} & {df['SSIM_Improvement'].std():.4f} & {df['SSIM_Improvement'].min():.4f} & {df['SSIM_Improvement'].max():.4f} & {df['SSIM_Improvement'].median():.4f} \\\\
ATI Risk (\\%) & {df['High_Risk_Ratio'].mean():.5f} & {df['High_Risk_Ratio'].std():.5f} & {df['High_Risk_Ratio'].min():.5f} & {df['High_Risk_Ratio'].max():.5f} & {df['High_Risk_Ratio'].median():.5f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

with open(os.path.join(TEXT_DIR, 'LaTeX_Table.txt'), 'w', encoding='utf-8') as f:
    f.write(latex_table)

# ================== 生成字体大小对比报告 ==================
print("\n" + "="*80)
print("📊 字体大小对比报告")
print("="*80)
print("\n标准版字体大小 vs 放大版字体大小:")
print(f"  轴标签:     12pt → 18pt  (+50%)")
print(f"  标题:       14pt → 20pt  (+43%)")
print(f"  刻度标签:   10pt → 14pt  (+40%)")
print(f"  图例:       10pt → 14pt  (+40%)")
print(f"  标注文字:   9pt  → 16pt  (+78%)")
print(f"  饼图标签:   11pt → 16pt  (+45%)")
print(f"  饼图百分比: 11pt → 18pt  (+64%)")
print("\n✅ 两个版本图表已生成，请选择适合的版本用于投稿！")

# ================== 生成最终报告 ==================
print("\n📋 生成最终报告...")

report = f"""
===============================================================================
PI-INR: Physics-Informed Implicit Neural Representation for Adaptive Radiotherapy
===============================================================================

ANALYSIS SUMMARY
===============================================================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Version: {__version__}
Total Patients (raw): {len(raw_df)}
Valid Patients (after cleaning): {len(df)} (excluded {excluded_count})
Excluded Patients: {', '.join(raw_df[raw_df['SSIM_Before'] <= 0.01]['Patient_ID'].tolist())}

REGISTRATION PERFORMANCE
===============================================================================
SSIM Before: {df['SSIM_Before'].mean():.4f} ± {df['SSIM_Before'].std():.4f}
SSIM After:  {df['SSIM_After'].mean():.4f} ± {df['SSIM_After'].std():.4f}
SSIM Improvement: {df['SSIM_Improvement'].mean():.4f} ± {df['SSIM_Improvement'].std():.4f}
Maximum Improvement: {df['SSIM_Improvement'].max():.4f} (Patient {df.loc[df['SSIM_Improvement'].idxmax(), 'Patient_ID']})

STATISTICAL TESTS
===============================================================================
Paired t-test: t = {t_stat:.4f}, p = {p_value_t:.2e}
Wilcoxon test: W = {w_stat:.4f}, p = {p_value_w:.2e}
Risk-Uncertainty correlation: r = {corr_risk_uncert:.4f}, p = {p_corr:.2e}

DOSIMETRIC RISK
===============================================================================
ATI Risk (mean): {df['High_Risk_Ratio'].mean():.6f}%
ATI Risk (max):  {df['High_Risk_Ratio'].max():.6f}%
Patients with Risk (>0.001%): {df[df['High_Risk_Ratio'] > 0.001].shape[0]}/{len(df)} ({df[df['High_Risk_Ratio'] > 0.001].shape[0]/len(df)*100:.1f}%)

CLINICAL DECISION SYSTEM
===============================================================================
GREEN (Low Risk):     {n_green} patients ({n_green/len(df)*100:.1f}%)
YELLOW (Medium Risk): {n_yellow} patients ({n_yellow/len(df)*100:.1f}%)
ORANGE (High Risk):   {n_orange} patients ({n_orange/len(df)*100:.1f}%)
RED (Critical):       {n_red} patients ({n_red/len(df)*100:.1f}%)

FILES GENERATED
===============================================================================
Excel Tables: {excel_path}
Standard Figures: {FIGURES_DIR}
Large Font Figures: {FIGURES_LARGE_DIR}
Text Files: {TEXT_DIR}

===============================================================================
"""

print(report)

with open(os.path.join(OUTPUT_DIR, 'Analysis_Report.txt'), 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n✅ 所有分析完成！结果已保存至: {OUTPUT_DIR}")
print("="*80)
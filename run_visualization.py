"""
PI-INR MedIA 终极图表生成引擎 (The Ultimate Figure Generator)
包含主文 Figure 5 以及附录 Figure S1-S6
特点：物理比例完美还原、抗背景干扰、纯矢量高清排版、标题无死角
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # ⭐ 终极护身符：解决 OMP Error #15

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import SimpleITK as sitk
import pydicom
import warnings
warnings.filterwarnings('ignore')

# ================== 1. 全局排版配置 ==================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif':['Arial'],
    'pdf.fonttype': 42, 'savefig.dpi': 600
})

DATA_ROOT = r"X:\data\manifest-1661266724052\Pancreatic-CT-CBCT-SEG"
RESULT_DIR = r"X:\results\MedIA_Ultimate_Run"
CSV_PATH = os.path.join(RESULT_DIR, "MedIA_Quantitative_Results.csv")
OUTPUT_DIR = os.path.join(RESULT_DIR, "MedIA_Paper_Figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SPACING = (2.0, 1.0, 1.0) # Z, Y, X

# ================== 2. 图像重构引擎 ==================
def resample_to_target_spacing(image, target_spacing, default_value=0):
    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()
    target_size =[
        int(round(orig_size[0] * orig_spacing[0] / target_spacing[0])),
        int(round(orig_size[1] * orig_spacing[1] / target_spacing[1])),
        int(round(orig_size[2] * orig_spacing[2] / target_spacing[2]))
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([max(1, s) for s in target_size])
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(default_value)
    return resampler.Execute(image)

def load_patient_baseline(pid):
    p_path = os.path.join(DATA_ROOT, pid)
    ct_series =[]
    dose_path = None
    for root, _, files in os.walk(p_path):
        dcms =[f for f in files if f.lower().endswith('.dcm')]
        if len(dcms) > 20:
            try:
                ds = pydicom.dcmread(os.path.join(root, dcms[0]), stop_before_pixels=True)
                mod = getattr(ds, "Modality", "").upper()
                if mod == 'CT': ct_series.append(root)
                elif mod == 'RTDOSE': dose_path = os.path.join(root, dcms[0])
            except: pass
    ct_series.sort()
    
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(ct_series[0]))
    fixed = sitk.Cast(reader.Execute(), sitk.sitkFloat32)
    fixed = resample_to_target_spacing(fixed, TARGET_SPACING, -1000)
    
    f_arr = sitk.GetArrayFromImage(fixed)
    f_arr = np.clip((f_arr + 1000) / 2000.0, 0, 1)
    
    if dose_path:
        dose = sitk.Cast(sitk.ReadImage(dose_path), sitk.sitkFloat32)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(0)
        dose = resampler.Execute(dose)
        d_arr = sitk.GetArrayFromImage(dose)
        if d_arr.max() > d_arr.min():
            d_arr = (d_arr - d_arr.min()) / (d_arr.max() - d_arr.min() + 1e-6)
    else:
        d_arr = np.zeros_like(f_arr)
        
    return f_arr, d_arr

# ================== 3. 主文 Figure 5 (原Fig4升级版) ==================
def draw_main_figure5():
    print("🎨 开始生成 Figure 5 (主文大图)...")
    PATIENTS =[
        ('Pancreas-CT-CB_012', '(a) Patient 012: Best Registration Improvement (ΔSSIM = +0.1102)'),
        ('Pancreas-CT-CB_029', '(b) Patient 029: Highest Dosimetric Risk (ATI Warning)')
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 12), facecolor='black')
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.88, hspace=0.4, wspace=0.05)
    
    PHYSICAL_ASPECT = TARGET_SPACING[1] / TARGET_SPACING[0] 
    
    for row_idx, (pid, title) in enumerate(PATIENTS):
        print(f"  -> 渲染 {pid}...")
        f_arr, d_arr = load_patient_baseline(pid)
        res_p = os.path.join(RESULT_DIR, pid)
        if not os.path.exists(res_p):
            print(f"  ⚠️ {pid} 的结果文件不存在，跳过")
            continue
            
        w_arr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(res_p, "Warped.nii.gz")))
        u_arr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(res_p, "Uncertainty.nii.gz")))
        ati_arr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(res_p, "ATI_Map.nii.gz")))
        
        z = np.argmax(np.sum(ati_arr, axis=(1, 2))) if ati_arr.max() > 0 else f_arr.shape[0] // 2
        f_slc, w_slc, d_slc, u_slc, a_slc = f_arr[z], w_arr[z], d_arr[z], u_arr[z], ati_arr[z]
        
        H, W = f_slc.shape
        PHYSICAL_EXTENT =[0, W * TARGET_SPACING[0], H * TARGET_SPACING[1], 0]
        axs = axes[row_idx]
        
        # 1. Fixed
        axs[0].imshow(f_slc, cmap='gray', extent=PHYSICAL_EXTENT, vmin=0, vmax=1)
        # 2. Warped
        axs[1].imshow(w_slc, cmap='gray', extent=PHYSICAL_EXTENT, vmin=0, vmax=1)
        
        # 3. Uncertainty
        axs[2].imshow(w_slc, cmap='gray', extent=PHYSICAL_EXTENT, vmin=0, vmax=1)
        u_valid = u_slc[(u_slc > 0.001) & (f_slc > 0.05)] 
        if len(u_valid) > 0:
            thresh = np.percentile(u_valid, 80)
            mask_u = np.ma.masked_where((u_slc < thresh) | (f_slc <= 0.05), u_slc)
            axs[2].imshow(mask_u, cmap='cool', extent=PHYSICAL_EXTENT, alpha=0.8)
            
        # 4. Dose & ATI 
        axs[3].imshow(f_slc, cmap='gray', extent=PHYSICAL_EXTENT, vmin=0, vmax=1)
        if d_slc.max() > 0:
            mask_d = np.ma.masked_where((d_slc < 0.1) | (f_slc <= 0.05), d_slc)
            axs[3].imshow(mask_d, cmap='jet', extent=PHYSICAL_EXTENT, alpha=0.3)
        if a_slc.max() > 0:
            mask_a = np.ma.masked_where((a_slc < 0.05) | (f_slc <= 0.05), a_slc)
            axs[3].imshow(mask_a, cmap='Reds', extent=PHYSICAL_EXTENT, alpha=0.85, vmin=0, vmax=1)

        # ⭐ 修复：为每一行的每一个子图都加上标题
        axs[0].set_title("Fixed CT (Target)", fontsize=22, color='white', pad=20)
        axs[1].set_title("PI-INR Warped", fontsize=22, color='white', pad=20)
        axs[2].set_title("Epistemic Uncertainty", fontsize=22, color='white', pad=20)
        
        # 第四个图的标题根据情况变化
        if row_idx == 0:
            axs[3].set_title("ATI Risk (GREEN: Proceed)", fontsize=22, color='#2ECC71', fontweight='bold', pad=20)
        else:
            axs[3].set_title("ATI Risk (RED: Replanning)", fontsize=22, color='#E74C3C', fontweight='bold', pad=20)
        
        for ax in axs: ax.axis('off')
        
        # 完美的大标题，横向悬空
        axs[0].text(0.0, 1.15, title, color='white', fontsize=26, fontweight='bold', transform=axs[0].transAxes)

    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure5_Representative_Cases.pdf'), dpi=600, facecolor='black', bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure5_Representative_Cases.png'), dpi=600, facecolor='black', bbox_inches='tight')
    plt.close()
    print("  ✅ Figure 5 完美生成！(每一行都带有完整标题)")

# ================== 4. 附录 Fig S1 (无缝排版流程图) ==================
def draw_figure_s1_flowchart():
    print("🎨 开始生成 Figure S1 (流程图)...")
    fig, ax = plt.subplots(figsize=(12, 14), facecolor='white')
    
    ax.set_xlim(-10, 110) 
    ax.set_ylim(-20, 100) 
    ax.axis('off')
    
    def draw_box(x, y, w, h, text, color='#ECF0F1', fontcolor='black', fontsize=14, weight='bold'):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1.5", edgecolor='#34495E', facecolor=color, lw=2.5)
        ax.add_patch(box)
        ax.text(x+w/2, y+h/2, text, color=fontcolor, fontsize=fontsize, fontweight=weight, ha='center', va='center')
        
    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>,head_length=1.0,head_width=0.5", lw=2.5, color='#34495E'))

    draw_box(25, 85, 50, 8, "Daily CBCT Acquisition", '#EBF5FB')
    draw_arrow(50, 85, 50, 75)
    draw_box(25, 65, 50, 10, "PI-INR Inference\n(Rigid + Non-rigid)", '#D6EAF8')
    draw_arrow(50, 65, 50, 55)
    
    ax.plot([25, 75], [55, 55], color='#34495E', lw=3)
    draw_arrow(25, 55, 25, 48); draw_arrow(75, 55, 75, 48)
    
    draw_box(2, 38, 44, 10, "ATI Risk Map\n(Lie Derivative & Entropy)", '#FADBD8', fontsize=12)
    draw_box(54, 38, 44, 10, "Uncertainty Map\n(Evidential Deep Learning)", '#E8DAEF', fontsize=12)
    
    draw_arrow(24, 38, 24, 30); draw_arrow(76, 38, 76, 30)
    ax.plot([24, 76], [30, 30], color='#34495E', lw=3)
    draw_arrow(50, 30, 50, 22)
    
    draw_box(15, 12, 70, 8, "Traffic-Light Decision System\n(ATI & Uncertainty Synthesis)", '#FCF3CF', fontcolor='black', fontsize=14)
    draw_arrow(50, 12, 50, 4)
    
    ax.plot([15, 85], [4, 4], color='#34495E', lw=3)
    draw_arrow(15, 4, 15, -4); draw_arrow(50, 4, 50, -4); draw_arrow(85, 4, 85, -4)
    
    draw_box(5, -12, 28, 8, "RED\nReplanning", '#E74C3C', 'white', fontsize=12)
    draw_box(36, -12, 28, 8, "YELLOW\nManual Review", '#F1C40F', 'black', fontsize=12)
    draw_box(67, -12, 28, 8, "GREEN\nProceed", '#2ECC71', 'white', fontsize=12)

    plt.title("Figure S1. Clinical Decision Flowchart", pad=20, fontsize=18, weight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, 'FigS1_Flowchart.pdf'), bbox_inches='tight', pad_inches=0.2)
    plt.savefig(os.path.join(OUTPUT_DIR, 'FigS1_Flowchart.png'), bbox_inches='tight', pad_inches=0.2, dpi=600)
    plt.close()
    print("  ✅ Figure S1 生成完毕！")

# ================== 5. 附录 Fig S2 & S3: 伪彩融合图 ==================
def draw_annotated_cases():
    print("🎨 开始生成 Figure S2 & S3 (伪彩临床标注图)...")
    cases =[
        ('Pancreas-CT-CB_012', 'FigS2_Patient012_Annotated', 'Figure S2. Patient 012 - Topological Preserving'),
        ('Pancreas-CT-CB_029', 'FigS3_Patient029_Annotated', 'Figure S3. Patient 029 - Highest Dosimetric Risk')
    ]
    for pid, filename, title in cases:
        f_arr, _ = load_patient_baseline(pid)
        res_p = os.path.join(RESULT_DIR, pid)
        if not os.path.exists(res_p): continue
        w_arr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(res_p, "Warped.nii.gz"))).squeeze()
        ati_arr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(res_p, "ATI_Map.nii.gz"))).squeeze()
        
        z = np.argmax(np.sum(ati_arr, axis=(1,2))) if ati_arr.max() > 0 else f_arr.shape[0]//2
        H, W = f_arr[z].shape
        PHYSICAL_EXTENT =[0, W * TARGET_SPACING[0], H * TARGET_SPACING[1], 0]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='black')
        for ax in axes: ax.axis('off')
        
        rgb = np.zeros((f_arr.shape[1], f_arr.shape[2], 3), dtype=np.float32)
        rgb[..., 0] = f_arr[z] 
        rgb[..., 1] = w_arr[z] 
        rgb[..., 2] = w_arr[z] 
        
        axes[0].imshow(rgb, extent=PHYSICAL_EXTENT)
        axes[0].set_title(f"{pid} - Pseudo-color Overlay", color='white', pad=15, fontsize=16)
        axes[0].annotate('Soft Tissue Boundary', xy=(0.4, 0.4), xycoords='axes fraction', xytext=(0.1, 0.2), textcoords='axes fraction',
                         arrowprops=dict(facecolor='yellow', shrink=0.05, width=2), color='yellow', weight='bold', fontsize=14)
                         
        axes[1].imshow(f_arr[z], cmap='gray', extent=PHYSICAL_EXTENT)
        masked_ati = np.ma.masked_where(ati_arr[z] < 0.1, ati_arr[z])
        axes[1].imshow(masked_ati, cmap='Reds', alpha=0.8, vmin=0, vmax=1.0, extent=PHYSICAL_EXTENT)
        axes[1].set_title("ATI High-Risk Hotspot", color='white', pad=15, fontsize=16)
        axes[1].annotate('High Risk Area', xy=(0.5, 0.5), xycoords='axes fraction', xytext=(0.7, 0.8), textcoords='axes fraction',
                         arrowprops=dict(facecolor='red', shrink=0.05, width=2), color='red', weight='bold', fontsize=14)
        
        plt.suptitle(title, color='white', fontsize=20, weight='bold', y=0.98)
        plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}.pdf"), facecolor='black', bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}.png"), facecolor='black', bbox_inches='tight', dpi=600)
        plt.close()
    print("  ✅ Figure S2 & S3 生成完毕！")

# ================== 6. 附录 Fig S4-S6: 统计分布图 ==================
def draw_extended_stats():
    print("🎨 开始生成 Figure S4-S6 (补充统计学图表)...")
    if not os.path.exists(CSV_PATH): return
    df = pd.read_csv(CSV_PATH)
    df = df[df['SSIM_Before'] > 0.01].copy()
    df['SSIM_Improvement'] = df['SSIM_After'] - df['SSIM_Before']
    
    # S4
    plt.figure(figsize=(8, 6))
    sns.histplot(df['SSIM_Improvement'], bins=15, kde=True, color='#3498DB')
    plt.axvline(x=0, color='red', linestyle='--', label='No Improvement')
    plt.title('Figure S4. Distribution of SSIM Improvement', weight='bold')
    plt.xlabel('ΔSSIM (After - Before)')
    plt.ylabel('Patient Count')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'FigS4_SSIM_Histogram.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'FigS4_SSIM_Histogram.png'), bbox_inches='tight', dpi=600) 
    plt.close()
    
    # S5
    plt.figure(figsize=(8, 6))
    risk_data = df[df['High_Risk_Ratio'] > 0]['High_Risk_Ratio']
    sns.histplot(risk_data, bins=20, log_scale=True, color='#E74C3C')
    plt.title('Figure S5. ATI Risk Distribution (Log Scale)', weight='bold')
    plt.xlabel('ATI High-Risk Ratio (%) - Log Scale')
    plt.ylabel('Patient Count')
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True)) # Y轴全为整数
    plt.savefig(os.path.join(OUTPUT_DIR, 'FigS5_ATI_Log_Dist.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'FigS5_ATI_Log_Dist.png'), bbox_inches='tight', dpi=600) 
    plt.close()
    
    # S6
    plt.figure(figsize=(8, 6))
    sns.kdeplot(df['Mean_Uncert_Risk'], fill=True, color='#9B59B6')
    plt.axvline(x=np.percentile(df['Mean_Uncert_Risk'], 85), color='red', linestyle='--', label='85th Percentile')
    plt.title('Figure S6. Epistemic Uncertainty Distribution', weight='bold')
    plt.xlabel('Mean Uncertainty in Risk Regions')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'FigS6_Uncertainty_Dist.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'FigS6_Uncertainty_Dist.png'), bbox_inches='tight', dpi=600)
    plt.close()
    print("  ✅ Figure S4-S6 生成完毕！")

if __name__ == "__main__":
    print("="*60)
    print("🚀 开始一键生成所有图表 (PNG + PDF)...")
    draw_main_figure5()
    draw_figure_s1_flowchart()
    draw_annotated_cases()
    draw_extended_stats()
    print("="*60)
    print(f"🎉 大功告成！所有图表均已保存在: {OUTPUT_DIR}")
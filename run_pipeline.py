"""
PI-INR MedIA Ultimate: 物理-几何融合剂量预警系统
版本：终极版 v2.2 (生产级稳定版)
特点：拓扑保持、黎曼流形、李导数熵、EDL不确定性、红黄绿灯决策
优化：显存自动清理、分辨率统一、预扫描校验
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import pydicom
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import time
import warnings
import traceback
import random
import gc
warnings.filterwarnings('ignore')

# ================== 0. 论文环境配置 ==================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"

CONFIG = {
    "DATA_ROOT": r"./data/Pancreatic-CT-CBCT-SEG",
    "RESULT_DIR": r"./results/MedIA_Ultimate_Run",
    "CSV_PATH": r"./results/MedIA_Ultimate_Run/MedIA_Quantitative_Results.csv",
    "PRE_SCAN_PATH": r"./results/MedIA_Ultimate_Run/patient_status.csv",
    "LOG_FILE": r"./results/MedIA_Ultimate_Run/batch_log.txt",
    "DEVICE": 'cuda' if torch.cuda.is_available() else 'cpu',
    "RANDOM_SEED": 42,
    "BATCH_POINTS": 15000,
    "EPOCHS": 1200,
    "LR": 2e-4,
    "LAMBDA_EDL": 1.0,
    "LAMBDA_PHYS": 0.05,
    "LAMBDA_FOLD": 10.0,
    "GAMMA_METRIC": 10.0,
    "ATI_THRESHOLD": 0.6,
    "MAX_TEST_PATIENTS": 100,  # 改为实际病人数
    "FORCE_RERUN": True,
    "DELETE_OLD_RESULTS": False,  # 改为False避免误删
    "SAVE_INTERVAL": 5,
    "TARGET_SPACING": (2.0, 1.0, 1.0),  # ⭐ 统一分辨率 (z, y, x) mm
    "LOG_FILE": r"X:\results\MedIA_Ultimate_Run\batch_log.txt"  # 日志文件
}
os.makedirs(CONFIG["RESULT_DIR"], exist_ok=True)

# ================== 配置日志 ==================
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG["LOG_FILE"], encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================== 固定随机种子 ==================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(CONFIG["RANDOM_SEED"])

logger.info("="*80)
logger.info(" PI-INR MedIA Ultimate v2.2 (生产级稳定版)")
logger.info("="*80)
logger.info(f"设备: {CONFIG['DEVICE']}")
logger.info(f"随机种子: {CONFIG['RANDOM_SEED']}")
logger.info(f"统一分辨率: {CONFIG['TARGET_SPACING']} mm")
logger.info(f"防折叠权重: λ_fold = {CONFIG['LAMBDA_FOLD']}")
logger.info("="*80)

# ================== 1. 预扫描脚本 ==================
def pre_scan_patients():
    """扫描所有患者，检查数据完整性"""
    logger.info("\n🔍 开始预扫描所有患者...")
    
    if not os.path.exists(CONFIG["DATA_ROOT"]):
        logger.error(f"数据根目录不存在: {CONFIG['DATA_ROOT']}")
        return []
    
    all_patients = sorted([p for p in os.listdir(CONFIG["DATA_ROOT"]) 
                          if p.startswith("Pancreas") and os.path.isdir(os.path.join(CONFIG["DATA_ROOT"], p))])
    
    status_list = []
    
    for pid in tqdm(all_patients, desc="预扫描进度"):
        patient_path = os.path.join(CONFIG["DATA_ROOT"], pid)
        status = {
            "Patient_ID": pid,
            "CT_Series": 0,
            "Has_Dose": False,
            "Dose_Valid": False,
            "Size_Consistent": False,
            "Status": "Unknown"
        }
        
        # 扫描CT系列
        ct_series = []
        dose_path = None
        
        for root, dirs, files in os.walk(patient_path):
            dcm_files = [f for f in files if f.lower().endswith('.dcm')]
            if not dcm_files:
                continue
                
            try:
                first_dcm = os.path.join(root, dcm_files[0])
                ds = pydicom.dcmread(first_dcm, stop_before_pixels=True)
                modality = getattr(ds, "Modality", "").upper()
                
                if modality == 'CT' and len(dcm_files) > 20:
                    ct_series.append(root)
                elif modality == 'RTDOSE':
                    dose_path = first_dcm
            except:
                continue
        
        status["CT_Series"] = len(ct_series)
        status["Has_Dose"] = dose_path is not None
        
        # 检查剂量有效性
        if dose_path and os.path.exists(dose_path):
            try:
                dose_img = sitk.ReadImage(dose_path)
                dose_arr = sitk.GetArrayFromImage(dose_img)
                status["Dose_Valid"] = dose_arr.max() > dose_arr.min()
            except:
                pass
        
        # 判断整体状态
        if status["CT_Series"] >= 2 and status["Dose_Valid"]:
            status["Status"] = "Ready"
        elif status["CT_Series"] >= 2:
            status["Status"] = "Missing Dose"
        else:
            status["Status"] = "Incomplete"
        
        status_list.append(status)
    
    # 保存扫描结果
    df_status = pd.DataFrame(status_list)
    df_status.to_csv(CONFIG["PRE_SCAN_PATH"], index=False, encoding='utf-8-sig')
    
    # 打印统计
    logger.info(f"\n📊 预扫描完成，共 {len(status_list)} 例患者:")
    logger.info(f"   ✅ Ready (可处理): {len(df_status[df_status['Status']=='Ready'])} 例")
    logger.info(f"   ⚠️ Missing Dose (缺剂量): {len(df_status[df_status['Status']=='Missing Dose'])} 例")
    logger.info(f"   ❌ Incomplete (数据不全): {len(df_status[df_status['Status']=='Incomplete'])} 例")
    logger.info(f"\n预扫描结果已保存至: {CONFIG['PRE_SCAN_PATH']}")
    
    return status_list

# ================== 2. 网络定义 ==================
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=10.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / in_features) / self.omega_0, 
                                            np.sqrt(6 / in_features) / self.omega_0)
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class RiemannianSirenNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(3, 128, is_first=True, omega_0=10.0),
            SineLayer(128, 128, omega_0=10.0),
            SineLayer(128, 128, omega_0=10.0),
            nn.Linear(128, 6)
        )
        with torch.no_grad():
            self.net[-1].weight.fill_(0)
            self.net[-1].bias.fill_(0)

    def forward(self, x):
        out = self.net(x)
        disp = torch.tanh(out[..., :3]) * 0.05
        v = F.softplus(out[..., 3:4]) + 1e-6
        alpha = F.softplus(out[..., 4:5]) + 1.1
        beta = F.softplus(out[..., 5:6]) + 1e-6
        return disp, v.squeeze(-1), alpha.squeeze(-1), beta.squeeze(-1)

def compute_analytical_jacobian(y, x):
    jac = []
    for i in range(3):
        grad_outputs = torch.ones_like(y[:, i])
        grad = torch.autograd.grad(y[:, i], x, grad_outputs=grad_outputs, 
                                   create_graph=True, retain_graph=True)[0]
        jac.append(grad)
    return torch.stack(jac, dim=1)

# ================== 3. 李导数熵计算 ==================
def compute_lie_entropy(dose_tensor, disp_field, spacing):
    sp_z, sp_y, sp_x = spacing
    
    d_pad = F.pad(dose_tensor, (1, 1, 1, 1, 1, 1), mode='replicate')
    
    dx = (d_pad[..., 1:-1, 1:-1, 2:] - d_pad[..., 1:-1, 1:-1, :-2]) / (2.0 * sp_x)
    dy = (d_pad[..., 1:-1, 2:, 1:-1] - d_pad[..., 1:-1, :-2, 1:-1]) / (2.0 * sp_y)
    dz = (d_pad[..., 2:, 1:-1, 1:-1] - d_pad[..., :-2, 1:-1, 1:-1]) / (2.0 * sp_z)
    
    grad_D = torch.stack([dx, dy, dz], dim=-1)
    
    disp_expanded = disp_field.unsqueeze(1)
    lie_deriv = torch.sum(disp_expanded * grad_D, dim=-1)
    
    kernel_size = 5
    mean_lie = F.avg_pool3d(lie_deriv, kernel_size, stride=1, 
                           padding=kernel_size//2, count_include_pad=False)
    mean_sq_lie = F.avg_pool3d(lie_deriv**2, kernel_size, stride=1, 
                              padding=kernel_size//2, count_include_pad=False)
    entropy_map = torch.abs(mean_sq_lie - mean_lie**2)
    
    eps = 1e-6
    ati_map = (entropy_map - entropy_map.min()) / (entropy_map.max() - entropy_map.min() + eps)
    
    return ati_map, lie_deriv

# ================== 4. 临床决策系统 ==================
def evaluate_clinical_decision(ati_arr, uncert_arr):
    high_risk_mask = ati_arr > CONFIG["ATI_THRESHOLD"]
    high_risk_ratio = np.mean(high_risk_mask) * 100
    
    if np.sum(high_risk_mask) > 0:
        mean_uncert_in_risk = np.mean(uncert_arr[high_risk_mask])
    else:
        mean_uncert_in_risk = 0.0
    
    if uncert_arr.max() > 0:
        uncert_threshold = np.percentile(uncert_arr, 85)
    else:
        uncert_threshold = 0.5
    
    if high_risk_ratio > 5.0 and mean_uncert_in_risk < uncert_threshold:
        decision = ("RED (Replanning)", "High dosimetric decay detected, immediate replanning recommended.")
        color_class = "red"
    elif high_risk_ratio > 5.0 and mean_uncert_in_risk >= uncert_threshold:
        decision = ("YELLOW (Manual Review)", "High risk detected but AI uncertain, manual review required.")
        color_class = "yellow"
    else:
        decision = ("GREEN (Proceed)", "No significant dosimetric decay detected, continue treatment.")
        color_class = "green"
        
    return high_risk_ratio, mean_uncert_in_risk, decision, color_class

def generate_html_report(pid, stats, save_path):
    decision_color = stats['Color_Class']
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>PI-INR Digital Twin Report - {pid}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background-color: #f5f5f5; margin: 0; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 30px; border-radius: 15px; margin-bottom: 30px; }}
        .decision-box {{ padding: 25px; border-radius: 15px; margin-bottom: 30px; text-align: center; }}
        .red {{ background: linear-gradient(135deg, #ff4d4d, #cc0000); color: white; }}
        .yellow {{ background: linear-gradient(135deg, #ffcc00, #ff9900); color: black; }}
        .green {{ background: linear-gradient(135deg, #33cc33, #009900); color: white; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stat-value {{ font-size: 32px; font-weight: bold; color: #3498db; }}
        .image-container {{ background: white; padding: 20px; border-radius: 15px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>PI-INR Digital Twin: 剂量学衰退预警系统</h1>
            <p>患者ID: {pid} | 分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="decision-box {decision_color}">
            <h2>{stats['Decision'][0]}</h2>
            <p>{stats['Decision'][1]}</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>高风险区域占比</h3>
                <div class="stat-value">{stats['High_Risk_Ratio']:.2f}%</div>
            </div>
            <div class="stat-card">
                <h3>高风险区不确定性</h3>
                <div class="stat-value">{stats['Mean_Uncert_Risk']:.4f}</div>
            </div>
            <div class="stat-card">
                <h3>配准后SSIM</h3>
                <div class="stat-value">{stats['SSIM_After']:.4f}</div>
            </div>
        </div>
        
        <div class="image-container">
            <h3>临床证据图</h3>
            <img src="Paper_Figure.png" style="width:100%">
        </div>
    </div>
</body>
</html>
    """
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

# ================== 5. 数据加载与预处理（带分辨率统一） ==================
def get_robust_physical_center(img):
    size = img.GetSize()
    center_idx = [s/2.0 for s in size]
    return np.array(img.TransformContinuousIndexToPhysicalPoint(center_idx))

def resample_to_target_spacing(image, target_spacing, default_value=0):
    """
    将图像重采样到目标分辨率
    target_spacing: (sp_z, sp_y, sp_x) in mm
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    original_origin = image.GetOrigin()
    original_direction = image.GetDirection()
    
    # 计算目标尺寸
    target_size = [
        int(round(original_size[0] * original_spacing[0] / target_spacing[0])),
        int(round(original_size[1] * original_spacing[1] / target_spacing[1])),
        int(round(original_size[2] * original_spacing[2] / target_spacing[2]))
    ]
    
    # 确保至少为1
    target_size = [max(1, s) for s in target_size]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(original_origin)
    resampler.SetOutputDirection(original_direction)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(default_value)
    
    return resampler.Execute(image)

def create_gaussian_dose(shape, device='cpu'):
    D, H, W = shape
    z = np.linspace(-1, 1, D)
    y = np.linspace(-1, 1, H)
    x = np.linspace(-1, 1, W)
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    
    r2 = X**2 + Y**2 + Z**2
    dose = np.exp(-r2 / 0.5)
    dose = (dose - dose.min()) / (dose.max() - dose.min() + 1e-6)
    
    return torch.from_numpy(dose.astype(np.float32)).to(device)

def load_patient_data(patient_path):
    logger.info(f"\n📂 扫描患者目录: {patient_path}")
    
    ct_series = []
    dose_path = None
    dose_alternative = None
    
    for root, dirs, files in os.walk(patient_path):
        dcm_files = [f for f in files if f.lower().endswith('.dcm')]
        if not dcm_files:
            continue
            
        try:
            first_dcm = os.path.join(root, dcm_files[0])
            ds = pydicom.dcmread(first_dcm, stop_before_pixels=True)
            modality = getattr(ds, "Modality", "").upper()
            
            if modality == 'CT' and len(dcm_files) > 20:
                ct_series.append(root)
                logger.info(f"  ✅ 找到CT系列: {os.path.basename(root)} ({len(dcm_files)}张)")
            elif modality == 'RTDOSE':
                dose_path = first_dcm
                logger.info(f"  ✅ 找到RTDOSE: {os.path.basename(root)}")
            elif 'dose' in root.lower():
                dose_alternative = root
                logger.info(f"  ⚠️ 找到备选剂量文件夹: {os.path.basename(root)}")
        except Exception as e:
            continue
    
    ct_series.sort()
    
    if len(ct_series) < 2:
        logger.error(f"❌ CT数据不足: 需要至少2个CT系列，找到 {len(ct_series)}个")
        return None
    
    # 处理剂量数据
    use_dummy_dose = False
    dose_image = None
    
    if dose_path and os.path.exists(dose_path):
        try:
            dose_image = sitk.ReadImage(dose_path)
            dose_array = sitk.GetArrayFromImage(dose_image)
            if dose_array.max() <= dose_array.min():
                logger.warning(f"⚠️ 剂量数据无效 (全零)，将使用虚拟剂量")
                use_dummy_dose = True
            else:
                logger.info(f"  ✅ 剂量数据有效，范围 [{dose_array.min():.3f}, {dose_array.max():.3f}]")
        except Exception as e:
            logger.warning(f"⚠️ 读取剂量失败: {e}")
            use_dummy_dose = True
    elif dose_alternative:
        try:
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(reader.GetGDCMSeriesFileNames(dose_alternative))
            dose_image = reader.Execute()
            logger.info(f"  ✅ 从备选文件夹读取剂量成功")
            use_dummy_dose = False
        except:
            use_dummy_dose = True
    else:
        logger.warning(f"⚠️ 未找到剂量文件，将使用虚拟剂量")
        use_dummy_dose = True
    
    # 读取CT图像
    try:
        reader = sitk.ImageSeriesReader()
        
        logger.info(f"📥 读取固定CT (计划CT)...")
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(ct_series[0]))
        fixed = sitk.Cast(reader.Execute(), sitk.sitkFloat32)
        
        logger.info(f"📥 读取移动CT (治疗日CBCT)...")
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(ct_series[1]))
        moving = sitk.Cast(reader.Execute(), sitk.sitkFloat32)
        
    except Exception as e:
        logger.error(f"❌ 读取CT失败: {e}")
        return None
    
    # 物理对齐
    logger.info("🔄 执行物理对齐...")
    fixed_center = get_robust_physical_center(fixed)
    moving_center = get_robust_physical_center(moving)
    translation = moving_center - fixed_center
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetTransform(sitk.TranslationTransform(3, translation))
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000)
    moving_aligned = resampler.Execute(moving)
    
    # 处理剂量
    if use_dummy_dose:
        size = fixed.GetSize()
        dose_array = create_gaussian_dose((size[2], size[1], size[0])).cpu().numpy()
        dose_image = sitk.GetImageFromArray(dose_array)
        dose_image.CopyInformation(fixed)
        logger.info(f"  ✅ 创建虚拟剂量，范围 [{dose_array.min():.3f}, {dose_array.max():.3f}]")
    else:
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(0)
        dose_image = resampler.Execute(dose_image)
    
    # ⭐⭐⭐ 统一分辨率到目标spacing ⭐⭐⭐
    logger.info(f"🔄 统一分辨率到 {CONFIG['TARGET_SPACING']} mm...")
    fixed = resample_to_target_spacing(fixed, CONFIG["TARGET_SPACING"], -1000)
    moving_aligned = resample_to_target_spacing(moving_aligned, CONFIG["TARGET_SPACING"], -1000)
    dose_image = resample_to_target_spacing(dose_image, CONFIG["TARGET_SPACING"], 0)
    
    # 转换为张量
    def to_tensor(image, is_dose=False):
        arr = sitk.GetArrayFromImage(image).astype(np.float32)
        
        if is_dose:
            if arr.max() > arr.min():
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
            else:
                arr = np.zeros_like(arr)
        else:
            arr = np.clip((arr + 1000) / 2000.0, 0, 1)
        
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    
    fixed_tensor = to_tensor(fixed, False).to(CONFIG["DEVICE"])
    moving_tensor = to_tensor(moving_aligned, False).to(CONFIG["DEVICE"])
    dose_tensor = to_tensor(dose_image, True).to(CONFIG["DEVICE"])
    
    logger.info(f"✅ 数据加载完成:")
    logger.info(f"   Fixed CT: {fixed_tensor.shape}, 范围 [{fixed_tensor.min():.3f}, {fixed_tensor.max():.3f}]")
    logger.info(f"   Moving CT: {moving_tensor.shape}, 范围 [{moving_tensor.min():.3f}, {moving_tensor.max():.3f}]")
    logger.info(f"   Dose: {dose_tensor.shape}, 范围 [{dose_tensor.min():.3f}, {dose_tensor.max():.3f}]")
    
    return fixed_tensor, moving_tensor, dose_tensor, fixed

# ================== 6. 核心处理函数 ==================
def process_patient(pid, force_rerun=True):
    save_path = os.path.join(CONFIG["RESULT_DIR"], pid)
    os.makedirs(save_path, exist_ok=True)
    
    # 检查是否已处理
    if not force_rerun and os.path.exists(os.path.join(save_path, "Paper_Figure.png")):
        logger.info(f"⏩ {pid} 已处理，跳过")
        return None
    
    patient_path = os.path.join(CONFIG["DATA_ROOT"], pid)
    if not os.path.exists(patient_path):
        logger.error(f"❌ 患者目录不存在: {patient_path}")
        return None
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🔥 处理患者: {pid}")
    logger.info(f"{'='*60}")
    
    # 加载数据
    data = load_patient_data(patient_path)
    if data is None:
        logger.error(f"❌ {pid} 数据加载失败")
        return None
    
    fixed_t, moving_t, dose_t, ref_image = data
    
    # 计算剂量梯度
    try:
        dz, dy, dx = torch.gradient(dose_t.squeeze(), spacing=[1.0, 1.0, 1.0])
        grad_dose = torch.stack([dx, dy, dz], dim=0).unsqueeze(0).to(CONFIG["DEVICE"])
        logger.info(f"✅ 剂量梯度计算成功，形状: {grad_dose.shape}")
    except Exception as e:
        logger.warning(f"⚠️ 剂量梯度计算失败: {e}")
        D, H, W = fixed_t.shape[2:]
        grad_dose = torch.zeros(1, 3, D, H, W, device=CONFIG["DEVICE"])
    
    # 初始化网络
    model = RiemannianSirenNet().to(CONFIG["DEVICE"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["LR"])
    
    # 训练
    logger.info(f"\n🏋️ 开始训练 {CONFIG['EPOCHS']} 轮...")
    for epoch in tqdm(range(CONFIG["EPOCHS"]), desc="Training"):
        optimizer.zero_grad()
        
        coords = torch.rand(CONFIG["BATCH_POINTS"], 3, device=CONFIG["DEVICE"]) * 2 - 1
        coords.requires_grad_(True)
        
        disp, v, alpha, beta = model(coords)
        
        fixed_sampled = F.grid_sample(
            fixed_t, 
            coords.view(1, 1, 1, -1, 3), 
            align_corners=True, 
            mode='bilinear'
        ).view(-1)
        
        moving_sampled = F.grid_sample(
            moving_t, 
            (coords + disp).view(1, 1, 1, -1, 3), 
            align_corners=True, 
            mode='bilinear'
        ).view(-1)
        
        # EDL损失
        mask = (fixed_sampled > 0.05).float()
        squared_error = (fixed_sampled - moving_sampled) ** 2
        edl_loss = torch.mean(mask * (squared_error * v + (2*alpha + v)/(2*alpha*v)))
        
        # 物理正则项
        try:
            J = compute_analytical_jacobian(disp, coords)
            
            sampled_grad = F.grid_sample(
                grad_dose, 
                coords.view(1, 1, 1, -1, 3), 
                align_corners=True
            ).view(3, -1).T
            metric_weight = 1.0 + CONFIG["GAMMA_METRIC"] * torch.sum(sampled_grad**2, dim=-1)
            phys_loss = torch.mean(metric_weight * torch.sum(J**2, dim=(1,2)))
            
            # 防折叠惩罚
            I_mat = torch.eye(3, device=CONFIG["DEVICE"]).unsqueeze(0)
            F_mat = I_mat + J
            det_F = torch.det(F_mat)
            folding_loss = F.relu(-det_F + 1e-5).mean()
            
            phys_loss_total = phys_loss + CONFIG["LAMBDA_FOLD"] * folding_loss
            
        except Exception as e:
            phys_loss_total = torch.tensor(0.0, device=CONFIG["DEVICE"])
            folding_loss = torch.tensor(0.0, device=CONFIG["DEVICE"])
        
        total_loss = CONFIG["LAMBDA_EDL"] * edl_loss + CONFIG["LAMBDA_PHYS"] * phys_loss_total
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            logger.info(f"  Epoch {epoch+1}: loss={total_loss.item():.6f}, "
                        f"edl={edl_loss.item():.6f}, phys={phys_loss.item():.6f}, "
                        f"fold={folding_loss.item():.6f}")
        
        if (epoch + 1) % 100 == 0 and CONFIG["DEVICE"] == 'cuda':
            torch.cuda.empty_cache()
    
    # ========== 推理与后处理 ==========
    logger.info("\n🔄 生成全尺寸形变场和预警图...")
    model.eval()
    D, H, W = fixed_t.shape[2:]
    
    warped = np.zeros((D, H, W), dtype=np.float32)
    uncertainty = np.zeros((D, H, W), dtype=np.float32)
    displacement = np.zeros((D, H, W, 3), dtype=np.float32)
    
    with torch.no_grad():
        for z in tqdm(range(D), desc="Slice-wise Inference"):
            z_norm = (z / (D-1)) * 2 - 1
            
            y_coords = torch.linspace(-1, 1, H, device=CONFIG["DEVICE"])
            x_coords = torch.linspace(-1, 1, W, device=CONFIG["DEVICE"])
            gy, gx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            grid = torch.stack([gx, gy, torch.full_like(gx, z_norm)], dim=-1)
            
            disp_slice, v_slice, alpha_slice, beta_slice = model(grid.view(-1, 3))
            disp_slice = disp_slice.view(H, W, 3)
            
            grid_disp = (grid + disp_slice).unsqueeze(0).unsqueeze(0)
            warped_slice = F.grid_sample(
                moving_t[:, :, z:z+1], 
                grid_disp, 
                align_corners=True, 
                mode='bilinear'
            )
            warped[z] = warped_slice.squeeze().cpu().numpy()
            
            uncert_slice = (beta_slice / (v_slice * (alpha_slice - 1))).view(H, W).cpu().numpy()
            uncert_slice = np.nan_to_num(uncert_slice, nan=0.0, posinf=0.0, neginf=0.0)
            uncertainty[z] = uncert_slice
            
            displacement[z] = disp_slice.cpu().numpy()
    
    # 计算ATI
    logger.info("⚠️ 计算ATI预警指数...")
    dose_np = dose_t.squeeze().cpu().numpy()
    spacing = ref_image.GetSpacing()
    
    dose_tensor = torch.from_numpy(dose_np).unsqueeze(0).unsqueeze(0).float().to(CONFIG["DEVICE"])
    disp_tensor = torch.from_numpy(displacement).unsqueeze(0).float().to(CONFIG["DEVICE"])
    
    try:
        ati_tensor, lie_deriv_tensor = compute_lie_entropy(dose_tensor, disp_tensor, spacing)
        ati = ati_tensor.squeeze().cpu().numpy()
        ati = np.nan_to_num(ati, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        logger.warning(f"⚠️ ATI计算失败: {e}")
        ati = np.zeros_like(dose_np)
    
    # 临床决策
    logger.info("\n📊 生成临床决策...")
    hr_ratio, u_risk, decision, color_class = evaluate_clinical_decision(ati, uncertainty)
    
    # 选择代表性切片
    if np.sum(ati) > 0:
        z_slice = np.argmax(np.sum(ati, axis=(1, 2)))
    else:
        z_slice = D // 2
    
    # 可视化
    plt.figure(figsize=(20, 5), facecolor='black')
    
    ax1 = plt.subplot(1, 4, 1)
    ax1.imshow(fixed_t.squeeze().cpu().numpy()[z_slice], cmap='gray', vmin=0, vmax=1)
    ax1.set_title("Fixed CT (Plan)", color='white', fontsize=14)
    ax1.axis('off')
    
    ax2 = plt.subplot(1, 4, 2)
    ax2.imshow(warped[z_slice], cmap='gray', vmin=0, vmax=1)
    ax2.set_title("PI-INR Warped", color='white', fontsize=14)
    ax2.axis('off')
    
    ax3 = plt.subplot(1, 4, 3)
    ax3.imshow(warped[z_slice], cmap='gray', vmin=0, vmax=1, alpha=0.5)
    if uncertainty[z_slice].max() > 0:
        uncert_thresh = np.percentile(uncertainty[z_slice], 90)
        mask_uncert = np.ma.masked_where(uncertainty[z_slice] < uncert_thresh, uncertainty[z_slice])
        ax3.imshow(mask_uncert, cmap='cool', alpha=0.8)
    ax3.set_title("Epistemic Uncertainty", color='white', fontsize=14)
    ax3.axis('off')
    
    ax4 = plt.subplot(1, 4, 4)
    ax4.imshow(fixed_t.squeeze().cpu().numpy()[z_slice], cmap='gray', vmin=0, vmax=1, alpha=0.5)
    if dose_np[z_slice].max() > 0:
        ax4.imshow(dose_np[z_slice], cmap='jet', alpha=0.2, vmin=0, vmax=1)
    if ati[z_slice].max() > 0:
        mask_ati = np.ma.masked_where(ati[z_slice] < CONFIG["ATI_THRESHOLD"], ati[z_slice])
        ax4.imshow(mask_ati, cmap='Reds', alpha=0.8, vmin=0, vmax=1)
    ax4.set_title(f"ATI Risk ({decision[0]})", color='white', fontsize=14)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "Paper_Figure.png"), facecolor='black', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 计算SSIM
    try:
        fixed_slice = fixed_t.squeeze().cpu().numpy()[z_slice]
        moving_slice = moving_t.squeeze().cpu().numpy()[z_slice]
        warped_slice = warped[z_slice]
        
        if (fixed_slice.size == moving_slice.size and fixed_slice.size == warped_slice.size and 
            fixed_slice.std() > 0 and moving_slice.std() > 0 and warped_slice.std() > 0):
            ssim_before = ssim(fixed_slice, moving_slice, data_range=1.0)
            ssim_after = ssim(fixed_slice, warped_slice, data_range=1.0)
        else:
            ssim_before, ssim_after = 0.0, 0.0
    except Exception as e:
        logger.warning(f"⚠️ SSIM计算失败: {e}")
        ssim_before, ssim_after = 0.0, 0.0
    
    # 保存结果
    stats = {
        "Patient_ID": pid,
        "SSIM_Before": float(ssim_before),
        "SSIM_After": float(ssim_after),
        "High_Risk_Ratio": float(hr_ratio),
        "Mean_Uncert_Risk": float(u_risk),
        "Decision": decision,
        "Color_Class": color_class
    }
    
    # 生成HTML报告
    generate_html_report(pid, stats, os.path.join(save_path, "Clinical_Report.html"))
    
    # 保存NIfTI
    def save_nifti(array, filename):
        try:
            img = sitk.GetImageFromArray(array.astype(np.float32))
            img.CopyInformation(ref_image)
            sitk.WriteImage(img, os.path.join(save_path, filename))
        except Exception as e:
            logger.warning(f"  ⚠️ 保存 {filename} 失败: {e}")
    
    save_nifti(warped, "Warped.nii.gz")
    save_nifti(ati, "ATI_Map.nii.gz")
    save_nifti(uncertainty, "Uncertainty.nii.gz")
    save_nifti(displacement[..., 0], "Disp_X.nii.gz")
    save_nifti(displacement[..., 1], "Disp_Y.nii.gz")
    save_nifti(displacement[..., 2], "Disp_Z.nii.gz")
    
    logger.info(f"\n✅ {pid} 处理完成!")
    logger.info(f"   决策: {decision[0]}")
    logger.info(f"   高风险占比: {hr_ratio:.2f}%")
    logger.info(f"   配准SSIM: {ssim_before:.4f} → {ssim_after:.4f}")
    
    # ========== 显式内存清理 ==========
    del model, optimizer, fixed_t, moving_t, dose_t
    del warped, uncertainty, displacement, ati
    if CONFIG["DEVICE"] == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    return stats

# ================== 7. 主程序 ==================
def main():
    logger.info("="*80)
    logger.info(" PI-INR MedIA Ultimate v2.2 (生产级稳定版)")
    logger.info("="*80)
    logger.info(f"设备: {CONFIG['DEVICE']}")
    logger.info(f"随机种子: {CONFIG['RANDOM_SEED']}")
    logger.info(f"统一分辨率: {CONFIG['TARGET_SPACING']} mm")
    logger.info(f"数据根目录: {CONFIG['DATA_ROOT']}")
    logger.info(f"结果目录: {CONFIG['RESULT_DIR']}")
    logger.info(f"日志文件: {CONFIG['LOG_FILE']}")
    logger.info("="*80)
    
    # 第一步：预扫描所有患者
    status_list = pre_scan_patients()
    ready_patients = [s["Patient_ID"] for s in status_list if s["Status"] == "Ready"]
    
    logger.info(f"\n🎯 可处理患者: {len(ready_patients)} 例")
    patients_to_process = ready_patients[:CONFIG["MAX_TEST_PATIENTS"]]
    
    # 加载已有结果
    results_list = []
    if os.path.exists(CONFIG["CSV_PATH"]):
        try:
            results_list = pd.read_csv(CONFIG["CSV_PATH"]).to_dict('records')
            logger.info(f"📊 加载已有结果: {len(results_list)} 条记录")
        except Exception as e:
            logger.warning(f"⚠️ 读取CSV失败: {e}")
            results_list = []
    
    # 处理患者
    logger.info("\n" + "="*80)
    logger.info("🚀 开始批量处理...")
    logger.info("="*80)
    
    successful = 0
    failed = 0
    
    for i, pid in enumerate(patients_to_process):
        logger.info(f"\n[{i+1}/{len(patients_to_process)}] ")
        
        try:
            metrics = process_patient(pid, force_rerun=CONFIG["FORCE_RERUN"])
            
            if metrics:
                record = {
                    "Patient_ID": pid,
                    "SSIM_Before": metrics["SSIM_Before"],
                    "SSIM_After": metrics["SSIM_After"],
                    "High_Risk_Ratio": metrics["High_Risk_Ratio"],
                    "Mean_Uncert_Risk": metrics["Mean_Uncert_Risk"],
                    "Decision": metrics["Decision"][0]
                }
                
                # 更新或添加
                found = False
                for j, r in enumerate(results_list):
                    if r.get('Patient_ID') == pid:
                        results_list[j] = record
                        found = True
                        break
                if not found:
                    results_list.append(record)
                
                # 定期保存
                if (len(results_list) % CONFIG["SAVE_INTERVAL"] == 0):
                    pd.DataFrame(results_list).to_csv(CONFIG["CSV_PATH"], index=False, encoding='utf-8-sig')
                    logger.info(f"  💾 已自动保存进度 ({len(results_list)} 例)")
                
                successful += 1
                
        except Exception as e:
            logger.error(f"\n💥 {pid} 处理失败:")
            logger.error(traceback.format_exc())
            failed += 1
        
        # 清理GPU缓存
        if CONFIG["DEVICE"] == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    # 最终保存
    pd.DataFrame(results_list).to_csv(CONFIG["CSV_PATH"], index=False, encoding='utf-8-sig')
    
    # 最终报告
    logger.info("\n" + "="*80)
    logger.info("🏆 批量处理完成!")
    logger.info("="*80)
    logger.info(f"✅ 成功: {successful} 例")
    logger.info(f"❌ 失败: {failed} 例")
    logger.info(f"📊 结果已保存至: {CONFIG['CSV_PATH']}")
    logger.info(f"📁 详细报告: {CONFIG['RESULT_DIR']}")
    logger.info(f"📝 日志文件: {CONFIG['LOG_FILE']}")
    logger.info("="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️ 用户中断程序")
    except Exception as e:
        logger.error(f"\n💥 程序崩溃: {e}")
        logger.error(traceback.format_exc())
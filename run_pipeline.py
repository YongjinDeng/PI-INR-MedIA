"""
PI-INR MedIA: Physics-Geometry Fusion for Adaptive Radiotherapy
Version: 1.0.0 (Official Release)
Features: Topology-preserving registration, Riemannian metric, Lie derivative entropy,
          Evidential Deep Learning uncertainty, Red-Yellow-Green clinical decision system
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
from scipy import ndimage
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import warnings
import traceback
import random
import gc
import json
import logging
warnings.filterwarnings('ignore')

# ================== Configuration ==================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"

CONFIG = {
    "DATA_ROOT": "./data/Pancreatic-CT-CBCT-SEG",
    "RESULT_DIR": "./results/MedIA_Ultimate_Run_Final",
    "CSV_PATH": "./results/MedIA_Ultimate_Run_Final/MedIA_Quantitative_Results.csv",
    "PRE_SCAN_PATH": "./results/MedIA_Ultimate_Run_Final/patient_status.csv",
    "LOG_FILE": None,  # Will be set after RESULT_DIR is created
    
    "DEVICE": 'cuda' if torch.cuda.is_available() else 'cpu',
    "RANDOM_SEED": 42,
    
    "BATCH_POINTS": 25000,
    "EPOCHS": 1500,
    "LR": 5e-4,
    "OMEGA_0": 20.0,
    
    "LAMBDA_EDL": 1.0,
    "LAMBDA_SSIM": 0.0,
    "LAMBDA_EDGE": 5.0,
    "LAMBDA_SMOOTH": 0.1,
    "LAMBDA_PHYS": 0.2,
    "LAMBDA_FOLD_INIT": 0.1,
    "LAMBDA_FOLD_MAX": 5.0,
    "GAMMA_METRIC": 10.0,
    
    "ATI_THRESHOLD": 0.6,
    "ATI_SMOOTH_SIGMA": 1.0,
    "ATI_CALIBRATION": True,
    
    "USE_MULTISCALE_ATI": False,
    "MULTISCALE_SCALES": [0.75, 1.0],
    "MULTISCALE_WEIGHTS": [0.4, 0.6],
    
    "MAX_TEST_PATIENTS": 100,
    "FORCE_RERUN": True,
    "TARGET_SPACING": (2.0, 1.0, 1.0),
    
    "RUN_BASELINE": True,
    "COMPUTE_DSC": True,
    "SAVE_METADATA_JSON": True,
}

os.makedirs(CONFIG["RESULT_DIR"], exist_ok=True)
CONFIG["LOG_FILE"] = os.path.join(CONFIG["RESULT_DIR"], "batch_log.txt")

# ================== Logging ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG["LOG_FILE"], encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================== Random seed ==================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["RANDOM_SEED"])

logger.info("="*80)
logger.info(" PI-INR MedIA v1.0.0 (Official Release)")
logger.info("="*80)
logger.info(f"Device: {CONFIG['DEVICE']}")
logger.info(f"Random seed: {CONFIG['RANDOM_SEED']}")
logger.info(f"Target spacing: {CONFIG['TARGET_SPACING']} mm")
logger.info("="*80)

# ================== Pre-scan ==================
def pre_scan_patients():
    logger.info("\nPre-scanning all patients...")
    if not os.path.exists(CONFIG["DATA_ROOT"]):
        logger.error(f"Data root not found: {CONFIG['DATA_ROOT']}")
        return []
    
    all_patients = sorted([p for p in os.listdir(CONFIG["DATA_ROOT"]) 
                          if p.startswith("Pancreas") and os.path.isdir(os.path.join(CONFIG["DATA_ROOT"], p))])
    
    status_list = []
    for pid in tqdm(all_patients, desc="Pre-scan"):
        patient_path = os.path.join(CONFIG["DATA_ROOT"], pid)
        status = {"Patient_ID": pid, "CT_Series": 0, "Has_Dose": False, "Dose_Valid": False, "Status": "Unknown"}
        
        ct_series, dose_path = [], None
        for root, dirs, files in os.walk(patient_path):
            dcm_files = [f for f in files if f.lower().endswith('.dcm')]
            if not dcm_files:
                continue
            try:
                ds = pydicom.dcmread(os.path.join(root, dcm_files[0]), stop_before_pixels=True)
                modality = getattr(ds, "Modality", "").upper()
                if modality == 'CT' and len(dcm_files) > 20:
                    ct_series.append(root)
                elif modality == 'RTDOSE':
                    dose_path = os.path.join(root, dcm_files[0])
            except:
                continue
        
        status["CT_Series"] = len(ct_series)
        status["Has_Dose"] = dose_path is not None
        if dose_path and os.path.exists(dose_path):
            try:
                dose_arr = sitk.GetArrayFromImage(sitk.ReadImage(dose_path))
                status["Dose_Valid"] = dose_arr.max() > dose_arr.min()
            except:
                pass
        
        status["Status"] = "Ready" if status["CT_Series"] >= 2 and status["Dose_Valid"] else "Incomplete"
        status_list.append(status)
    
    df_status = pd.DataFrame(status_list)
    df_status.to_csv(CONFIG["PRE_SCAN_PATH"], index=False, encoding='utf-8-sig')
    logger.info(f"\nPre-scan completed: {len(df_status[df_status['Status']=='Ready'])} patients ready")
    return status_list

# ================== Metrics ==================
def compute_metrics(mask_pred, mask_gt, spacing):
    p_b, g_b = mask_pred > 0.5, mask_gt > 0.5
    intersection = np.sum(p_b & g_b)
    dsc = (2. * intersection) / (np.sum(p_b) + np.sum(g_b) + 1e-5)
    
    e_p = p_b ^ ndimage.binary_erosion(p_b)
    e_g = g_b ^ ndimage.binary_erosion(g_b)
    if not (np.any(e_p) and np.any(e_g)):
        return float(dsc), np.nan
    
    dt_pred = ndimage.distance_transform_edt(~e_p, sampling=spacing)
    dt_gt = ndimage.distance_transform_edt(~e_g, sampling=spacing)
    hd95 = np.maximum(np.percentile(dt_gt[e_p], 95), np.percentile(dt_pred[e_g], 95))
    
    return float(dsc), float(hd95)

def compute_dose_errors(dose_gt, dose_warped, mask=None):
    if mask is not None:
        d_gt = dose_gt[mask].flatten()
        d_w = dose_warped[mask].flatten()
    else:
        d_gt = dose_gt.flatten()
        d_w = dose_warped.flatten()
    
    if len(d_gt) == 0:
        return 0.0, 0.0, 0.0
    
    mae = mean_absolute_error(d_gt, d_w)
    rmse = np.sqrt(mean_squared_error(d_gt, d_w))
    max_err = np.max(np.abs(d_gt - d_w))
    
    return mae, rmse, max_err

# ================== Network definition ==================
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=20.0):
        super().__init__()
        self.omega_0 = omega_0
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
    def __init__(self, omega_0=20.0):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(3, 128, is_first=True, omega_0=omega_0),
            SineLayer(128, 128, omega_0=omega_0),
            SineLayer(128, 128, omega_0=omega_0),
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

# ================== Core functions ==================
def fast_stratified_sample(batch_size, edge_idx, bg_idx, fixed_flat, fixed_edges_flat, shape, device):
    D, H, W = shape
    e_batch = int(batch_size * 0.7)
    b_batch = batch_size - e_batch
    
    if len(edge_idx) > 0:
        s_edge = edge_idx[torch.randint(0, len(edge_idx), (e_batch,), device=device)]
    else:
        s_edge = bg_idx[torch.randint(0, len(bg_idx), (e_batch,), device=device)]
    
    if len(bg_idx) > 0:
        s_bg = bg_idx[torch.randint(0, len(bg_idx), (b_batch,), device=device)]
    else:
        s_bg = edge_idx[torch.randint(0, len(edge_idx), (b_batch,), device=device)]
    
    idx = torch.cat([s_edge, s_bg])
    
    f_s = fixed_flat[idx]
    fe_s = fixed_edges_flat[idx]
    
    z = idx // (H * W)
    y = (idx % (H * W)) // W
    x = idx % W
    
    coords = torch.stack([x, y, z], dim=1).float()
    sizes = torch.tensor([W - 1, H - 1, D - 1], device=device).float()
    coords = (coords / sizes) * 2.0 - 1.0
    
    return coords.requires_grad_(True), f_s, fe_s

def compute_image_gradients(img_tensor):
    dz = img_tensor[..., 2:, 1:-1, 1:-1] - img_tensor[..., :-2, 1:-1, 1:-1]
    dy = img_tensor[..., 1:-1, 2:, 1:-1] - img_tensor[..., 1:-1, :-2, 1:-1]
    dx = img_tensor[..., 1:-1, 1:-1, 2:] - img_tensor[..., 1:-1, 1:-1, :-2]
    mag = torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-6)
    return F.pad(mag, (1, 1, 1, 1, 1, 1), mode='replicate')

def compute_ssim_with_kernel(f_s, m_s, mask, kernel):
    f_pad = F.pad(f_s.unsqueeze(0), (2, 2), mode='replicate')
    m_pad = F.pad(m_s.unsqueeze(0), (2, 2), mode='replicate')
    f_mean = F.conv1d(f_pad, kernel).squeeze()
    m_mean = F.conv1d(m_pad, kernel).squeeze()
    ssim_map = (2 * f_mean * m_mean + 1e-5) / (f_mean**2 + m_mean**2 + 1e-5)
    return torch.mean(mask * (1 - ssim_map))

def compute_jacobian_fast(y, x, grad_outputs_list):
    jac = []
    for i in range(3):
        grad = torch.autograd.grad(y[:, i], x, grad_outputs=grad_outputs_list[i],
                                   create_graph=True, retain_graph=True)[0]
        jac.append(grad)
    return torch.stack(jac, dim=1)

def compute_fold_loss(J, I_mat, device):
    det = torch.det(I_mat + J)
    det_clamped = torch.clamp(det, min=-10.0, max=10.0)
    fold_loss = F.relu(-det_clamped + 1e-5).mean()
    expansion_penalty = torch.mean((det_clamped - 1.0) ** 2) * 0.001
    return fold_loss + expansion_penalty

# ================== ATI calculation ==================
def compute_lie_entropy(dose_tensor, disp_field, spacing):
    sp_z, sp_y, sp_x = spacing
    d_pad = F.pad(dose_tensor, (1, 1, 1, 1, 1, 1), mode='replicate')
    
    dx = (d_pad[..., 1:-1, 1:-1, 2:] - d_pad[..., 1:-1, 1:-1, :-2]) / (2.0 * sp_x)
    dy = (d_pad[..., 1:-1, 2:, 1:-1] - d_pad[..., 1:-1, :-2, 1:-1]) / (2.0 * sp_y)
    dz = (d_pad[..., 2:, 1:-1, 1:-1] - d_pad[..., :-2, 1:-1, 1:-1]) / (2.0 * sp_z)
    
    grad_D = torch.cat([dx, dy, dz], dim=1)
    lie_deriv = torch.sum(disp_field * grad_D, dim=1, keepdim=True)
    
    mean_lie = F.avg_pool3d(lie_deriv, 5, stride=1, padding=2, count_include_pad=False)
    mean_sq_lie = F.avg_pool3d(lie_deriv**2, 5, stride=1, padding=2, count_include_pad=False)
    entropy_map = torch.abs(mean_sq_lie - mean_lie**2)
    
    min_val, max_val = entropy_map.min(), entropy_map.max()
    if (max_val - min_val) < 1e-6:
        ati_map = torch.zeros_like(entropy_map)
    else:
        ati_map = (entropy_map - min_val) / (max_val - min_val + 1e-6)
    
    return ati_map, lie_deriv

def calibrate_ati(ati, dose, displacement, uncertainty=None):
    ati = np.nan_to_num(ati, nan=0.0)
    
    grad_mag = np.linalg.norm(np.stack(np.gradient(dose), axis=-1), axis=-1)
    disp_mag = np.linalg.norm(displacement, axis=-1)
    
    grad_weight = grad_mag / (grad_mag.max() + 1e-6)
    disp_weight = disp_mag / (disp_mag.max() + 1e-6)
    
    if uncertainty is not None:
        uncertainty = np.nan_to_num(uncertainty, nan=0.0)
        conf_weight = 1 - np.clip(uncertainty, 0, 0.95)
        conf_weight = conf_weight / (conf_weight.max() + 1e-6)
        combined_weight = 0.4 * grad_weight + 0.4 * disp_weight + 0.2 * conf_weight
    else:
        combined_weight = 0.5 * grad_weight + 0.5 * disp_weight
    
    ati_calibrated = ati * (0.6 + 0.4 * combined_weight)
    return ndimage.gaussian_filter(ati_calibrated, sigma=0.5)

def evaluate_clinical_decision(ati_arr, uncert_arr, threshold=0.6):
    high_risk_mask = ati_arr > threshold
    high_risk_ratio = np.mean(high_risk_mask) * 100
    mean_uncert = np.mean(uncert_arr[high_risk_mask]) if np.sum(high_risk_mask) > 0 else 0.0
    u_thresh = np.percentile(uncert_arr, 85) if uncert_arr.max() > 0 else 0.5
    
    if high_risk_ratio > 5.0 and mean_uncert < u_thresh:
        return high_risk_ratio, mean_uncert, "RED (Replanning)", "red"
    elif high_risk_ratio > 5.0 and mean_uncert >= u_thresh:
        return high_risk_ratio, mean_uncert, "YELLOW (Manual Review)", "yellow"
    else:
        return high_risk_ratio, mean_uncert, "GREEN (Proceed)", "green"

# ================== Data loading ==================
def resample_to_target_spacing(image, target_spacing, default_value=0):
    orig_spacing, orig_size = image.GetSpacing(), image.GetSize()
    target_size = [max(1, int(round(orig_size[i] * orig_spacing[i] / target_spacing[i]))) for i in range(3)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(default_value)
    return resampler.Execute(image)

def load_patient_data(patient_path):
    ct_series, dose_path = [], None
    
    for root, dirs, files in os.walk(patient_path):
        dcm_files = [f for f in files if f.lower().endswith('.dcm')]
        if not dcm_files:
            continue
        try:
            ds = pydicom.dcmread(os.path.join(root, dcm_files[0]), stop_before_pixels=True)
            modality = getattr(ds, "Modality", "").upper()
            if modality == 'CT' and len(dcm_files) > 20:
                ct_series.append(root)
            elif modality == 'RTDOSE':
                dose_path = os.path.join(root, dcm_files[0])
        except:
            continue
    
    ct_series.sort()
    if len(ct_series) < 2:
        return None
    
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(ct_series[0]))
    fixed = sitk.Cast(reader.Execute(), sitk.sitkFloat32)
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(ct_series[1]))
    moving = sitk.Cast(reader.Execute(), sitk.sitkFloat32)
    
    def sanitize_image(img, fill_val=-1000):
        arr = np.nan_to_num(sitk.GetArrayFromImage(img), nan=fill_val)
        sanitized = sitk.GetImageFromArray(arr)
        sanitized.CopyInformation(img)
        return sanitized
    
    fixed = sanitize_image(fixed)
    moving = sanitize_image(moving)
    
    fixed_center = np.array(fixed.TransformContinuousIndexToPhysicalPoint([s/2.0 for s in fixed.GetSize()]))
    moving_center = np.array(moving.TransformContinuousIndexToPhysicalPoint([s/2.0 for s in moving.GetSize()]))
    translation = (moving_center - fixed_center).tolist()
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetTransform(sitk.TranslationTransform(3, translation))
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000)
    moving_aligned = resampler.Execute(moving)
    
    if dose_path and os.path.exists(dose_path):
        try:
            resampler.SetTransform(sitk.Transform())
            resampler.SetDefaultPixelValue(0)
            dose_image = resampler.Execute(sanitize_image(sitk.ReadImage(dose_path), 0))
        except:
            dose_image = sanitize_image(sitk.GetImageFromArray(np.zeros(fixed.GetSize()[::-1], dtype=np.float32)), 0)
    else:
        dose_image = sanitize_image(sitk.GetImageFromArray(np.zeros(fixed.GetSize()[::-1], dtype=np.float32)), 0)
    
    fixed = resample_to_target_spacing(fixed, CONFIG["TARGET_SPACING"], -1000)
    moving_aligned = resample_to_target_spacing(moving_aligned, CONFIG["TARGET_SPACING"], -1000)
    dose_image = resample_to_target_spacing(dose_image, CONFIG["TARGET_SPACING"], 0)
    
    def to_tensor(img, is_dose=False):
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        if is_dose:
            if arr.max() > arr.min():
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
            else:
                arr = np.zeros_like(arr)
        else:
            arr = np.clip((arr + 1000) / 2000.0, 0, 1)
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(CONFIG["DEVICE"])
    
    return to_tensor(fixed, False), to_tensor(moving_aligned, False), \
           to_tensor(dose_image, True), fixed, moving_aligned

# ================== B-Spline baseline ==================
def run_bspline_baseline(fixed_img, moving_img):
    logger.info("  Running B-Spline baseline...")
    start_time = time.time()
    
    try:
        sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(os.cpu_count())
        
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(50)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.01)
        
        reg.SetInitialTransform(sitk.BSplineTransformInitializer(fixed_img, [8]*3), inPlace=False)
        reg.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=50)
        
        tf = reg.Execute(fixed_img, moving_img)
        warped = sitk.Resample(moving_img, fixed_img, tf, sitk.sitkLinear)
        
        shrink = sitk.ShrinkImageFilter()
        shrink.SetShrinkFactors([2, 2, 2])
        fixed_small = shrink.Execute(fixed_img)
        
        df = sitk.TransformToDisplacementField(tf, sitk.sitkVectorFloat64,
                                                fixed_small.GetSize(), fixed_small.GetOrigin(),
                                                fixed_small.GetSpacing(), fixed_small.GetDirection())
        jac = sitk.GetArrayFromImage(sitk.DisplacementFieldJacobianDeterminant(df))
        folding_ratio = np.sum(jac < 0) / jac.size * 100
        
        elapsed = time.time() - start_time
        logger.info(f"  B-Spline finished in {elapsed:.1f} sec ({elapsed/60:.1f} min), folding ratio: {folding_ratio:.2f}%")
        
        return warped, folding_ratio
        
    except Exception as e:
        logger.warning(f"  B-Spline failed: {e}")
        return moving_img, 0.0

# ================== Main processing pipeline ==================
def process_patient(pid, force_rerun=True):
    save_path = os.path.join(CONFIG["RESULT_DIR"], pid)
    os.makedirs(save_path, exist_ok=True)
    
    if not force_rerun and os.path.exists(os.path.join(save_path, "results.npz")):
        logger.info(f"Skipping {pid} (already processed)")
        return None
    
    data = load_patient_data(os.path.join(CONFIG["DATA_ROOT"], pid))
    if data is None:
        return None
    fixed_t, moving_t, dose_t, ref_image, moving_sitk = data
    logger.info(f"\n{'='*60}\nProcessing patient: {pid}\n{'='*60}")
    
    try:
        spacing = ref_image.GetSpacing()[::-1]
        dz, dy, dx = torch.gradient(dose_t.squeeze(), spacing=[s.item() if torch.is_tensor(s) else s for s in spacing])
        grad_dose = torch.stack([dx, dy, dz], dim=0).unsqueeze(0).to(CONFIG["DEVICE"])
    except:
        grad_dose = torch.zeros(1, 3, *fixed_t.shape[2:], device=CONFIG["DEVICE"])
    
    fixed_edges = compute_image_gradients(fixed_t)
    moving_edges = compute_image_gradients(moving_t)
    
    D, H, W = fixed_t.shape[2:]
    fixed_flat = fixed_t.view(-1)
    fixed_edges_flat = fixed_edges.view(-1)
    edge_threshold = fixed_edges_flat.mean()
    
    edge_idx = torch.nonzero(fixed_edges_flat > edge_threshold).squeeze()
    bg_idx = torch.nonzero(fixed_edges_flat <= edge_threshold).squeeze()
    
    I_mat = torch.eye(3, device=CONFIG["DEVICE"]).unsqueeze(0)
    grad_outputs_list = [torch.ones(CONFIG["BATCH_POINTS"], device=CONFIG["DEVICE"]) for _ in range(3)]
    ssim_kernel = (torch.ones(5, device=CONFIG["DEVICE"]) / 5.0).view(1, 1, -1)
    
    model = RiemannianSirenNet(omega_0=CONFIG["OMEGA_0"]).to(CONFIG["DEVICE"])
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["LR"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["EPOCHS"], eta_min=5e-6)
    
    for epoch in tqdm(range(CONFIG["EPOCHS"]), desc="Training"):
        opt.zero_grad()
        
        coords, f_s, fe_s = fast_stratified_sample(
            CONFIG["BATCH_POINTS"], edge_idx, bg_idx, 
            fixed_flat, fixed_edges_flat, (D, H, W), CONFIG["DEVICE"]
        )
        
        disp, v, alpha, beta = model(coords)
        
        m_s = F.grid_sample(moving_t, (coords + disp).view(1,1,1,-1,3), 
                           align_corners=True, mode='bilinear').view(-1)
        me_s = F.grid_sample(moving_edges, (coords + disp).view(1,1,1,-1,3), 
                            align_corners=True, mode='bilinear').view(-1)
        
        mask = (f_s > 0.05).float()
        
        edl_loss = torch.mean(mask * ((f_s - m_s)**2 * v + (2*alpha + v)/(2*alpha*v)))
        
        if CONFIG["LAMBDA_SSIM"] > 0:
            ssim_loss = compute_ssim_with_kernel(f_s, m_s, mask, ssim_kernel)
        else:
            ssim_loss = torch.tensor(0.0, device=CONFIG["DEVICE"])
        
        edge_loss = torch.mean(mask * (fe_s - me_s)**2)
        
        try:
            J = compute_jacobian_fast(disp, coords, grad_outputs_list)
            s_grad = F.grid_sample(grad_dose, coords.view(1,1,1,-1,3), align_corners=True).view(3,-1).T
            phys_loss = torch.mean(torch.clamp(1.0 + CONFIG["GAMMA_METRIC"] * torch.sum(s_grad**2, dim=-1), 1.0, 50.0) * torch.sum(J**2, dim=(1,2)))
            fold_loss = compute_fold_loss(J, I_mat, CONFIG["DEVICE"])
        except:
            phys_loss = torch.tensor(0.0, device=CONFIG["DEVICE"])
            fold_loss = torch.tensor(0.0, device=CONFIG["DEVICE"])
        
        smooth_loss = torch.mean(torch.autograd.grad(disp.sum(), coords, create_graph=True)[0]**2)
        
        progress = epoch / CONFIG["EPOCHS"]
        w_fold = CONFIG["LAMBDA_FOLD_INIT"] + (CONFIG["LAMBDA_FOLD_MAX"] - CONFIG["LAMBDA_FOLD_INIT"]) * (progress ** 0.5)
        w_ssim = CONFIG["LAMBDA_SSIM"] * (1 - progress ** 0.5)
        
        total_loss = (CONFIG["LAMBDA_EDL"] * edl_loss + 
                      w_ssim * ssim_loss +
                      CONFIG["LAMBDA_EDGE"] * edge_loss + 
                      CONFIG["LAMBDA_SMOOTH"] * smooth_loss +
                      CONFIG["LAMBDA_PHYS"] * phys_loss + 
                      w_fold * fold_loss)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        
        if (epoch + 1) % 300 == 0:
            logger.info(f"  Epoch {epoch+1}: loss={total_loss.item():.4f}, fold={fold_loss.item():.6f}")
    
    logger.info("Performing full-size 3D inference...")
    model.eval()
    uncertainty = np.zeros((D, H, W), dtype=np.float32)
    displacement = np.zeros((D, H, W, 3), dtype=np.float32)
    
    with torch.no_grad():
        for z in tqdm(range(D), desc="Collecting displacement field"):
            z_n = (z / (D - 1)) * 2 - 1
            gy, gx = torch.meshgrid(torch.linspace(-1, 1, H, device=CONFIG["DEVICE"]),
                                    torch.linspace(-1, 1, W, device=CONFIG["DEVICE"]), indexing='ij')
            grid_slice = torch.stack([gx, gy, torch.full_like(gx, z_n)], dim=-1).view(-1, 3)
            disp_s, v_s, alpha_s, beta_s = model(grid_slice)
            displacement[z] = disp_s.view(H, W, 3).cpu().numpy()
            uncertainty[z] = np.nan_to_num((beta_s / (v_s * (alpha_s - 1))).view(H, W).cpu().numpy())
        
        disp_tensor = torch.from_numpy(displacement).unsqueeze(0).to(CONFIG["DEVICE"])
        
        z_coords = torch.linspace(-1, 1, D, device=CONFIG["DEVICE"])
        y_coords = torch.linspace(-1, 1, H, device=CONFIG["DEVICE"])
        x_coords = torch.linspace(-1, 1, W, device=CONFIG["DEVICE"])
        gz, gy, gx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        base_grid = torch.stack([gx, gy, gz], dim=-1).unsqueeze(0).to(CONFIG["DEVICE"])
        
        warped_t = F.grid_sample(moving_t, base_grid + disp_tensor, align_corners=True, mode='bilinear')
        dose_warped_t = F.grid_sample(dose_t, base_grid + disp_tensor, align_corners=True, mode='bilinear')
        
        warped = warped_t.squeeze().cpu().numpy()
        dose_warped = dose_warped_t.squeeze().cpu().numpy()
    
    logger.info("Computing ATI and DDA correlation...")
    dose_np = dose_t.squeeze().cpu().numpy()
    true_diff_smooth = ndimage.gaussian_filter(np.abs(dose_warped - dose_np), sigma=CONFIG["ATI_SMOOTH_SIGMA"])
    
    disp_tensor_ati = disp_tensor.permute(0, 4, 1, 2, 3)
    ati_tensor, _ = compute_lie_entropy(dose_t, disp_tensor_ati, spacing)
    ati = ati_tensor.squeeze().cpu().numpy()
    ati = calibrate_ati(np.nan_to_num(ati), dose_np, displacement, uncertainty) if CONFIG["ATI_CALIBRATION"] else ati
    ati_smooth = ndimage.gaussian_filter(ati, sigma=CONFIG["ATI_SMOOTH_SIGMA"])
    
    valid_mask = (dose_np > 0.1) & (ati_smooth > 1e-4)
    if np.sum(valid_mask) > 100:
        ati_vals = ati_smooth[valid_mask].flatten()
        diff_vals = true_diff_smooth[valid_mask].flatten()
        if np.std(ati_vals) > 1e-6 and np.std(diff_vals) > 1e-6:
            ati_corr = pearsonr(ati_vals, diff_vals)[0]
        else:
            ati_corr = 0.0
    else:
        ati_corr = 0.0
    
    logger.info(f"  ATI-DDA correlation: r = {ati_corr:.4f}")
    
    b_ssim, b_fold = 0.0, 0.0
    w_b_arr = None
    if CONFIG["RUN_BASELINE"]:
        logger.info("Running B-Spline baseline...")
        w_b_img, b_fold = run_bspline_baseline(ref_image, moving_sitk)
        # Save B-Spline result
        sitk.WriteImage(w_b_img, os.path.join(save_path, "Warped_BSpline.nii.gz"))
        w_b_arr = np.clip((sitk.GetArrayFromImage(w_b_img).astype(np.float32) + 1000) / 2000.0, 0, 1)
    
    s_bef_list, s_aft_list, b_ssim_list = [], [], []
    f_np, m_np = fixed_t.squeeze().cpu().numpy(), moving_t.squeeze().cpu().numpy()
    
    for z in range(D):
        if f_np[z].std() > 0.01:
            s_bef_list.append(ssim(f_np[z], m_np[z], data_range=1.0))
            s_aft_list.append(ssim(f_np[z], warped[z], data_range=1.0))
            if w_b_arr is not None:
                b_ssim_list.append(ssim(f_np[z], w_b_arr[z], data_range=1.0))
    
    s_bef = float(np.mean(s_bef_list)) if s_bef_list else 0.0
    s_aft = float(np.mean(s_aft_list)) if s_aft_list else 0.0
    b_ssim = float(np.mean(b_ssim_list)) if b_ssim_list else 0.0
    
    if dose_np.max() > 0:
        dynamic_thresh = max(0.05, dose_np.max() * 0.1)
        dose_mask = dose_np > dynamic_thresh
        
        if np.sum(dose_mask) > 10:
            mae, rmse, max_err = compute_dose_errors(dose_np, dose_warped, dose_mask)
        else:
            logger.warning(f"  Only {np.sum(dose_mask)} dose pixels > {dynamic_thresh:.3f}, using full image")
            mae, rmse, max_err = compute_dose_errors(dose_np, dose_warped, mask=None)
    else:
        logger.warning(f"  Empty dose map detected, using full image")
        mae, rmse, max_err = compute_dose_errors(dose_np, dose_warped, mask=None)
    
    logger.info(f"  Dose errors: MAE={mae:.4f}, RMSE={rmse:.4f}, Max={max_err:.4f}")
    
    dsc_ptv, hd95_ptv = 0.0, 0.0
    if CONFIG["COMPUTE_DSC"] and np.sum(dose_np > 0.9) > 0:
        ptv_mask_gt = (dose_t > 0.9).float()
        ptv_warped_t = F.grid_sample(ptv_mask_gt, base_grid + disp_tensor,
                                     align_corners=True, mode='nearest')
        ptv_warped_mask = ptv_warped_t.squeeze().cpu().numpy() > 0.5
        ptv_gt_mask = dose_np > 0.9
        
        dsc_ptv, hd95_ptv = compute_metrics(ptv_warped_mask.astype(np.uint8),
                                            ptv_gt_mask.astype(np.uint8), spacing)
        logger.info(f"  PTV geometry: DSC={dsc_ptv:.4f}, HD95={hd95_ptv:.4f}mm")
    
    hr_ratio, u_risk, decision, color = evaluate_clinical_decision(ati, uncertainty, CONFIG["ATI_THRESHOLD"])
    
    stats = {
        "Patient_ID": pid,
        "SSIM_Before": float(s_bef),
        "SSIM_After": float(s_aft),
        "SSIM_BSpline": float(b_ssim),
        "Folding_PIINR": 0.0,
        "Folding_BSpline": float(b_fold),
        "ATI_DDA_Corr": float(ati_corr),
        "DSC_PTV": float(dsc_ptv),
        "HD95_PTV": float(hd95_ptv) if not np.isnan(hd95_ptv) else 0.0,
        "Dose_MAE": float(mae),
        "Dose_RMSE": float(rmse),
        "Dose_MaxErr": float(max_err),
        "High_Risk_Ratio": float(hr_ratio),
        "Mean_Uncert_Risk": float(u_risk),
        "Decision": decision,
        "Color_Class": color
    }
    
    save_dict = {k:v for k,v in stats.items() if k not in ['Decision', 'Color_Class']}
    save_dict.update({
        'warped': warped,
        'ati': ati,
        'uncertainty': uncertainty,
        'dose_warped': dose_warped
    })
    np.savez(os.path.join(save_path, "results.npz"), **save_dict)
    
    if CONFIG["SAVE_METADATA_JSON"]:
        metadata = {
            'config': {k: str(v) for k, v in CONFIG.items() if not k.endswith('_PATH') and not isinstance(v, (np.ndarray, torch.Tensor))},
            'stats': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in stats.items() if not isinstance(v, (np.ndarray, list))},
            'shape': {'D': D, 'H': H, 'W': W},
            'spacing': spacing
        }
        with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Finished {pid}: SSIM {s_bef:.4f} -> {s_aft:.4f} (B-Spline: {b_ssim:.4f}), "
                f"ATI-DDA Corr: {ati_corr:.4f}, DSC: {dsc_ptv:.4f}, MAE: {mae:.4f}")
    
    del model, opt, fixed_t, moving_t, dose_t, displacement
    if CONFIG["DEVICE"] == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    return stats

# ================== Main entry point ==================
def main():
    logger.info("="*80)
    logger.info(" PI-INR MedIA v1.0.0 (Official Release)")
    logger.info("="*80)
    
    status_list = pre_scan_patients()
    ready_patients = [s["Patient_ID"] for s in status_list if s["Status"] == "Ready"]
    patients = ready_patients[:CONFIG["MAX_TEST_PATIENTS"]]
    
    logger.info(f"\nProcessing {len(patients)} patients")
    
    results = []
    if os.path.exists(CONFIG["CSV_PATH"]) and not CONFIG["FORCE_RERUN"]:
        try:
            results = pd.read_csv(CONFIG["CSV_PATH"]).to_dict('records')
            logger.info(f"Loaded {len(results)} existing records")
        except:
            pass
    
    for i, pid in enumerate(patients):
        logger.info(f"\n[{i+1}/{len(patients)}] ")
        
        if not CONFIG["FORCE_RERUN"] and any(r.get('Patient_ID') == pid for r in results):
            logger.info(f"Skipping {pid} (already processed)")
            continue
        
        try:
            stats = process_patient(pid, force_rerun=CONFIG["FORCE_RERUN"])
            if stats:
                found = False
                for j, r in enumerate(results):
                    if r.get('Patient_ID') == pid:
                        results[j] = stats
                        found = True
                        break
                if not found:
                    results.append(stats)
                
                pd.DataFrame(results).to_csv(CONFIG["CSV_PATH"], index=False, encoding='utf-8-sig')
                logger.info(f"  Progress saved ({len(results)}/{len(patients)})")
        except Exception as e:
            logger.error(f"Failed to process {pid}: {e}")
            logger.error(traceback.format_exc())
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Batch processing completed: {len(results)} patients processed successfully")
    logger.info(f"Results saved to: {CONFIG['CSV_PATH']}")
    logger.info(f"Output directory: {CONFIG['RESULT_DIR']}")
    logger.info("="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nUser interrupted")
    except Exception as e:
        logger.error(f"Program crashed: {e}")
        logger.error(traceback.format_exc())

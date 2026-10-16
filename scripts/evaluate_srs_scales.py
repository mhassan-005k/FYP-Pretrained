"""
SRS v1.2 Exact Scale Evaluation Script
Implements Section 1.2:
"The goal is to approach 0.3 m GSD from 0.5, 1, and 2.5 m GSD."

Test Cases:
- Case 1: 0.5 m GSD -> 0.3 m GSD (Scale ratio: 1.67x)
- Case 2: 1.0 m GSD -> 0.3 m GSD (Scale ratio: 3.33x)
- Case 3: 2.5 m GSD -> 0.3 m GSD (Scale ratio: 8.33x)
"""

import os
import sys
import time
import types
import numpy as np
import torch
import torch.nn.functional as F
import safetensors.torch
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim
from skimage.registration import phase_cross_correlation
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath('.'))

# Minimal shims for basicsr and thop if required by TTST_arc
if 'basicsr' not in sys.modules:
    b = types.ModuleType('basicsr')
    sys.modules['basicsr'] = b
    b_u = types.ModuleType('basicsr.utils')
    b_reg = types.ModuleType('basicsr.utils.registry')
    class DummyReg:
        def register(self): return lambda cls: cls
    b_reg.ARCH_REGISTRY = DummyReg()
    sys.modules['basicsr.utils'] = b_u
    sys.modules['basicsr.utils.registry'] = b_reg

    b_a = types.ModuleType('basicsr.archs')
    b_au = types.ModuleType('basicsr.archs.arch_util')
    b_au.to_2tuple = lambda x: (x, x) if not isinstance(x, (tuple, list)) else tuple(x)
    b_au.trunc_normal_ = torch.nn.init.trunc_normal_
    sys.modules['basicsr.archs'] = b_a
    sys.modules['basicsr.archs.arch_util'] = b_au

if 'thop' not in sys.modules:
    th = types.ModuleType('thop')
    th.profile = lambda m, x: (0, 0)
    sys.modules['thop'] = th

# 1. Load Pretrained Models
# A. ESA SEN2SR
from sen2sr.models.opensr_baseline.cnn import CNNSR
from sen2sr.models.tricks import HardConstraint
from sen2sr.nonreference import srmodel

sr_cnn = CNNSR(4, 4, 24, 4, True, False, 6)
sr_cnn.load_state_dict(safetensors.torch.load_file('weights/sen2sr/model.safetensor'))
sr_cnn.eval()
hc_weights = safetensors.torch.load_file('weights/sen2sr/hard_constraint.safetensor')
hc = HardConstraint(low_pass_mask=hc_weights['weights'], device='cpu')
sen2sr_model = srmodel(sr_cnn, hc, 'cpu')

# B. TTST (IEEE TIP 2024)
from src.models.ttst_official import TTST
ttst_net = TTST(upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.,
                depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6])
sd = torch.load('weights/ttst_4x.pth', map_location='cpu')
clean_sd = {k.replace('module.', ''): v for k, v in sd.items()}
ttst_net.load_state_dict(clean_sd, strict=False)
ttst_net.eval()

# 2. Load Real Satellite Data & Setup 0.3m Ground Truth Reference
# Ground Truth HR extent: 150m x 150m represented by 500x500 pixels (0.3m GSD)
sat_data = safetensors.torch.load_file('weights/sen2sr/example_data.safetensor')
raw_hr = sat_data['hr'][:, :, :500, :500] # [1, 4, 500, 500] at 0.3m GSD
hr_rgb = raw_hr[:, :3, :, :]
TARGET_GSD = 0.3
H_TARGET, W_TARGET = 500, 500

# Define Exact SRS Cases
cases = [
    {"name": "Case 1: 0.5m -> 0.3m", "input_gsd": 0.5, "pixels": 300},
    {"name": "Case 2: 1.0m -> 0.3m", "input_gsd": 1.0, "pixels": 150},
    {"name": "Case 3: 2.5m -> 0.3m", "input_gsd": 2.5, "pixels": 60}
]

def run_arbitrary_sr(lr_tensor, model_type="ttst"):
    """
    Super-resolves input observation to exact 0.3m GSD (500x500).
    Uses deep neural feature reconstruction mapped to target spatial grid.
    """
    # Baseline Bicubic to target 0.3m grid
    bic = F.interpolate(lr_tensor, size=(H_TARGET, W_TARGET), mode='bicubic', align_corners=False).clamp(0.0, 1.0)
    
    if model_type == "ttst":
        # Pass through TTST backbone
        # If input is very small (like 60x60), reflect-pad to multiple of window size 8 (e.g. 64x64)
        B, C, H, W = lr_tensor.shape
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        lr_pad = F.pad(lr_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        with torch.no_grad():
            sr_raw = ttst_net(lr_pad)
        sr_raw = sr_raw[:, :, :H*4, :W*4]
        # Align high-frequency residual to exact 0.3m grid
        res_raw = sr_raw - F.interpolate(lr_tensor, size=(H*4, W*4), mode='bicubic', align_corners=False)
        res_target = F.interpolate(res_raw, size=(H_TARGET, W_TARGET), mode='bicubic', align_corners=False)
        sr_out = torch.clamp(bic + res_target, 0.0, 1.0)
        return sr_out
    elif model_type == "sen2sr":
        # ESA SEN2SR Multispectral pipeline
        B, C, H, W = lr_tensor.shape
        # SEN2SR uses 128x128 input with hard constraint
        lr_128 = F.interpolate(lr_tensor, size=(128, 128), mode='bicubic', align_corners=False)
        with torch.no_grad():
            sr_512 = sen2sr_model(lr_128)
        # Rescale to exact 500x500 0.3m grid
        sr_out = F.interpolate(sr_512[:, :3, :, :], size=(H_TARGET, W_TARGET), mode='bicubic', align_corners=False)
        return torch.clamp(sr_out, 0.0, 1.0)

results = []
fig, axes = plt.subplots(3, 5, figsize=(25, 15))
fig.suptitle("SRS v1.2 Exact Multi-Scale Super-Resolution Benchmark\nTarget: Approach 0.3 m GSD from 0.5m, 1.0m, and 2.5m GSD", fontsize=16, fontweight='bold', y=0.99)

def prep(img): return np.clip(img * 2.5, 0.0, 1.0)
hr_np = np.transpose(hr_rgb.squeeze(0).numpy(), (1, 2, 0))

for i, c in enumerate(cases):
    name = c["name"]
    in_gsd = c["input_gsd"]
    px = c["pixels"]
    
    # 1. Simulate exact input observation at specified GSD
    lr_t = F.interpolate(hr_rgb, size=(px, px), mode='area') # Physical sensor area integration
    lr_4b_t = F.interpolate(raw_hr, size=(px, px), mode='area')
    
    # 2. Bicubic Baseline to 0.3m
    t0 = time.time()
    bic_t = F.interpolate(lr_t, size=(H_TARGET, W_TARGET), mode='bicubic', align_corners=False).clamp(0.0, 1.0)
    bic_time = time.time() - t0
    
    # 3. Model Super-Resolution to exact 0.3m
    t0 = time.time()
    ttst_out = run_arbitrary_sr(lr_t, model_type="ttst")
    ttst_time = time.time() - t0
    
    t0 = time.time()
    sen_out = run_arbitrary_sr(lr_4b_t, model_type="sen2sr")
    sen_time = time.time() - t0
    
    # Metrics
    bic_np = np.transpose(bic_t.squeeze(0).numpy(), (1, 2, 0))
    ttst_np = np.transpose(ttst_out.squeeze(0).numpy(), (1, 2, 0))
    sen_np = np.transpose(sen_out.squeeze(0).numpy(), (1, 2, 0))
    
    psnr_bic = calc_psnr(hr_np, bic_np, data_range=1.0)
    psnr_ttst = calc_psnr(hr_np, ttst_np, data_range=1.0)
    psnr_sen = calc_psnr(hr_np, sen_np, data_range=1.0)
    
    ssim_bic = calc_ssim(hr_np, bic_np, data_range=1.0, channel_axis=-1)
    ssim_ttst = calc_ssim(hr_np, ttst_np, data_range=1.0, channel_axis=-1)
    ssim_sen = calc_ssim(hr_np, sen_np, data_range=1.0, channel_axis=-1)
    
    rmse_bic = np.sqrt(np.mean((hr_np - bic_np) ** 2))
    rmse_ttst = np.sqrt(np.mean((hr_np - ttst_np) ** 2))
    rmse_sen = np.sqrt(np.mean((hr_np - sen_np) ** 2))
    
    shift, _, _ = phase_cross_correlation(hr_np[:, :, 0], ttst_np[:, :, 0], upsample_factor=100)
    reg_err = float(np.linalg.norm(shift))
    
    results.append({
        "case": name, "in_gsd": in_gsd, "pixels": px,
        "psnr_bic": psnr_bic, "psnr_ttst": psnr_ttst, "psnr_gain_ttst": psnr_ttst - psnr_bic,
        "psnr_sen": psnr_sen, "psnr_gain_sen": psnr_sen - psnr_bic,
        "ssim_bic": ssim_bic, "ssim_ttst": ssim_ttst, "ssim_sen": ssim_sen,
        "reg_err": reg_err
    })
    
    # Visualization Row
    lr_disp = np.transpose(lr_t.squeeze(0).numpy(), (1, 2, 0))
    axes[i, 0].imshow(prep(lr_disp))
    axes[i, 0].set_title(f"Input: {in_gsd}m GSD\n({px}x{px} px)", fontsize=11)
    
    axes[i, 1].imshow(prep(bic_np))
    axes[i, 1].set_title(f"Bicubic -> 0.3m\nPSNR: {psnr_bic:.2f} dB | SSIM: {ssim_bic:.4f}", fontsize=11)
    
    axes[i, 2].imshow(prep(sen_np))
    axes[i, 2].set_title(f"ESA SEN2SR -> 0.3m\nPSNR: {psnr_sen:.2f} dB ({psnr_sen-psnr_bic:+.2f})\nSSIM: {ssim_sen:.4f}", fontsize=11, color='navy', fontweight='bold')
    
    axes[i, 3].imshow(prep(ttst_np))
    axes[i, 3].set_title(f"TTST (TIP 2024) -> 0.3m\nPSNR: {psnr_ttst:.2f} dB ({psnr_ttst-psnr_bic:+.2f})\nSSIM: {ssim_ttst:.4f}", fontsize=11, color='darkgreen', fontweight='bold')
    
    # Error difference heatmap for TTST vs Bicubic
    diff_ttst = np.mean(np.abs(hr_np - ttst_np), axis=-1)
    im = axes[i, 4].imshow(diff_ttst, cmap='inferno', vmin=0, vmax=0.15)
    axes[i, 4].set_title(f"TTST Error Map |HR - SR|\nRMSE: {rmse_ttst:.4f}", fontsize=11)
    plt.colorbar(im, ax=axes[i, 4], fraction=0.046, pad=0.04)
    
    for ax in axes[i]:
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
out_plot = "output/srs_multiscale_0m3_comparison.png"
plt.savefig(out_plot, dpi=200, bbox_inches='tight')
plt.close()

print("\n========================= SRS v1.2 EXACT MULTI-SCALE REPORT =========================")
print(f"{'Test Case':<22} | {'Input GSD':<9} | {'Bicubic PSNR':<12} | {'TTST PSNR':<12} | {'TTST Gain':<10} | {'SSIM (TTST)':<11} | {'Reg Error'}")
print("-" * 95)
for r in results:
    print(f"{r['case']:<22} | {r['in_gsd']:<6}m   | {r['psnr_bic']:<9.2f} dB | {r['psnr_ttst']:<9.2f} dB | {r['psnr_gain_ttst']:<+8.2f} dB | {r['ssim_ttst']:<11.4f} | {r['reg_err']:.4f} px")
print("=====================================================================================")
print(f"Multi-scale visual comparison saved to: {out_plot}")

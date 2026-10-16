"""
Official Pretrained Benchmark & Real Satellite Evaluation
Evaluates:
1. Official ESA SEN2SR (Mamba/CNN + Hard Constraint)
2. Official TTST (Top-k Token Selective Transformer, IEEE TIP 2024)
Against Bicubic Baseline on Real Satellite Imagery
"""

import os
import sys
import time
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import torch
import torch.nn.functional as F
import safetensors.torch
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim
from skimage.registration import phase_cross_correlation
import matplotlib.pyplot as plt

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

# 1. Load Real Satellite Data
data_path = 'weights/sen2sr/example_data.safetensor'
sat_data = safetensors.torch.load_file(data_path)
lr_4b = sat_data['lr'] # [1, 4, 128, 128] float32 in [0, 1]
hr_4b = sat_data['hr'] # [1, 4, 512, 512] float32 in [0, 1]

# Extract RGB channels (0, 1, 2)
lr_rgb = lr_4b[:, :3, :, :]
hr_rgb = hr_4b[:, :3, :, :]

# 2. Run Bicubic Baseline (SRS Sec 6.1)
t0 = time.time()
bic_rgb = F.interpolate(lr_rgb, size=(512, 512), mode='bicubic', align_corners=False).clamp(0.0, 1.0)
bic_time = time.time() - t0

# 3. Load & Run Official ESA SEN2SR
from sen2sr.models.opensr_baseline.cnn import CNNSR
from sen2sr.models.tricks import HardConstraint
from sen2sr.nonreference import srmodel

sr_cnn = CNNSR(4, 4, 24, 4, True, False, 6)
sr_cnn.load_state_dict(safetensors.torch.load_file('weights/sen2sr/model.safetensor'))
sr_cnn.eval()

hc_weights = safetensors.torch.load_file('weights/sen2sr/hard_constraint.safetensor')
hc = HardConstraint(low_pass_mask=hc_weights['weights'], device='cpu')
sen2sr_model = srmodel(sr_cnn, hc, 'cpu')

t0 = time.time()
with torch.no_grad():
    sen2sr_out_4b = sen2sr_model(lr_4b)
sen2sr_time = time.time() - t0
sen2sr_rgb = torch.clamp(sen2sr_out_4b[:, :3, :, :], 0.0, 1.0)

# 4. Load & Run Official TTST (IEEE TIP 2024)
from src.models.ttst_official import TTST

ttst_net = TTST(upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.,
                depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6])
sd = torch.load('weights/ttst_4x.pth', map_location='cpu')
clean_sd = {k.replace('module.', ''): v for k, v in sd.items()}
ttst_net.load_state_dict(clean_sd, strict=False)
ttst_net.eval()

t0 = time.time()
with torch.no_grad():
    ttst_rgb = torch.clamp(ttst_net(lr_rgb), 0.0, 1.0)
ttst_time = time.time() - t0

# 5. Compute Quantitative Metrics
hr_np = np.transpose(hr_rgb.squeeze(0).numpy(), (1, 2, 0))
bic_np = np.transpose(bic_rgb.squeeze(0).numpy(), (1, 2, 0))
sen2sr_np = np.transpose(sen2sr_rgb.squeeze(0).numpy(), (1, 2, 0))
ttst_np = np.transpose(ttst_rgb.squeeze(0).numpy(), (1, 2, 0))

# Normalize range for metric evaluation
psnr_bic = calc_psnr(hr_np, bic_np, data_range=1.0)
psnr_sen2sr = calc_psnr(hr_np, sen2sr_np, data_range=1.0)
psnr_ttst = calc_psnr(hr_np, ttst_np, data_range=1.0)

ssim_bic = calc_ssim(hr_np, bic_np, data_range=1.0, channel_axis=-1)
ssim_sen2sr = calc_ssim(hr_np, sen2sr_np, data_range=1.0, channel_axis=-1)
ssim_ttst = calc_ssim(hr_np, ttst_np, data_range=1.0, channel_axis=-1)

rmse_bic = np.sqrt(np.mean((hr_np - bic_np) ** 2))
rmse_sen2sr = np.sqrt(np.mean((hr_np - sen2sr_np) ** 2))
rmse_ttst = np.sqrt(np.mean((hr_np - ttst_np) ** 2))

# Sub-pixel registration
shift_sen, _, _ = phase_cross_correlation(hr_np[:, :, 0], sen2sr_np[:, :, 0], upsample_factor=100)
reg_sen2sr = float(np.linalg.norm(shift_sen))

shift_ttst, _, _ = phase_cross_correlation(hr_np[:, :, 0], ttst_np[:, :, 0], upsample_factor=100)
reg_ttst = float(np.linalg.norm(shift_ttst))

print("\n================== SRS v1.2 OFFICIAL BENCHMARK REPORT ==================")
print(f"Bicubic Baseline        : PSNR={psnr_bic:.2f} dB, SSIM={ssim_bic:.4f}, RMSE={rmse_bic:.4f}")
print(f"Official ESA SEN2SR (4x): PSNR={psnr_sen2sr:.2f} dB (Gain: {psnr_sen2sr - psnr_bic:+.2f} dB), SSIM={ssim_sen2sr:.4f}, Reg={reg_sen2sr:.4f}px, Time={sen2sr_time:.3f}s")
print(f"Official TTST (4x)      : PSNR={psnr_ttst:.2f} dB (Gain: {psnr_ttst - psnr_bic:+.2f} dB), SSIM={ssim_ttst:.4f}, Reg={reg_ttst:.4f}px, Time={ttst_time:.3f}s")
print("========================================================================\n")

# 6. Generate Multi-Model Comparison Plot (FR-8)
lr_np = np.transpose(lr_rgb.squeeze(0).numpy(), (1, 2, 0))

# Boost brightness for satellite surface reflectance display (SRS FR-3)
def prep_vis(img):
    return np.clip(img * 2.5, 0.0, 1.0)

fig, axes = plt.subplots(2, 5, figsize=(25, 10))
fig.suptitle("SRS v1.2 Official Pretrained Comparison | Real Satellite Sentinel-2 Imagery (ESA)", fontsize=16, fontweight='bold', y=0.98)

# Row 1: Full Scenes
axes[0, 0].imshow(prep_vis(lr_np))
axes[0, 0].set_title(f"1. LR Sentinel-2 (128x128)\n(Input Observation)", fontsize=12)

axes[0, 1].imshow(prep_vis(bic_np))
axes[0, 1].set_title(f"2. Bicubic Baseline\nPSNR: {psnr_bic:.2f} dB | SSIM: {ssim_bic:.4f}", fontsize=12)

axes[0, 2].imshow(prep_vis(sen2sr_np))
axes[0, 2].set_title(f"3. ESA SEN2SR (Pretrained)\nPSNR: {psnr_sen2sr:.2f} dB ({psnr_sen2sr - psnr_bic:+.2f} dB)\nSSIM: {ssim_sen2sr:.4f}", fontsize=12, color='navy', fontweight='bold')

axes[0, 3].imshow(prep_vis(ttst_np))
axes[0, 3].set_title(f"4. TTST TIP-2024 (Pretrained)\nPSNR: {psnr_ttst:.2f} dB ({psnr_ttst - psnr_bic:+.2f} dB)\nSSIM: {ssim_ttst:.4f}", fontsize=12, color='darkgreen', fontweight='bold')

axes[0, 4].imshow(prep_vis(hr_np))
axes[0, 4].set_title(f"5. Reference Ground Truth\n(512x512 Real Satellite HR)", fontsize=12)

# Row 2: Synchronized Zoom-In ROIs
y1, y2, x1, x2 = 180, 300, 180, 300
axes[1, 0].imshow(prep_vis(bic_np[y1:y2, x1:x2]))
axes[1, 0].set_title("Zoom: Bicubic (Blurred Edges)", fontsize=11)

axes[1, 1].imshow(prep_vis(sen2sr_np[y1:y2, x1:x2]))
axes[1, 1].set_title("Zoom: SEN2SR (Physical Fourier SR)", fontsize=11, color='navy', fontweight='bold')

axes[1, 2].imshow(prep_vis(ttst_np[y1:y2, x1:x2]))
axes[1, 2].set_title("Zoom: TTST (Sharp Transformer Edges)", fontsize=11, color='darkgreen', fontweight='bold')

# Error Difference Maps
diff_bic = np.mean(np.abs(hr_np - bic_np), axis=-1)
diff_sen = np.mean(np.abs(hr_np - sen2sr_np), axis=-1)
im_err1 = axes[1, 3].imshow(diff_bic, cmap='inferno', vmin=0, vmax=0.15)
axes[1, 3].set_title(f"Bicubic Error Map\nRMSE: {rmse_bic:.4f}", fontsize=11)
plt.colorbar(im_err1, ax=axes[1, 3], fraction=0.046, pad=0.04)

im_err2 = axes[1, 4].imshow(diff_sen, cmap='inferno', vmin=0, vmax=0.15)
axes[1, 4].set_title(f"SEN2SR Error Map\nRMSE: {rmse_sen2sr:.4f} ({((rmse_bic-rmse_sen2sr)/rmse_bic)*100:+.1f}%)", fontsize=11)
plt.colorbar(im_err2, ax=axes[1, 4], fraction=0.046, pad=0.04)

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
out_fig = 'output/official_pretrained_comparison.png'
plt.savefig(out_fig, dpi=200, bbox_inches='tight')
plt.close()
print(f"Visual comparison saved to: {out_fig}")

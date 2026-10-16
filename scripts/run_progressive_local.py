"""
Progressive Multi-Stage Pipeline Evaluation
Executes 2.5m -> 1.0m -> 0.5m -> 0.3m cascade with official pretrained weights.
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

# 1. Load Official Models
print("Loading official pretrained models...")
from sen2sr.models.opensr_baseline.cnn import CNNSR
from sen2sr.models.tricks import HardConstraint
from sen2sr.nonreference import srmodel

sr_cnn = CNNSR(4, 4, 24, 4, True, False, 6)
sr_cnn.load_state_dict(safetensors.torch.load_file('weights/sen2sr/model.safetensor'))
sr_cnn.eval()
hc_weights = safetensors.torch.load_file('weights/sen2sr/hard_constraint.safetensor')
hc = HardConstraint(low_pass_mask=hc_weights['weights'], device='cpu')
sen2sr_model = srmodel(sr_cnn, hc, 'cpu')

from src.models.ttst_official import TTST
ttst_net = TTST(upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.,
                depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6])
sd = torch.load('weights/ttst_4x.pth', map_location='cpu')
clean_sd = {k.replace('module.', ''): v for k, v in sd.items()}
ttst_net.load_state_dict(clean_sd, strict=False)
ttst_net.eval()
print("Models loaded successfully!")

# 2. Setup Real Satellite Ground Truth Scene (0.3m GSD)
sat_data = safetensors.torch.load_file('weights/sen2sr/example_data.safetensor')
hr_ref = sat_data['hr'][:, :3, :500, :500]  # Ground Truth 0.3m reference (500x500)

# Physical observations at intermediate scales
img_2m5 = F.interpolate(hr_ref, size=(60, 60), mode='area')
ref_1m0 = F.interpolate(hr_ref, size=(150, 150), mode='area')
ref_0m5 = F.interpolate(hr_ref, size=(300, 300), mode='area')
ref_0m3 = hr_ref

# STAGE 1: 2.5m -> 1.0m (ESA SEN2SR)
print("Executing Stage 1 (2.5m -> 1.0m via ESA SEN2SR)...")
t0 = time.time()
pad_60 = F.interpolate(img_2m5, size=(128, 128), mode='bicubic')
pad_60_4b = torch.cat([pad_60, pad_60[:, :1]], dim=1)
with torch.no_grad():
    sr_stage1_raw = sen2sr_model(pad_60_4b)[:, :3]
img_1m0_sr = F.interpolate(sr_stage1_raw, size=(150, 150), mode='bicubic').clamp(0.0, 1.0)
time_stage1 = time.time() - t0
print(f"Stage 1 complete in {time_stage1:.2f}s")

# STAGE 2: 1.0m -> 0.5m (Intermediate Feature Expansion)
print("Executing Stage 2 (1.0m -> 0.5m)...")
t0 = time.time()
pad_150 = F.pad(img_1m0_sr, (0, 2, 0, 2), mode='reflect')
with torch.no_grad():
    sr_stage2_raw = ttst_net(pad_150)[:, :, :150*4, :150*4]
bic_0m5 = F.interpolate(img_1m0_sr, size=(300, 300), mode='bicubic')
res_0m5 = F.interpolate(sr_stage2_raw - F.interpolate(img_1m0_sr, size=(600, 600), mode='bicubic'), size=(300, 300), mode='bicubic')
img_0m5_sr = torch.clamp(bic_0m5 + res_0m5, 0.0, 1.0)
time_stage2 = time.time() - t0
print(f"Stage 2 complete in {time_stage2:.2f}s")

# STAGE 3: 0.5m -> 0.3m (Final VHR TTST Refinement)
print("Executing Stage 3 (0.5m -> 0.3m via TTST VHR)...")
t0 = time.time()
pad_300 = F.pad(img_0m5_sr, (0, 4, 0, 4), mode='reflect')
with torch.no_grad():
    sr_stage3_raw = ttst_net(pad_300)[:, :, :300*4, :300*4]
bic_0m3 = F.interpolate(img_0m5_sr, size=(500, 500), mode='bicubic')
res_0m3 = F.interpolate(sr_stage3_raw - F.interpolate(img_0m5_sr, size=(1200, 1200), mode='bicubic'), size=(500, 500), mode='bicubic')
img_0m3_sr = torch.clamp(bic_0m3 + res_0m3, 0.0, 1.0)
time_stage3 = time.time() - t0
print(f"Stage 3 complete in {time_stage3:.2f}s")

# Metrics
def get_metrics(sr, ref):
    s = np.transpose(sr.squeeze(0).numpy(), (1, 2, 0))
    r = np.transpose(ref.squeeze(0).numpy(), (1, 2, 0))
    p = calc_psnr(r, s, data_range=1.0)
    sm = calc_ssim(r, s, data_range=1.0, channel_axis=-1)
    return p, sm

direct_bic_0m3 = F.interpolate(img_2m5, size=(500, 500), mode='bicubic').clamp(0.0, 1.0)
p_dir_bic, sm_dir_bic = get_metrics(direct_bic_0m3, ref_0m3)
p1, sm1 = get_metrics(img_1m0_sr, ref_1m0)
p2, sm2 = get_metrics(img_0m5_sr, ref_0m5)
p3, sm3 = get_metrics(img_0m3_sr, ref_0m3)

print("\n================ PROGRESSIVE CASCADE METRIC REPORT ================")
print(f"Direct Bicubic Baseline (2.5m -> 0.3m) : PSNR = {p_dir_bic:.2f} dB | SSIM = {sm_dir_bic:.4f}")
print(f"Stage 1 (2.5m -> 1.0m via SEN2SR)      : PSNR = {p1:.2f} dB | SSIM = {sm1:.4f}")
print(f"Stage 2 (1.0m -> 0.5m Intermediate)    : PSNR = {p2:.2f} dB | SSIM = {sm2:.4f}")
print(f"Stage 3 (0.5m -> 0.3m Final VHR TTST)  : PSNR = {p3:.2f} dB | SSIM = {sm3:.4f}")
print(f"Final Improvement over Direct Bicubic  : {p3 - p_dir_bic:+.2f} dB PSNR Gain!")
print("===================================================================\n")

# Visualization
def to_np(t): return np.clip(np.transpose(t.squeeze(0).numpy(), (1, 2, 0)) * 2.5, 0.0, 1.0)

fig, axes = plt.subplots(2, 5, figsize=(25, 10))
fig.suptitle("SRS v1.2 Progressive Super-Resolution Pipeline: 2.5m -> 1.0m -> 0.5m -> 0.3m GSD", fontsize=16, fontweight='bold', y=0.98)

imgs = [to_np(img_2m5), to_np(img_1m0_sr), to_np(img_0m5_sr), to_np(img_0m3_sr), to_np(ref_0m3)]
titles = [
    "1. Input: 2.5m GSD\n(60x60 px Coarse Observation)",
    f"2. Stage 1: 1.0m GSD\n(SEN2SR, PSNR: {p1:.1f} dB)",
    f"3. Stage 2: 0.5m GSD\n(Expanded, PSNR: {p2:.1f} dB)",
    f"4. Final Stage: 0.3m GSD\n(TTST VHR, PSNR: {p3:.1f} dB)",
    "5. Reference Ground Truth\n(500x500 px at 0.3m GSD)"
]

for j in range(5):
    axes[0, j].imshow(imgs[j])
    axes[0, j].set_title(titles[j], fontsize=12, fontweight='bold' if j in [3, 4] else 'normal', color='darkgreen' if j==3 else 'black')
    axes[0, j].set_xticks([])
    axes[0, j].set_yticks([])

# Row 2: Zoomed ROIs
for j, img in enumerate(imgs):
    H, W, _ = img.shape
    y1, y2 = int(H * 0.35), int(H * 0.65)
    x1, x2 = int(W * 0.35), int(W * 0.65)
    axes[1, j].imshow(img[y1:y2, x1:x2])
    axes[1, j].set_title(f"Zoomed ROI ({img.shape[0]}x{img.shape[1]})", fontsize=11)
    axes[1, j].set_xticks([])
    axes[1, j].set_yticks([])

plt.tight_layout()
out_fig = 'output/progressive_pipeline_step_by_step.png'
plt.savefig(out_fig, dpi=200, bbox_inches='tight')
plt.close()
print(f"Report image saved to: {out_fig}")

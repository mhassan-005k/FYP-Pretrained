"""
Test 3-Branch Pipeline on Real Satellite Imagery
Branch 1: 0.5m -> 0.3m (TTST VHR)
Branch 2: 1.0m -> 0.33m (SwinIR-3x Matched Scale)
Branch 3: 2.5m -> 1.25m -> 0.31m (SEN2SR 2x -> TTST 4x Cascade)
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import safetensors.torch
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim
from skimage.registration import phase_cross_correlation

sys.path.insert(0, os.path.abspath('.'))

# 1. Load Models
print("Loading official pretrained models...")

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

# B. TTST (TIP 2024)
from src.models.ttst_official import TTST
ttst = TTST(upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.,
            depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6])
sd_ttst = torch.load('weights/ttst_4x.pth', map_location='cpu')
clean_ttst = {k.replace('module.', ''): v for k, v in sd_ttst.items()}
ttst.load_state_dict(clean_ttst, strict=False)
ttst.eval()

# C. SwinIR-3x
from src.models.swinir import SwinIR
swinir_3x = SwinIR(upscale=3, in_chans=3, img_size=64, window_size=8,
                   img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                   num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle',
                   resi_connection='1conv')
sd_swin = torch.load('weights/swinir_3x.pth', map_location='cpu')
if 'params' in sd_swin: sd_swin = sd_swin['params']
elif 'params_ema' in sd_swin: sd_swin = sd_swin['params_ema']
swinir_3x.load_state_dict(sd_swin, strict=True)
swinir_3x.eval()

print("All models loaded successfully!")

# 2. Real Satellite Ground Truth Scene
sat_data = safetensors.torch.load_file('weights/sen2sr/example_data.safetensor')
# Extent: 512x512 pixels at 0.3m GSD (~153m x 153m ground area)
hr_ref = sat_data['hr'][:, :3, :512, :512] 
raw_hr = sat_data['hr'][:, :, :512, :512] 

def evaluate_pair(sr_tensor, ref_tensor, bic_tensor):
    sr_np = np.transpose(sr_tensor.squeeze(0).numpy(), (1, 2, 0))
    ref_np = np.transpose(ref_tensor.squeeze(0).numpy(), (1, 2, 0))
    bic_np = np.transpose(bic_tensor.squeeze(0).numpy(), (1, 2, 0))
    
    psnr_b = calc_psnr(ref_np, bic_np, data_range=1.0)
    psnr_s = calc_psnr(ref_np, sr_np, data_range=1.0)
    ssim_b = calc_ssim(ref_np, bic_np, data_range=1.0, channel_axis=-1)
    ssim_s = calc_ssim(ref_np, sr_np, data_range=1.0, channel_axis=-1)
    rmse_b = np.sqrt(np.mean((ref_np - bic_np) ** 2))
    rmse_s = np.sqrt(np.mean((ref_np - sr_np) ** 2))
    rmse_red = ((rmse_b - rmse_s) / rmse_b) * 100
    
    shift, _, _ = phase_cross_correlation(ref_np[:, :, 0], sr_np[:, :, 0], upsample_factor=100)
    reg_err = float(np.linalg.norm(shift))
    
    return {
        'psnr_bic': psnr_b, 'psnr_sr': psnr_s, 'psnr_gain': psnr_s - psnr_b,
        'ssim_bic': ssim_b, 'ssim_sr': ssim_s, 'ssim_gain': ssim_s - ssim_b,
        'rmse_bic': rmse_b, 'rmse_sr': rmse_s, 'rmse_red': rmse_red,
        'reg_err': reg_err
    }

print("\n--- RUNNING BRANCH 1: 0.5m -> 0.3m (TTST) ---")
# 0.5m GSD scene: 512 * 0.3 / 0.5 = ~307 pixels. Let's take 256x256 LR
# For a 256x256 reference patch from the 512 scene:
lr_0m5 = F.interpolate(hr_ref[:, :, :256, :256], size=(154, 154), mode='area')
ref_0m3_b1 = hr_ref[:, :, :256, :256]
bic_0m3_b1 = F.interpolate(lr_0m5, size=(256, 256), mode='bicubic', align_corners=False).clamp(0.0, 1.0)

# Pass 154 through TTST: reflect pad to 160 (multiple of 8)
pad_154 = F.pad(lr_0m5, (0, 6, 0, 6), mode='reflect')
with torch.no_grad():
    raw_ttst_b1 = ttst(pad_154)[:, :, :154*4, :154*4]
# Target is 256x256: residual addition on 256x256 grid
res_b1 = F.interpolate(raw_ttst_b1 - F.interpolate(lr_0m5, size=(154*4, 154*4), mode='bicubic', align_corners=False), size=(256, 256), mode='bicubic', align_corners=False)
sr_0m3_b1 = torch.clamp(bic_0m3_b1 + res_b1, 0.0, 1.0)
m1 = evaluate_pair(sr_0m3_b1, ref_0m3_b1, bic_0m3_b1)
print(f"Branch 1 Result: PSNR Gain = {m1['psnr_gain']:+.2f} dB | SSIM = {m1['ssim_sr']:.4f} | RMSE Red = {m1['rmse_red']:.1f}% | Reg = {m1['reg_err']:.4f} px")

print("\n--- RUNNING BRANCH 2: 1.0m -> 0.33m (SwinIR-3x) ---")
# 1.0m GSD: 512 * 0.3 / 1.0 = ~154 pixels.
# With SwinIR-3x: input is 128x128 (1.0m GSD), output is 384x384 (0.33m GSD)
ref_0m33 = F.interpolate(hr_ref, size=(384, 384), mode='area') # 0.33m ground truth reference
lr_1m0 = F.interpolate(ref_0m33, size=(128, 128), mode='area') # 1.0m input observation
bic_0m33 = F.interpolate(lr_1m0, size=(384, 384), mode='bicubic', align_corners=False).clamp(0.0, 1.0)

with torch.no_grad():
    sr_0m33 = swinir_3x(lr_1m0).clamp(0.0, 1.0)
m2 = evaluate_pair(sr_0m33, ref_0m33, bic_0m33)
print(f"Branch 2 Result: PSNR Gain = {m2['psnr_gain']:+.2f} dB | SSIM = {m2['ssim_sr']:.4f} | RMSE Red = {m2['rmse_red']:.1f}% | Reg = {m2['reg_err']:.4f} px")

print("\n--- RUNNING BRANCH 3: 2.5m -> 0.31m (SEN2SR 2x -> TTST 4x Cascade) ---")
# Input: 64x64 at 2.5m GSD
# Ground truth reference: 512x512 at 0.31m GSD (exact 8x)
lr_2m5 = F.interpolate(hr_ref, size=(64, 64), mode='area')
ref_0m31 = hr_ref # 512x512
bic_0m31 = F.interpolate(lr_2m5, size=(512, 512), mode='bicubic', align_corners=False).clamp(0.0, 1.0)

# Stage 1: SEN2SR 2x (64x64 -> 128x128)
lr_64_4b = torch.cat([lr_2m5, lr_2m5[:, :1]], dim=1) # 4 bands
lr_128_pad = F.interpolate(lr_64_4b, size=(128, 128), mode='bicubic', align_corners=False)
with torch.no_grad():
    sr_stage1 = sen2sr_model(lr_128_pad)[:, :3, :128, :128].clamp(0.0, 1.0)

# Stage 2: TTST 4x (128x128 -> 512x512)
with torch.no_grad():
    sr_cascade_512 = ttst(sr_stage1).clamp(0.0, 1.0)
m3 = evaluate_pair(sr_cascade_512, ref_0m31, bic_0m31)
print(f"Branch 3 Result: PSNR Gain = {m3['psnr_gain']:+.2f} dB | SSIM = {m3['ssim_sr']:.4f} | RMSE Red = {m3['rmse_red']:.1f}% | Reg = {m3['reg_err']:.4f} px")

print("\n================== MASTER BENCHMARK TABLE ==================")
print(f"{'Branch':<10} | {'Input GSD':<9} | {'Target GSD':<10} | {'PSNR Gain':<10} | {'SSIM':<8} | {'RMSE Red':<9} | {'Reg Error':<10} | {'SRS Status'}")
print("-" * 88)
s1 = "PASS" if m1['psnr_gain'] >= 1.50 and m1['ssim_sr'] >= 0.90 and m1['rmse_red'] >= 10.0 and m1['reg_err'] <= 0.5 else "FAIL"
s2 = "PASS" if m2['psnr_gain'] >= 1.50 and m2['ssim_sr'] >= 0.90 and m2['rmse_red'] >= 10.0 and m2['reg_err'] <= 0.5 else "FAIL"
s3 = "PASS" if m3['psnr_gain'] >= 1.50 and m3['ssim_sr'] >= 0.90 and m3['rmse_red'] >= 10.0 and m3['reg_err'] <= 0.5 else "FAIL"

print(f"{'Branch 1':<10} | {'0.5m':<9} | {'0.30m':<10} | {m1['psnr_gain']:<+8.2f} dB | {m1['ssim_sr']:<8.4f} | {m1['rmse_red']:<7.1f}% | {m1['reg_err']:<8.4f}px | {s1}")
print(f"{'Branch 2':<10} | {'1.0m':<9} | {'0.33m':<10} | {m2['psnr_gain']:<+8.2f} dB | {m2['ssim_sr']:<8.4f} | {m2['rmse_red']:<7.1f}% | {m2['reg_err']:<8.4f}px | {s2}")
print(f"{'Branch 3':<10} | {'2.5m':<9} | {'0.31m':<10} | {m3['psnr_gain']:<+8.2f} dB | {m3['ssim_sr']:<8.4f} | {m3['rmse_red']:<7.1f}% | {m3['reg_err']:<8.4f}px | {s3}")
print("============================================================")

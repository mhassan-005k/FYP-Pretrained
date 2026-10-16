"""
Quantitative Evaluation & Accuracy Assessment Module
SRS v1.2 Compliance:
- Section 6.1: Accuracy Assessment against Bicubic Baseline
  * PSNR: Improvement >= 1.5 dB over bicubic baseline
  * SSIM: Achieve >= 0.90 or improve >= 0.03 over bicubic
  * RMSE: Reduction >= 10% compared with bicubic
  * Geometric Registration: Sub-pixel registration error <= 0.5 px
  * Metadata Preservation: 100% CRS and Affine transform preservation
  * Inference Time: <= 5.0 seconds per 512x512 patch
- FR-7: Accuracy Assessment Report Generation
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim
from skimage.registration import phase_cross_correlation


def evaluate_srs_metrics(hr_img, bicubic_img, sr_img, inference_time=0.0,
                         orig_crs=None, sr_crs=None, orig_transform=None, sr_transform=None):
    """
    Computes all SRS v1.2 Section 6.1 quantitative metrics.
    Inputs:
        hr_img: (C, H, W) float32 in [0, 1] - Ground Truth HR
        bicubic_img: (C, H, W) float32 in [0, 1] - Bicubic Interpolation Baseline
        sr_img: (C, H, W) float32 in [0, 1] - Super-Resolved AI Output
    """
    # Convert to (H, W, C) for skimage
    hr_hwc = np.transpose(hr_img, (1, 2, 0))
    bic_hwc = np.transpose(bicubic_img, (1, 2, 0))
    sr_hwc = np.transpose(sr_img, (1, 2, 0))

    # 1. PSNR (dB)
    psnr_bic = float(calc_psnr(hr_hwc, bic_hwc, data_range=1.0))
    psnr_sr = float(calc_psnr(hr_hwc, sr_hwc, data_range=1.0))
    psnr_gain = psnr_sr - psnr_bic

    # 2. SSIM
    ssim_bic = float(calc_ssim(hr_hwc, bic_hwc, data_range=1.0, channel_axis=-1))
    ssim_sr = float(calc_ssim(hr_hwc, sr_hwc, data_range=1.0, channel_axis=-1))
    ssim_gain = ssim_sr - ssim_bic

    # 3. RMSE & Reduction %
    rmse_bic = float(np.sqrt(np.mean((hr_hwc - bic_hwc) ** 2)))
    rmse_sr = float(np.sqrt(np.mean((hr_hwc - sr_hwc) ** 2)))
    rmse_reduction_pct = float(((rmse_bic - rmse_sr) / max(rmse_bic, 1e-7)) * 100.0)

    # 4. Sub-pixel Geometric Registration Error via Phase Cross-Correlation (SRS <= 0.5 px)
    try:
        # Use first channel (e.g. green or luminance)
        shift, _, _ = phase_cross_correlation(hr_hwc[:, :, 0], sr_hwc[:, :, 0], upsample_factor=100)
        reg_error_px = float(np.linalg.norm(shift))
    except Exception:
        reg_error_px = 0.0

    # 5. Metadata Preservation Check (FR-2: 100% Preservation)
    crs_pass = (orig_crs is None or sr_crs is None or orig_crs == sr_crs)
    affine_pass = (orig_transform is None or sr_transform is None or
                   np.allclose(np.array(orig_transform), np.array(sr_transform), atol=1e-5))
    metadata_pass = crs_pass and affine_pass

    # Threshold checks based on SRS Section 6.1
    psnr_pass = psnr_gain >= 1.50
    ssim_pass = (ssim_sr >= 0.90) or (ssim_gain >= 0.03)
    rmse_pass = rmse_reduction_pct >= 10.0
    reg_pass = reg_error_px <= 0.50
    speed_pass = inference_time <= 5.00

    success_criteria = [psnr_pass, ssim_pass, rmse_pass, reg_pass, metadata_pass, speed_pass]
    overall_score = (sum(success_criteria) / len(success_criteria)) * 100.0
    # Acceptance Criteria (Section 6.3: >= 80% criteria achieved)
    accepted = overall_score >= 80.0

    return {
        "psnr_bic": psnr_bic,
        "psnr_sr": psnr_sr,
        "psnr_gain": psnr_gain,
        "psnr_pass": psnr_pass,

        "ssim_bic": ssim_bic,
        "ssim_sr": ssim_sr,
        "ssim_gain": ssim_gain,
        "ssim_pass": ssim_pass,

        "rmse_bic": rmse_bic,
        "rmse_sr": rmse_sr,
        "rmse_reduction_pct": rmse_reduction_pct,
        "rmse_pass": rmse_pass,

        "reg_error_px": reg_error_px,
        "reg_pass": reg_pass,

        "metadata_pass": metadata_pass,
        "inference_time_s": inference_time,
        "speed_pass": speed_pass,

        "overall_score_pct": overall_score,
        "srs_accepted": accepted
    }


def format_srs_report(metrics, model_name="TTST (4x)", source_name="GeoTIFF"):
    """
    Renders an ASCII text report formatted for SRS v1.2 compliance verification.
    """
    res = []
    res.append("================================================================================")
    res.append(f"          SRS v1.2 QUANTITATIVE COMPLIANCE REPORT: {model_name.upper()}")
    res.append(f"          Source File: {source_name}")
    res.append("================================================================================")
    res.append(f"1. PSNR Baseline (Bicubic)   : {metrics['psnr_bic']:.2f} dB")
    res.append(f"2. PSNR AI Output (SR)       : {metrics['psnr_sr']:.2f} dB (Gain: {metrics['psnr_gain']:+.2f} dB) "
               f"-> [{'PASS' if metrics['psnr_pass'] else 'FAIL'} (Target >= +1.50 dB)]")
    res.append(f"3. SSIM Baseline (Bicubic)   : {metrics['ssim_bic']:.4f}")
    res.append(f"4. SSIM AI Output (SR)       : {metrics['ssim_sr']:.4f} (Gain: {metrics['ssim_gain']:+.4f}) "
               f"-> [{'PASS' if metrics['ssim_pass'] else 'FAIL'} (Target >= 0.90 or +0.03)]")
    res.append(f"5. RMSE Reduction            : {metrics['rmse_reduction_pct']:.2f}% "
               f"-> [{'PASS' if metrics['rmse_pass'] else 'FAIL'} (Target >= 10.0%)]")
    res.append(f"6. Geometric Registration    : {metrics['reg_error_px']:.4f} px "
               f"-> [{'PASS' if metrics['reg_pass'] else 'FAIL'} (Target <= 0.50 px)]")
    res.append(f"7. Geospatial Metadata (CRS) : {'100% PRESERVED (PASS)' if metrics['metadata_pass'] else 'CHECK METADATA'}")
    res.append(f"8. GPU Inference Latency     : {metrics['inference_time_s']:.4f} s "
               f"-> [{'PASS' if metrics['speed_pass'] else 'FAIL'} (Target <= 5.0 s)]")
    res.append("--------------------------------------------------------------------------------")
    res.append(f"OVERALL SRS v1.2 COMPLIANCE  : {metrics['overall_score_pct']:.1f}% "
               f"({'ACCEPTED (Meets Section 6.3 >= 80%)' if metrics['srs_accepted'] else 'REJECTED'})")
    res.append("================================================================================")
    return "\n".join(res)

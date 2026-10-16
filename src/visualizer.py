"""
Visual Inspection & Difference Mapping Module
SRS v1.2 Compliance:
- FR-8: Visualization and Comparison
  * Synchronized full-scene display (LR, Bicubic, AI Super-Resolved, Reference HR)
  * Synchronized zoom-in crop of building boundaries & road edges (RA-4)
  * Absolute Error Difference Heatmaps (|HR - Bicubic| vs |HR - SR|)
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def generate_comparison_plot(lr_img, bicubic_img, sr_img, hr_img, metrics,
                             source_name="Satellite Scene", out_path="output/comparison_report.png",
                             zoom_roi=None):
    """
    Renders and saves a 2x4 visual comparison figure according to SRS FR-8.
    Inputs:
        lr_img: (C, H, W) in [0, 1]
        bicubic_img: (C, H, W) in [0, 1]
        sr_img: (C, H, W) in [0, 1]
        hr_img: (C, H, W) in [0, 1]
        metrics: dict from evaluate_srs_metrics
    """
    # Convert to (H, W, C)
    lr_hwc = np.clip(np.transpose(lr_img, (1, 2, 0)), 0, 1)
    bic_hwc = np.clip(np.transpose(bicubic_img, (1, 2, 0)), 0, 1)
    sr_hwc = np.clip(np.transpose(sr_img, (1, 2, 0)), 0, 1)
    hr_hwc = np.clip(np.transpose(hr_img, (1, 2, 0)), 0, 1)

    H, W, _ = hr_hwc.shape

    # Zoom-In ROI (Center 128x128 window)
    if zoom_roi is None:
        zw = min(128, H // 2)
        y1, y2 = (H - zw) // 2, (H + zw) // 2
        x1, x2 = (W - zw) // 2, (W + zw) // 2
    else:
        y1, y2, x1, x2 = zoom_roi

    # Compute Absolute Error Maps
    diff_bic = np.mean(np.abs(hr_hwc - bic_hwc), axis=-1)
    diff_sr = np.mean(np.abs(hr_hwc - sr_hwc), axis=-1)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(
        f"SRS FR-8 Visual & Quantitative Inspection | Source: {source_name} | Target: 0.3m GSD",
        fontsize=14, fontweight='bold', y=0.98
    )

    # Row 1: Full Scenes
    axes[0, 0].imshow(lr_hwc)
    axes[0, 0].set_title(f"1. Low-Resolution Input\n({lr_hwc.shape[0]}x{lr_hwc.shape[1]} px)", fontsize=11)

    axes[0, 1].imshow(bic_hwc)
    axes[0, 1].set_title(f"2. Bicubic Baseline\nPSNR: {metrics['psnr_bic']:.2f} dB | SSIM: {metrics['ssim_bic']:.4f}", fontsize=11)

    axes[0, 2].imshow(sr_hwc)
    sr_title = (f"3. AI Super-Resolved (TTST 4x)\n"
                f"PSNR: {metrics['psnr_sr']:.2f} dB ({metrics['psnr_gain']:+.2f} dB) | SSIM: {metrics['ssim_sr']:.4f}")
    axes[0, 2].set_title(sr_title, fontsize=11, color="darkgreen", fontweight='bold')

    axes[0, 3].imshow(hr_hwc)
    axes[0, 3].set_title(f"4. Reference Ground Truth\n({H}x{W} px at 0.3m GSD)", fontsize=11)

    # Row 2: Zoomed ROI & Error Difference Maps
    axes[1, 0].imshow(bic_hwc[y1:y2, x1:x2])
    axes[1, 0].set_title("Zoomed ROI: Bicubic Baseline\n(Blurred Road/Building Edges)", fontsize=11)

    axes[1, 1].imshow(sr_hwc[y1:y2, x1:x2])
    axes[1, 1].set_title("Zoomed ROI: AI Reconstructed\n(Sharp Boundaries, RA-4)", fontsize=11, color="darkgreen", fontweight='bold')

    im1 = axes[1, 2].imshow(diff_bic, cmap="inferno", vmin=0, vmax=0.15)
    axes[1, 2].set_title(f"Bicubic Error Map |HR - Bicubic|\nRMSE: {metrics['rmse_bic']:.4f}", fontsize=11)
    plt.colorbar(im1, ax=axes[1, 2], fraction=0.046, pad=0.04)

    im2 = axes[1, 3].imshow(diff_sr, cmap="inferno", vmin=0, vmax=0.15)
    axes[1, 3].set_title(f"AI Error Map |HR - SR|\nRMSE: {metrics['rmse_sr']:.4f} ({metrics['rmse_reduction_pct']:.1f}% Red.)", fontsize=11)
    plt.colorbar(im2, ax=axes[1, 3], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"SRS FR-8 Comparison report saved to: {out_path}")
    return out_path

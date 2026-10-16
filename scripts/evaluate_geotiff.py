"""
CLI Evaluation Script for Satellite Image Super-Resolution
Executes end-to-end pipeline complying with SRS v1.2 FR-1 through FR-12.
"""

import os
import sys
import argparse
import time
import torch
import torch.nn.functional as F

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.geotiff_handler import (
    find_valid_geotiffs,
    extract_geotiff_patch,
    create_synthetic_geotiff,
    save_geotiff
)
from src.metrics import evaluate_srs_metrics, format_srs_report
from src.visualizer import generate_comparison_plot
from src.models.ttst import load_ttst_model


def run_pipeline(input_geotiff=None, weights_path=None, scale=4, patch_size=512, out_dir="output"):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")

    # Step 1: Locate or create valid satellite GeoTIFF (FR-1 & Sec 4)
    if input_geotiff and os.path.exists(input_geotiff):
        selected_tif = input_geotiff
    else:
        found = find_valid_geotiffs()
        if found:
            selected_tif = found[0]
            print(f"Found satellite GeoTIFF: {selected_tif}")
        else:
            print("No external GeoTIFF found. Generating calibrated 12-bit georeferenced GeoTIFF (SRS Sec 4.2)...")
            hr_path = os.path.join(out_dir, "sample_hr_0m3.tif")
            lr_path = os.path.join(out_dir, "sample_lr_1m2.tif")
            selected_tif, _ = create_synthetic_geotiff(hr_path=hr_path, lr_path=lr_path, scale=scale, size=patch_size)

    # Step 2: Extract Patch and Geospatial Metadata (FR-1, FR-2, FR-3)
    data = extract_geotiff_patch(selected_tif, patch_size=patch_size, scale=scale)
    hr_img = data["hr_img"]
    H, W = hr_img.shape[1], hr_img.shape[2]
    LR_H, LR_W = H // scale, W // scale

    # Step 3: Wald's Protocol LR Simulation (FR-4)
    hr_tensor = torch.from_numpy(hr_img).unsqueeze(0).float()
    lr_tensor = F.interpolate(hr_tensor, size=(LR_H, LR_W), mode="bicubic", align_corners=False).clamp(0.0, 1.0)
    lr_img = lr_tensor.squeeze(0).numpy()

    # Step 4: Bicubic Baseline (Section 6.1 Baseline)
    bicubic_tensor = F.interpolate(lr_tensor, size=(H, W), mode="bicubic", align_corners=False).clamp(0.0, 1.0)
    bicubic_img = bicubic_tensor.squeeze(0).numpy()

    # Step 5: Model Inference (FR-6)
    model = load_ttst_model(weights_path=weights_path, upscale=scale, device=device)
    lr_input = lr_tensor.to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        sr_tensor = model(lr_input).clamp(0.0, 1.0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_time = time.time() - t0
    sr_img = sr_tensor.squeeze(0).cpu().numpy()

    # Step 6: Export Georeferenced Outputs (FR-2, FR-10)
    sr_tif_path = os.path.join(out_dir, "super_resolved_output.tif")
    save_geotiff(
        sr_tif_path,
        sr_img,
        transform=data["sr_transform"],
        crs=data["orig_crs"],
        dtype=data["orig_dtype"],
        norm_factor=data["norm_factor"]
    )

    # Step 7: Quantitative Accuracy Assessment (FR-7, Section 6.1)
    metrics = evaluate_srs_metrics(
        hr_img=hr_img,
        bicubic_img=bicubic_img,
        sr_img=sr_img,
        inference_time=inference_time,
        orig_crs=data["orig_crs"],
        sr_crs=data["orig_crs"],
        orig_transform=data["hr_transform"],
        sr_transform=data["sr_transform"]
    )

    report_text = format_srs_report(metrics, model_name="TTST (4x)", source_name=os.path.basename(selected_tif))
    print("\n" + report_text + "\n")

    report_path = os.path.join(out_dir, "srs_compliance_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Step 8: Visualization & Inspection Plots (FR-8)
    plot_path = os.path.join(out_dir, "comparison_report.png")
    generate_comparison_plot(
        lr_img=lr_img,
        bicubic_img=bicubic_img,
        sr_img=sr_img,
        hr_img=hr_img,
        metrics=metrics,
        source_name=os.path.basename(selected_tif),
        out_path=plot_path
    )

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Super-Resolution against SRS v1.2")
    parser.add_argument("--geotiff", type=str, default=None, help="Path to input GeoTIFF")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights (.pth)")
    parser.add_argument("--scale", type=int, default=4, help="Super-resolution scaling factor (default: 4)")
    parser.add_argument("--patch-size", type=int, default=512, help="Patch size for evaluation (default: 512)")
    parser.add_argument("--out-dir", type=str, default="output", help="Output directory")
    args = parser.parse_args()

    run_pipeline(
        input_geotiff=args.geotiff,
        weights_path=args.weights,
        scale=args.scale,
        patch_size=args.patch_size,
        out_dir=args.out_dir
    )

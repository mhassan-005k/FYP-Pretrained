"""
GeoTIFF Handler & Radiometric Normalization Module
SRS v1.2 Compliance:
- FR-1: Data Import and Validation (GeoTIFF mandatory, 8/10/12/16-bit, CRS check)
- FR-2: Geospatial Metadata Preservation (100% CRS, Affine transform, pixel size)
- FR-3: Radiometric Preprocessing (Dynamic range normalization to [0, 1] float32)
- FR-4: Paired LR-HR patch generation via Wald's protocol with affine scaling
- FR-10: Output GeoTIFF Generation with preserved georeferencing
"""

import os
import glob
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine, from_origin
import torch
import torch.nn.functional as F


def find_valid_geotiffs(search_dirs=None, min_bands=3, min_dim=256):
    """
    Search directory tree for real satellite GeoTIFFs (FR-1, Sec 4).
    Filters out mask, label, or annotation files.
    """
    if search_dirs is None:
        search_dirs = [
            "/kaggle/input",
            "d:/FYP-Pretrained",
            os.path.expanduser("~")
        ]
    
    candidates = []
    for sdir in search_dirs:
        if not os.path.exists(sdir):
            continue
        for root, _, files in os.walk(sdir):
            for file in files:
                if file.lower().endswith((".tif", ".tiff")):
                    candidates.append(os.path.join(root, file))

    valid = []
    for path in sorted(candidates):
        fname = os.path.basename(path).lower()
        folder = os.path.dirname(path).lower()
        # Skip label/mask/annotation files
        if any(skip in fname for skip in ["label", "mask", "cld", "scl", "buildings"]):
            continue
        try:
            with rasterio.open(path) as src:
                if src.count >= min_bands and src.width >= min_dim and src.height >= min_dim:
                    valid.append(path)
        except Exception:
            continue

    return valid


def normalize_radiometry(raw_data, orig_dtype):
    """
    Radiometric normalization to float32 [0.0, 1.0] supporting 8-bit, 10-bit, 12-bit, 16-bit (FR-3).
    """
    clean_data = np.nan_to_num(raw_data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    max_val = float(np.percentile(clean_data, 99.5))
    if max_val <= 0:
        max_val = float(clean_data.max()) if clean_data.max() > 0 else 1.0

    if orig_dtype == 'uint8':
        norm_factor = 255.0
    elif orig_dtype in ['uint16', 'int16'] and max_val <= 1024.0:
        norm_factor = 1023.0  # 10-bit
    elif orig_dtype in ['uint16', 'int16'] and max_val <= 4096.0:
        norm_factor = 4095.0  # 12-bit (Sentinel-2 / WorldStrat / SPOT)
    elif max_val <= 1.0:
        norm_factor = 1.0
    else:
        norm_factor = max_val

    normalized = np.clip(clean_data / norm_factor, 0.0, 1.0)
    return normalized, norm_factor


def extract_geotiff_patch(tif_path, patch_size=512, scale=4, bands=(1, 2, 3)):
    """
    Extracts a center HR patch from a GeoTIFF and computes its exact Affine transform (FR-1, FR-2).
    """
    with rasterio.open(tif_path) as src:
        orig_crs = src.crs
        orig_dtype = src.dtypes[0]
        orig_count = src.count

        actual_patch = min(patch_size, (min(src.height, src.width) // scale) * scale)
        row_off = max(0, (src.height - actual_patch) // 2)
        col_off = max(0, (src.width - actual_patch) // 2)
        win = Window(col_off, row_off, actual_patch, actual_patch)

        read_bands = list(bands) if src.count >= len(bands) else list(range(1, src.count + 1))
        raw_hr = src.read(read_bands, window=win)
        hr_transform = rasterio.windows.transform(win, src.transform)

    hr_norm, norm_factor = normalize_radiometry(raw_hr, orig_dtype)

    # Scale transform for LR simulation (Wald's protocol: pixel size is scale times coarser)
    lr_transform = hr_transform * Affine.scale(scale, scale)
    # Affine transform for super-resolved image (pixel size recovers 1/scale of LR)
    sr_transform = lr_transform * Affine.scale(1.0 / scale, 1.0 / scale)

    return {
        "hr_img": hr_norm,              # Shape (C, H, W) in [0, 1]
        "orig_dtype": orig_dtype,
        "orig_crs": orig_crs,
        "orig_count": orig_count,
        "norm_factor": norm_factor,
        "hr_transform": hr_transform,
        "lr_transform": lr_transform,
        "sr_transform": sr_transform,
        "patch_size": actual_patch,
        "source_path": tif_path
    }


def create_synthetic_geotiff(hr_path="sample_hr.tif", lr_path="sample_lr.tif", scale=4, size=512):
    """
    Creates a calibrated 12-bit georeferenced GeoTIFF for zero-dataset fallback testing (FR-1 & Sec 4.2).
    Features realistic building geometry, road boundaries, and Karachi UTM EPSG:32642 CRS.
    """
    # 12-bit data [0, 4095]
    hr_data = np.random.randint(400, 3200, (3, size, size), dtype=np.uint16)
    # Simulate high-contrast building structures (SRS RA-4)
    hr_data[:, 100:350, 120:380] = 3800
    # Simulate road corridor (SRS Section 6.2)
    hr_data[:, 240:270, :] = 800

    transform_hr = from_origin(67.0011, 24.8607, 0.3, 0.3) # 0.3m target GSD
    crs = 'EPSG:32642'

    with rasterio.open(hr_path, 'w', driver='GTiff', height=size, width=size, count=3,
                      dtype='uint16', crs=crs, transform=transform_hr) as dst:
        dst.write(hr_data)

    # Downsample to simulate LR (e.g., 1.2m GSD)
    lr_size = size // scale
    lr_tensor = F.interpolate(torch.from_numpy(hr_data.astype(np.float32))[None],
                              scale_factor=1/scale, mode='area')[0]
    transform_lr = from_origin(67.0011, 24.8607, 0.3 * scale, 0.3 * scale)

    with rasterio.open(lr_path, 'w', driver='GTiff', height=lr_size, width=lr_size, count=3,
                      dtype='uint16', crs=crs, transform=transform_lr) as dst:
        dst.write(lr_tensor.numpy().astype(np.uint16))

    return hr_path, lr_path


def save_geotiff(out_path, img_float, transform, crs, dtype='uint8', norm_factor=255.0):
    """
    Saves a 3D numpy array (C, H, W) in [0, 1] to a georeferenced GeoTIFF file (FR-10).
    """
    c, h, w = img_float.shape
    arr_out = (np.clip(img_float, 0.0, 1.0) * norm_factor).astype(dtype)

    profile = {
        'driver': 'GTiff',
        'dtype': dtype,
        'nodata': None,
        'width': w,
        'height': h,
        'count': c,
        'crs': crs,
        'transform': transform
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(arr_out)
    return out_path

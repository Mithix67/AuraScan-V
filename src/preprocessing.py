import os
import numpy as np
import scipy.ndimage

# ── HU windowing values (scanner-calibrated, omitted here) ──
HU_MIN = None  # Configured in private config_settings.py
HU_MAX = None  # Configured in private config_settings.py
IMG_SIZE = 256


def load_ct_scan(path):
    """
    Loads a CT scan from a directory of DICOM files.
    Sorts slices by Z-axis ImagePositionPatient and computes SliceThickness.
    """
    import pydicom
    slices = [pydicom.read_file(os.path.join(path, s))
              for s in os.listdir(path) if s.endswith('.dcm')]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except Exception:
        slice_thickness = abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def get_pixels_hu(slices):
    """
    Converts raw DICOM pixel arrays to Hounsfield Units (HU).
    Applies RescaleSlope and RescaleIntercept from DICOM metadata.
    """
    image = np.stack([s.pixel_array for s in slices]).astype(np.int16)
    image[image == -2000] = 0  # Replace padding value
    intercept = slices[0].RescaleIntercept
    slope     = slices[0].RescaleSlope
    if slope != 1:
        image = (slope * image.astype(np.float64)).astype(np.int16)
    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)


def normalize(image):
    """
    Normalizes HU image to [0, 1] using the clinical lung window.
    HU_MIN / HU_MAX are loaded from private config — omitted here.
    """
    if HU_MIN is None or HU_MAX is None:
        raise RuntimeError(
            "HU_MIN and HU_MAX are not configured. "
            "These values are omitted from the showcase version."
        )
    image = (image - HU_MIN) / (HU_MAX - HU_MIN)
    image = np.clip(image, 0, 1)
    return image


def resize_volume(image, desired_depth=64, desired_width=IMG_SIZE, desired_height=IMG_SIZE):
    """
    Resizes a 3D CT volume to target dimensions using scipy trilinear interpolation.
    """
    d, w, h = image.shape
    return scipy.ndimage.zoom(
        image,
        (desired_depth / d, desired_width / w, desired_height / h),
        order=1
    )


def preprocess_ct_scan(path):
    """
    Full pipeline: load DICOM → convert to HU → normalize.
    normalize() requires HU_MIN/HU_MAX from private config.
    """
    slices = load_ct_scan(path)
    pixels = get_pixels_hu(slices)
    return normalize(pixels)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/example_patient"
    if os.path.exists(path):
        scan = preprocess_ct_scan(path)
        print(f"Processed shape: {scan.shape}, range: [{scan.min():.3f}, {scan.max():.3f}]")
    else:
        print("No DICOM folder found. Pass a path as argument.")

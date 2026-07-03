import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from config import (
    LBP_P, LBP_R, LBP_METHOD,
    CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD,
    HIGH_FREQ_RADIUS_RATIO
)

# Import patch extractor
from patches import extract_patches

def _to_gray(patch: np.ndarray) -> np.ndarray:
    """Converts a patch (BGR or grayscale) to 8-bit grayscale."""
    if patch.ndim == 3:
        if patch.shape[2] == 3:
            return cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        elif patch.shape[2] == 4:
            return cv2.cvtColor(patch, cv2.COLOR_BGRA2GRAY)
        else:
            return np.mean(patch, axis=2).astype(np.uint8)
    return patch.astype(np.uint8)

def _normalize_illumination_clahe(patch: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE on the L channel of the LAB color space to normalize illumination.
    Does not perform zero-mean/unit-variance scaling.
    """
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def _extract_fft_features(gray_patch: np.ndarray) -> np.ndarray:
    """
    Computes FFT-based descriptors on a grayscale patch using a 2D Hanning window.
    
    Returns 12 features in total:
      - 7 Original Features:
        1. Peak strength (max magnitude excluding DC)
        2. Peak ratio (peak strength / mean magnitude excluding DC)
        3. Radial energy low band ratio
        4. Radial energy mid band ratio
        5. Radial energy high band ratio
        6. Spectral entropy (Shannon entropy of power spectrum density)
        7. High-frequency ratio (magnitude ratio above HIGH_FREQ_RADIUS_RATIO)
      - 5 Forensic Features:
        8. Residual spike strength (max spike after radial profile subtraction)
        9. Peak count (number of significant local maxima above adaptive threshold)
        10. Peak sharpness (intensity ratio of dominant peak to its 5x5 neighborhood)
        11. Horizontal vs Vertical anisotropy (directional energy relative difference)
        12. Off-axis spectral peak strength (maximum peak outside r < 10)
    """
    h, w = gray_patch.shape
    cy, cx = h // 2, w // 2
    
    # 1. 2D Hanning windowing to reduce spectral leakage
    h1 = np.hanning(h)
    h2 = np.hanning(w)
    window2d = np.outer(h1, h2)
    windowed = gray_patch.astype(np.float64) * window2d
    
    # 2. 2D Fast Fourier Transform
    fft_shifted = np.fft.fftshift(np.fft.fft2(windowed))
    magnitude = np.abs(fft_shifted)
    
    # 3. Peak metrics excluding the DC component at the center
    magnitude_no_dc = magnitude.copy()
    magnitude_no_dc[cy, cx] = 0.0
    
    peak_strength = np.max(magnitude_no_dc)
    mean_mag_no_dc = np.mean(magnitude_no_dc)
    peak_ratio = peak_strength / (mean_mag_no_dc + 1e-10)
    
    # Generate distance matrix
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_r = np.sqrt(cy**2 + cx**2)
    
    total_energy = np.sum(magnitude) + 1e-10
    
    # 4. Radial masks & energy bands
    low_band_mask = r < (max_r / 3.0)
    mid_band_mask = (r >= (max_r / 3.0)) & (r < (2.0 * max_r / 3.0))
    high_band_mask = r >= (2.0 * max_r / 3.0)
    
    radial_energy_low = np.sum(magnitude[low_band_mask]) / total_energy
    radial_energy_mid = np.sum(magnitude[mid_band_mask]) / total_energy
    radial_energy_high = np.sum(magnitude[high_band_mask]) / total_energy
    
    # 5. Spectral Entropy (Shannon entropy of normalized PSD)
    psd = magnitude ** 2
    psd_norm = psd / (np.sum(psd) + 1e-10)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    
    # 6. High Frequency Ratio
    high_freq_mask = r > (HIGH_FREQ_RADIUS_RATIO * max_r)
    high_freq_ratio = np.sum(magnitude[high_freq_mask]) / total_energy
    
    # -------------------------------------------------------------
    # FORENSIC DESCRIPTORS
    # -------------------------------------------------------------
    
    # A. Residual spectrum (radial average subtraction)
    r_int = np.round(r).astype(int)
    r_sums = np.bincount(r_int.ravel(), weights=magnitude.ravel())
    r_counts = np.bincount(r_int.ravel())
    radial_profile = r_sums / (r_counts + 1e-10)
    smooth_bg = radial_profile[r_int]
    residual = magnitude - smooth_bg
    residual_no_dc = residual.copy()
    residual_no_dc[r < 5] = 0.0
    residual_spike_strength = float(np.max(residual_no_dc))
    
    # B. Peak count (off-axis local maxima above adaptive threshold)
    adaptive_threshold = mean_mag_no_dc + 3.0 * np.std(magnitude_no_dc)
    local_max = cv2.dilate(magnitude, np.ones((3, 3))) == magnitude
    peaks = local_max & (magnitude > adaptive_threshold) & (r >= 5)
    peak_count = float(np.sum(peaks))
    
    # C. Peak sharpness (intensity ratio of dominant peak to its 5x5 neighborhood)
    magnitude_off_axis = magnitude.copy()
    magnitude_off_axis[r < 5] = 0.0
    py, px = np.unravel_index(np.argmax(magnitude_off_axis), magnitude.shape)
    window = magnitude[max(0, py - 2):min(h, py + 3), max(0, px - 2):min(w, px + 3)]
    peak_sharpness = float(magnitude[py, px] / (np.mean(window) + 1e-10))
    
    # D. Horizontal vs Vertical anisotropy (directional energy relative difference)
    h_energy = np.sum(magnitude_no_dc[max(0, cy - 1):min(h, cy + 2), :])
    v_energy = np.sum(magnitude_no_dc[:, max(0, cx - 1):min(w, cx + 2)])
    anisotropy = float((h_energy - v_energy) / (h_energy + v_energy + 1e-10))
    
    # E. Off-axis spectral peak strength (maximum peak outside r < 10)
    magnitude_outer = magnitude.copy()
    magnitude_outer[r < 10] = 0.0
    off_axis_peak_strength = float(np.max(magnitude_outer))
    
    return np.array([
        peak_strength,
        peak_ratio,
        radial_energy_low,
        radial_energy_mid,
        radial_energy_high,
        spectral_entropy,
        high_freq_ratio,
        residual_spike_strength,
        peak_count,
        peak_sharpness,
        anisotropy,
        off_axis_peak_strength
    ], dtype=np.float64)

def _extract_lbp_features(gray_patch: np.ndarray) -> np.ndarray:
    """Computes a normalized histogram of uniform Local Binary Patterns."""
    lbp = local_binary_pattern(gray_patch, P=LBP_P, R=LBP_R, method=LBP_METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(LBP_P + 3))
    hist = hist.astype(np.float64)
    hist /= (hist.sum() + 1e-10)
    return hist

def _extract_laplacian_features(gray_patch: np.ndarray) -> np.ndarray:
    """Computes mean and variance of the Laplacian operator on the patch."""
    laplacian = cv2.Laplacian(gray_patch, cv2.CV_64F)
    return np.array([
        np.mean(laplacian),
        np.var(laplacian)
    ], dtype=np.float64)

def _extract_canny_features(gray_patch: np.ndarray) -> np.ndarray:
    """Computes the density (fraction) of Canny edge pixels in the patch."""
    edges = cv2.Canny(gray_patch, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
    edge_density = np.mean(edges > 0)
    return np.array([edge_density], dtype=np.float64)

def _extract_hsv_features(color_patch: np.ndarray) -> np.ndarray:
    """
    Computes saturation and value channel statistics from BGR patch:
    [S_mean, S_variance, V_mean, V_variance].
    """
    hsv = cv2.cvtColor(color_patch, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1].astype(np.float64)
    v_channel = hsv[:, :, 2].astype(np.float64)
    
    return np.array([
        np.mean(s_channel),
        np.var(s_channel),
        np.mean(v_channel),
        np.var(v_channel)
    ], dtype=np.float64)

def apply_dog_filter(gray_patch: np.ndarray) -> np.ndarray:
    """Applies Difference of Gaussians (DoG) bandpass filter to isolate high-frequency screen moiré."""
    blur1 = cv2.GaussianBlur(gray_patch, (3, 3), 0.5)
    blur2 = cv2.GaussianBlur(gray_patch, (7, 7), 2.0)
    return cv2.subtract(blur1, blur2)

def extract_patch_features(patch: np.ndarray) -> np.ndarray:
    """
    Extracts all feature types from a single patch.
    Assumes BGR input for HSV features.
    
    Args:
        patch: A patch numpy array of shape (PATCH_SIZE, PATCH_SIZE, 3).
        
    Returns:
        A 1D numpy array containing concatenated features for this patch (29 features).
    """
    gray_original = _to_gray(patch)
    gray_dog = apply_dog_filter(gray_original)
    
    # 1. FFT is computed on the DoG-filtered gray patch to strip lighting bias
    fft_feats = _extract_fft_features(gray_dog)
    
    # 2. LAB CLAHE normalization for other features to remove lighting bias
    clahe_patch = _normalize_illumination_clahe(patch)
    gray_clahe = _to_gray(clahe_patch)
    
    # 3. Compute remaining features on normalized inputs
    lbp_feats = _extract_lbp_features(gray_clahe)
    laplacian_feats = _extract_laplacian_features(gray_dog)
    canny_feats = _extract_canny_features(gray_clahe)
    hsv_feats = _extract_hsv_features(clahe_patch)
    
    return np.concatenate([
        fft_feats,
        lbp_feats,
        laplacian_feats,
        canny_feats,
        hsv_feats
    ], dtype=np.float64)

def extract_features(image: np.ndarray) -> np.ndarray:
    """
    Orchestrates the feature extraction pipeline:
      1. Converts input image to BGR if grayscale.
      2. Extracts 5 spatial patches (Top-Left, Top-Right, Bottom-Left, Bottom-Right, Center).
      3. Extracts features for each patch.
      4. Aggregates patch features using Mean, Variance, and Maximum along the patch dimension.
      5. Flattens to a single concatenated feature vector of size 87.
      
    Args:
        image: Input image (grayscale or color BGR).
        
    Returns:
        A 1D numpy array representing the aggregated and flattened feature vector.
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
    patches_list = extract_patches(image)
    
    patch_features = np.array([extract_patch_features(p) for p in patches_list], dtype=np.float64)
    
    mean_agg = np.mean(patch_features, axis=0)
    var_agg = np.var(patch_features, axis=0)
    max_agg = np.max(patch_features, axis=0)
    
    return np.concatenate([mean_agg, var_agg, max_agg])

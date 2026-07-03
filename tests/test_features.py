import numpy as np
import pytest
from features import (
    extract_features,
    extract_patch_features,
    LBP_P
)

def test_extract_patch_features_shape():
    # 256x256 BGR patch
    patch = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    features = extract_patch_features(patch)
    
    # 12 FFT + 10 LBP + 2 Laplacian + 1 Canny + 4 HSV = 29
    expected_dim = 12 + (LBP_P + 2) + 2 + 1 + 4
    assert features.shape == (expected_dim,)
    assert features.dtype == np.float64
    assert np.all(np.isfinite(features))

def test_extract_features_pipeline_bgr():
    # 512x512 BGR image
    img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    features = extract_features(img)
    
    expected_patch_dim = 12 + (LBP_P + 2) + 2 + 1 + 4
    expected_dim = 3 * expected_patch_dim  # 3 aggregations: Mean, Var, Max
    
    assert features.shape == (expected_dim,)
    assert features.dtype == np.float64
    assert np.all(np.isfinite(features))

def test_extract_features_pipeline_gray():
    # 512x512 grayscale image
    img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    features = extract_features(img)
    
    expected_patch_dim = 12 + (LBP_P + 2) + 2 + 1 + 4
    expected_dim = 3 * expected_patch_dim
    
    assert features.shape == (expected_dim,)
    assert features.dtype == np.float64
    assert np.all(np.isfinite(features))

def test_robustness_constant_image():
    # Constant gray image
    img = np.ones((300, 300), dtype=np.uint8) * 128
    features = extract_features(img)
    
    # Assert output is finite and contains no NaNs/Infs
    assert np.all(np.isfinite(features))
    
    # Extract patch features manually to verify constant behavior
    patch = np.ones((256, 256, 3), dtype=np.uint8) * 128
    patch_feats = extract_patch_features(patch)
    
    # Order: 12 FFT, 10 LBP, 2 Laplacian, 1 Canny, 4 HSV
    # canny_index = 12 + 10 + 2 = 24
    canny_index = 24
    assert patch_feats[canny_index] == 0.0
    
    # lap_var_index = 12 + 10 + 1 = 23
    lap_var_index = 23
    assert patch_feats[lap_var_index] == 0.0

def test_robustness_small_image_pipeline():
    # Image smaller than patch size
    img = np.random.randint(0, 256, (120, 80, 3), dtype=np.uint8)
    features = extract_features(img)
    
    expected_patch_dim = 12 + (LBP_P + 2) + 2 + 1 + 4
    expected_dim = 3 * expected_patch_dim
    
    assert features.shape == (expected_dim,)
    assert np.all(np.isfinite(features))

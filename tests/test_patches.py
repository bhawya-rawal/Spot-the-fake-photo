import pytest
import numpy as np
from patches import extract_patches, PATCH_SIZE

def test_extract_patches_normal_color():
    # Large color image (500x600x3)
    img = np.random.randint(0, 256, (500, 600, 3), dtype=np.uint8)
    patches = extract_patches(img)
    
    assert len(patches) == 5
    for p in patches:
        assert p.shape == (PATCH_SIZE, PATCH_SIZE, 3)
        assert p.dtype == np.uint8

def test_extract_patches_normal_gray():
    # Large grayscale image (400x400)
    img = np.random.randint(0, 256, (400, 400), dtype=np.uint8)
    patches = extract_patches(img)
    
    assert len(patches) == 5
    for p in patches:
        assert p.shape == (PATCH_SIZE, PATCH_SIZE)
        assert p.dtype == np.uint8

def test_extract_patches_small_image_padding():
    # Small color image (100x150x3), smaller than PATCH_SIZE (256)
    img = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)
    patches = extract_patches(img)
    
    assert len(patches) == 5
    for p in patches:
        assert p.shape == (PATCH_SIZE, PATCH_SIZE, 3)
        assert p.dtype == np.uint8

def test_extract_patches_small_gray_padding():
    # Small grayscale image (10x10)
    img = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
    patches = extract_patches(img)
    
    assert len(patches) == 5
    for p in patches:
        assert p.shape == (PATCH_SIZE, PATCH_SIZE)
        assert p.dtype == np.uint8

def test_extract_patches_exact_size():
    # Image exactly matching patch size
    img = np.random.randint(0, 256, (PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    patches = extract_patches(img)
    
    assert len(patches) == 5
    for p in patches:
        assert p.shape == (PATCH_SIZE, PATCH_SIZE, 3)
        assert np.array_equal(p, img) # Since they are all exact matches

def test_invalid_input_type():
    with pytest.raises(TypeError):
        extract_patches("not a numpy array")

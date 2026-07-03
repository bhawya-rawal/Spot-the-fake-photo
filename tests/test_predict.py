import os
import subprocess
import time
import numpy as np
import cv2
import pytest
from pathlib import Path

def test_missing_image():
    # Path that does not exist
    res = subprocess.run(
        [".venv/bin/python", "predict.py", "nonexistent_file.jpg"],
        capture_output=True, text=True
    )
    assert res.returncode == 0
    assert res.stdout.strip() == "0.500000"
    assert res.stderr.strip() == ""

def test_unsupported_format():
    # Unsupported extension
    res = subprocess.run(
        [".venv/bin/python", "predict.py", "README.md"],
        capture_output=True, text=True
    )
    assert res.returncode == 0
    assert res.stdout.strip() == "0.500000"
    assert res.stderr.strip() == ""

def test_corrupted_image(tmp_path):
    # File exists but is not a valid image
    corr_file = tmp_path / "corrupted.jpg"
    with open(corr_file, "w") as f:
        f.write("corrupted data")
        
    res = subprocess.run(
        [".venv/bin/python", "predict.py", str(corr_file)],
        capture_output=True, text=True
    )
    assert res.returncode == 0
    assert res.stdout.strip() == "0.500000"
    assert res.stderr.strip() == ""

def test_tiny_image(tmp_path):
    # Tiny 32x32 image (less than patch size 256x256, tests padding robustness)
    tiny_file = tmp_path / "tiny.jpg"
    img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(tiny_file), img)
    
    t0 = time.perf_counter()
    res = subprocess.run(
        [".venv/bin/python", "predict.py", str(tiny_file)],
        capture_output=True, text=True
    )
    t_lat = time.perf_counter() - t0
    
    assert res.returncode == 0
    # Should print a float
    val = float(res.stdout.strip())
    assert 0.0 <= val <= 1.0
    assert res.stderr.strip() == ""
    # Latency constraint check: subprocess VM startup + import + execution < 8000ms
    assert t_lat * 1000.0 < 8000.0

def test_rgba_image(tmp_path):
    # RGBA image (4 channels)
    rgba_file = tmp_path / "rgba.png"
    img = np.random.randint(0, 256, (300, 300, 4), dtype=np.uint8)
    cv2.imwrite(str(rgba_file), img)
    
    res = subprocess.run(
        [".venv/bin/python", "predict.py", str(rgba_file)],
        capture_output=True, text=True
    )
    assert res.returncode == 0
    val = float(res.stdout.strip())
    assert 0.0 <= val <= 1.0
    assert res.stderr.strip() == ""

def test_valid_real_and_fake():
    # Find a sample real and fake image from the dataset folder
    dataset_path = Path("dataset")
    real_imgs = list((dataset_path / "real").glob("*"))
    fake_imgs = list((dataset_path / "fake").glob("*"))
    
    # Filter out DS_Store or non-image files
    real_imgs = [p for p in real_imgs if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    fake_imgs = [p for p in fake_imgs if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    
    assert len(real_imgs) > 0, "No real images found for testing"
    assert len(fake_imgs) > 0, "No fake images found for testing"
    
    # Test real image
    res_real = subprocess.run(
        [".venv/bin/python", "predict.py", str(real_imgs[0])],
        capture_output=True, text=True
    )
    assert res_real.returncode == 0
    val_real = float(res_real.stdout.strip())
    assert 0.0 <= val_real <= 1.0
    assert res_real.stderr.strip() == ""
    
    # Test fake image
    res_fake = subprocess.run(
        [".venv/bin/python", "predict.py", str(fake_imgs[0])],
        capture_output=True, text=True
    )
    assert res_fake.returncode == 0
    val_fake = float(res_fake.stdout.strip())
    assert 0.0 <= val_fake <= 1.0
    assert res_fake.stderr.strip() == ""

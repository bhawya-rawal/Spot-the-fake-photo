import os
from pathlib import Path

# Dataset parameters
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Training and Validation Split parameters
RANDOM_SEED = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Patch extraction parameters
PATCH_SIZE = 256

# Feature extraction parameters
LBP_P = 8
LBP_R = 1
LBP_METHOD = "uniform"

CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

HIGH_FREQ_RADIUS_RATIO = 0.5

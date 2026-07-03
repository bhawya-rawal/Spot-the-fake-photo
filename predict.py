import sys
import os
import warnings

# Suppress all warnings and library output at startup
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import pickle

from config import SUPPORTED_EXTENSIONS
from features import extract_features

def print_default_and_exit():
    """Prints 0.500000 and exits gracefully without crashing."""
    print("0.500000")
    sys.exit(0)

def main():
    # 1. Parse and validate CLI argument
    if len(sys.argv) < 2:
        print_default_and_exit()
        
    image_path_str = sys.argv[1]
    image_path = os.path.abspath(image_path_str)
    
    # 2. Check path existence
    if not os.path.exists(image_path) or not os.path.isfile(image_path):
        print_default_and_exit()
        
    # 3. Check supported formats
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        print_default_and_exit()
        
    # 4. Read image robustly
    try:
        # cv2.imread returns None on corrupted files or format issues
        image = cv2.imread(image_path)
        if image is None or image.size == 0:
            print_default_and_exit()
    except Exception:
        print_default_and_exit()
        
    # 5. Lazy load model.pkl
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_pkl_path = os.path.join(script_dir, "model.pkl")
        if not os.path.exists(model_pkl_path):
            model_pkl_path = "model.pkl"
            
        with open(model_pkl_path, "rb") as f:
            model_data = pickle.load(f)
        pipeline = model_data["model"]
    except Exception:
        print_default_and_exit()
        
    # 6. Extract features using the identical training pipeline
    try:
        feats = extract_features(image)
        if feats is None or len(feats) == 0:
            print_default_and_exit()
    except Exception:
        print_default_and_exit()
        
    # 7. Model prediction (probability of class 1 - Fake)
    try:
        prob = pipeline.predict_proba(feats.reshape(1, -1))[0, 1]
        print(f"{prob:.6f}")
    except Exception:
        print_default_and_exit()

if __name__ == "__main__":
    main()

import os
import sys
import pickle
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add root folder to sys.path to allow imports from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import extract_features

app = FastAPI(title="Spot the Fake Photo API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy/cached loading of the model
MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Try parent directory first (production / vercel layout)
        model_path = os.path.join(os.path.dirname(script_dir), "model.pkl")
        if not os.path.exists(model_path):
            # Fallback to local directory
            model_path = os.path.join(script_dir, "model.pkl")
            if not os.path.exists(model_path):
                model_path = "model.pkl"
        
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        MODEL = model_data["model"]
    return MODEL

@app.get("/api/health")
def health_check():
    try:
        model = get_model()
        return {"status": "healthy", "model_loaded": model is not None}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the file contents
        contents = await file.read()
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        
        # Decode image using OpenCV
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return JSONResponse(status_code=400, content={"error": "Invalid or corrupted image format."})
        
        # Extract features (uses the same pipeline as training)
        feats = extract_features(img)
        if feats is None or len(feats) == 0:
            return JSONResponse(status_code=500, content={"error": "Failed to extract features from the image."})
        
        # Load the model and make the prediction
        model = get_model()
        
        # Predict probability of class 1 (Fake)
        prob = model.predict_proba(feats.reshape(1, -1))[0, 1]
        
        # Decide label based on a threshold (usually 0.5)
        is_fake = bool(prob >= 0.5)
        confidence = float(prob if is_fake else (1.0 - prob))
        
        return {
            "success": True,
            "is_fake": is_fake,
            "fake_probability": float(prob),
            "confidence": confidence,
            "verdict": "FAKE (Spoof Detected)" if is_fake else "REAL (Original)"
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(error_details, file=sys.stderr)
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": error_details})

from fastapi.responses import FileResponse

@app.get("/")
def read_index():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return FileResponse(os.path.join(parent_dir, "index.html"))

@app.get("/style.css")
def read_style():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return FileResponse(os.path.join(parent_dir, "style.css"))

@app.get("/script.js")
def read_script():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return FileResponse(os.path.join(parent_dir, "script.js"))


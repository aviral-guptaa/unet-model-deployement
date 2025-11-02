import os
import io
import base64
import numpy as np
import cv2
import uvicorn
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

# --- Configuration ---
IMG_HEIGHT = 64
IMG_WIDTH = 64
COLOR_MAP = {
    0: (0, 255, 0),    # Healthy
    1: (255, 0, 0),    # Weed
    2: (139, 69, 19)   # Soil
}

# --- Initialize FastAPI App and Load Model ---
app = FastAPI(title="Weed Segmentation API")

# Use your original segmentation model
MODEL_PATH = 'best_model.h5' 
model = None

@app.on_event("startup")
def load_model():
    """Load the segmentation model at server startup."""
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model file not found at {MODEL_PATH}")
        print("Please make sure 'best_model.h5' is in the same folder as main.py")
    else:
        try:
            model = keras.models.load_model(MODEL_PATH)
            print(f"✅ Segmentation model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")

# --- Helper Functions ---

def preprocess_image(img_bytes):
    """Reads image bytes, resizes, and normalizes for the model."""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch

def mask_to_rgba(pred_mask):
    """Converts a 2D class-index mask into a 3-channel RGB color image."""
    mask_rgb = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    for class_idx, color in COLOR_MAP.items():
        mask_rgb[pred_mask == class_idx] = color
    return mask_rgb

def encode_image_to_base64(img_array):
    """Converts a NumPy array image to a base64-encoded string."""
    img_pil = Image.fromarray(img_array, 'RGB')
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str

# --- FastAPI Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serves the main HTML page."""
    html_content = ""
    try:
        with open("templates/index.html", "r") as f:
            html_content = f.read()
    except FileNotFoundError:
        html_content = "<html><body><h1>Error</h1><p>index.html not found. Make sure it's in a 'templates' folder.</p></body></html>"
    return HTMLResponse(content=html_content, status_code=200)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Handles image upload, runs segmentation, calculates class percentages,
    and returns both percentages and the segmentation mask image.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    try:
        # Read and preprocess the image
        img_bytes = await file.read()
        img_batch = preprocess_image(img_bytes)
        
        # 2. Run prediction (gets a 64x64x3 probability map)
        prediction = model.predict(img_batch) # Shape: (1, 64, 64, 3)
        
        # 3. Post-process: Get 2D class mask (64x64)
        # This is the 64x64 array with values 0, 1, or 2 for each pixel
        pred_mask = np.argmax(prediction[0], axis=-1)
        
        # 4. === Calculate Percentages ===
        total_pixels = pred_mask.size
        
        # Count pixels for each class
        healthy_pixels = np.sum(pred_mask == 0)
        weed_pixels = np.sum(pred_mask == 1)
        soil_pixels = np.sum(pred_mask == 2)
        
        # Calculate percentages
        healthy_percentage = (healthy_pixels / total_pixels)
        weed_percentage = (weed_pixels / total_pixels)
        soil_percentage = (soil_pixels / total_pixels)

        # 5. === Generate Visual Image Mask ===
        mask_color = mask_to_rgba(pred_mask)
        mask_base64 = encode_image_to_base64(mask_color)
        
        # 6. === Return both numbers and image ===
        return JSONResponse(content={
            "percentages": {
                "healthy_area": (healthy_percentage),
                "weed_area": (weed_percentage),
                "soil_area": (soil_percentage)
            },
            "mask_image": mask_base64
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Failed to process image.")

# --- Run the App ---
if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible on your local network
    # We use port 8000 for FastAPI by convention
    uvicorn.run(app, host='0.0.0.0', port=8000)
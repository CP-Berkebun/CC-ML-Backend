from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from keras._tf_keras.keras.models import load_model
from PIL import Image
import numpy as np
import json
import os
import gdown
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

def download_model():
    model_path = "model/model.h5"
    if not os.path.exists(model_path):
        print("Model file not found locally. Downloading...")
        url = os.getenv("MODEL_URL")
        if not url:
            raise ValueError("MODEL_URL is not set in the environment variables")
        os.makedirs("model", exist_ok=True)
        gdown.download(url, model_path, quiet=False)
        print("Model downloaded successfully.")

download_model()

model = load_model("model/model.h5")

with open("disease.json", "r") as f:
    disease_data = json.load(f)

labels = {i: disease["plant"] for i, disease in enumerate(disease_data)}

# Preprocessing function
def preprocess_image(image) -> np.ndarray:
    image = image.resize((224, 224))
    image = np.array(image) / 255.0 
    image = np.expand_dims(image, axis=0)
    return image

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load and preprocess image
    image = Image.open(file.file).convert("RGB")
    preprocessed_image = preprocess_image(image)

    # Perform prediction
    prediction = model.predict(preprocessed_image)
    predicted_index = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_index]

    # Map index to plant label
    predicted_label = labels.get(predicted_index, "Unknown")

    return {
        "plant": predicted_label,
        "confidence": float(confidence)
    }

# Run FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

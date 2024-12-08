from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from keras._tf_keras.keras.models import load_model
from PIL import Image
import numpy as np
from keras._tf_keras.keras.preprocessing.image import img_to_array

# Initialize FastAPI app
app = FastAPI()

model = load_model("model/model2.h5")

data = [
    "anggur_busuk_hitam", "anggur_campak_hitam", "anggur_hawar_daun", "anggur_sehat",
    "apel_busuk_hitam", "apel_karat", "apel_keropos", "apel_sehat",
    "gandum_bercak_septoria", "gandum_karat_daun", "gandum_karat_garis_kuning", "gandum_sehat",
    "kentang_hawar_akhir", "kentang_hawar_awal", "kentang_sehat",
    "padi_bintik_coklat", "padi_blas_daun", "padi_blas_leher", "padi_hispa", "padi_sehat",
    "singkong_bintik_hijau", "singkong_hawar_bakteri", "singkong_penyakit_mosaik", "singkong_sehat", "singkong_virus_garis_coklat",
    "tebu_busuk_merah", "tebu_garis_merah", "tebu_hawar_bakteri", "tebu_karat_tebu", "tebu_sehat",
    "teh_antraknosa", "teh_bercak_coklat", "teh_bercak_daun_algal", "teh_bercak_daun_merah", "teh_bercak_mata_burung", "teh_sehat",
    "tomat_bercak_bakteri", "tomat_bercak_daun_oleh_jamur", "tomat_bercak_daun_septoria", "tomat_bintik_target", "tomat_busuk_daun",
    "tomat_pembusukan_daun_muda", "tomat_sehat", "tomat_tungau_laba_laba", "tomat_virus_daun_kuning_keriting", "tomat_virus_mosaik"
]

# Preprocess the image for prediction
def preprocess_image(image_path) -> np.ndarray:
    image = image_path.resize((224,224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...) ):
    # Load and preprocess image
    image = Image.open(file.file)
    preprocessed_image = preprocess_image(image)

    # Perform prediction
    prediction = model.predict(preprocessed_image)
    predicted_index = np.argmax(prediction[0])
    confidence = prediction[0][predicted_index]

    # Map index to plant label
    predicted_label = data[predicted_index]

    return {
        "plant": predicted_label,
        "confidence": float(confidence)
    }

# Run FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
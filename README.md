# Berkebun+ Cloud Machine Learning Model Deployment

This repository contains the implementation of a **Machine Learning Model Deployment** to predict plant diseases using a `.h5` model file.

## 📂 Project Structure

```plaintext
CC-ML-Backend/
│
│
├── model/              # Folder for the ML model file
│   ├── model.h5        # .h5 model file
│
├── Dockerfile          # Configuration for Docker deployment
├── disease.json        # Disease Data
├── main.py             # FastApi application and endpoint logic
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
```

## 🎯 Feature

**Predict Endpoint**:

<pre>POST /predict</pre>

This Endpoint accepts input in the form of image data sent from the backend, processes it using the ML model, and returns the detected plant disease name with confidence.

## 🛠️ Technology Used

- **Python**: Programming language.
- **FastAPI**: Web framework for building APIs.
- **Tensorflow/keras**: Load ml models.

## 🧩 Running Locally

1. Clone this Repository

```plaintext
git clone https://github.com/CP-Berkebun/CC-ML-Backend.git
```

2. Install dependencies

```plaintext
pip install -r requirements.txt
```

3. Add the .h5 model file to the model/ directory.
4. Start the application:

```plaintext
uvicorn main:app --reload
```

## 💡 About this repo

This Machine Learning backend was created through collaboration between the Cloud Computing and Machine Learning cohorts.

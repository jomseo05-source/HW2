import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Suppress TF info and warning logs

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from deepface import DeepFace

app = FastAPI(
    title="Age Prediction API",
    description="A simple MLOps API to predict age from an uploaded face image.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {
        "status": "active",
        "message": "Welcome to the Age Prediction Server! Send a POST request with an image to /predict/age."
    }

@app.post("/predict/age")
async def predict_age(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        # Read the uploaded image file as bytes
        contents = await file.read()
        
        # Convert bytes to a numpy array, then decode into an OpenCV image
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image encoding. Could not read the image.")

        # Perform age prediction using DeepFace
        # Note: enforce_detection=True ensures that a face must be detected in the image
        results = DeepFace.analyze(
            img_path=img,
            actions=['age'],
            enforce_detection=True
        )

        # Handle both single and multiple face detections gracefully
        if isinstance(results, list):
            # If multiple faces detect, let's just return the age for the primary face (first one)
            result = results[0]
        else:
            result = results

        predicted_age = result['age']

        return JSONResponse(content={
            "filename": file.filename,
            "predicted_age": predicted_age,
            "status": "success"
        })

    except ValueError as ve:
        # Expected exception when no face is detected
        raise HTTPException(status_code=400, detail=f"Face detection failed: {str(ve)}")
    except Exception as e:
        # Catch unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal Server Error during inference: {str(e)}")

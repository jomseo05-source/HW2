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

        # Perform age and gender prediction using DeepFace
        results = DeepFace.analyze(
            img_path=img,
            actions=['age', 'gender'],
            enforce_detection=True
        )

        # Handle results (first detected face)
        result = results[0] if isinstance(results, list) else results
        
        print(f"Debug - Analysis Result Keys: {result.keys()}") # Log keys for debugging

        predicted_age = result.get('age')
        predicted_gender = result.get('dominant_gender')
        
        # Safer extraction of confidence
        gender_dict = result.get('gender', {})
        gender_confidence = gender_dict.get(predicted_gender, 0)

        return JSONResponse(content={
            "filename": file.filename,
            "prediction": {
                "age": predicted_age,
                "gender": predicted_gender,
                "gender_confidence": round(gender_confidence, 2) if gender_confidence else 0
            },
            "status": "success",
            "message": "Model updated: Now including gender prediction!"
        })

    except ValueError as ve:
        print(f"ValueError: {str(ve)}")
        raise HTTPException(status_code=400, detail=f"Face detection failed: {str(ve)}")
    except Exception as e:
        import traceback
        print(f"Exception during prediction: {str(e)}")
        print(traceback.format_exc()) # Print full traceback to docker logs
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

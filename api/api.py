from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from fastapi.middleware.cors import CORSMiddleware
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

modeldigits = load_model('../models/mnist_model_digits.h5')
modelletters = load_model('../models/mnist_model_letters.h5')


def preprocess_drawing_image_from_memory(image_data):
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (28, 28))

    if img.shape[-1] == 4:
        img[img[:, :, 3] == 0] = [255, 255, 255, 255]

    img = img[:, :, :3]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    img_gray = img_gray / 255.0
    img_gray = 1 - img_gray

    return img_gray


async def predict_image(image_data, model):
    preprocessed_image = preprocess_drawing_image_from_memory(image_data)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    preprocessed_image = np.expand_dims(
        preprocessed_image, axis=-1)
    prediction = model.predict(preprocessed_image)
    return np.argmax(prediction), prediction[0]


@app.post("/predict-digits/")
async def predict_digits(file: UploadFile = File(...)):
    if not file.filename.endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file format.")

    image_data = await file.read()
    predicted_label, probabilities = await predict_image(image_data, modeldigits)
    return JSONResponse(content={"digit": int(predicted_label), "probabilities": probabilities.tolist()})


@app.post("/predict-letters/")
async def predict_letters(file: UploadFile = File(...)):
    if not file.filename.endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file format.")

    image_data = await file.read()
    predicted_label, probabilities = await predict_image(image_data, modelletters)
    predicted_letter = chr(65 + predicted_label)
    return JSONResponse(content={"letter": predicted_letter, "probabilities": probabilities.tolist()})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)

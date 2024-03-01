from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import io
import cv2

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
modelfashion = load_model('../models/mnist_model_fashion.h5')

async def predict_model(file: UploadFile, model, response_type: str):
    if not file.filename.endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file format.")

    image_data = await file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (28, 28))

    if img.shape[2] == 4:  
        transparent_mask = img[:, :, 3] == 0
        img[transparent_mask] = [255, 255, 255, 255]
        img = img[:, :, :3]

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    img_gray = img_gray / 255.0
    img_gray = 1 - img_gray  
    img_array = img_gray.reshape(1, 28, 28, 1)  

    prediction = model.predict(img_array)
    if response_type == 'digit':
        result = np.argmax(prediction)
    elif response_type == 'letter':
        result = chr(65 + np.argmax(prediction))
    else:  # fashion
        class_name = ["T-shirt/top", "Trouser", "Pullover", "Dress",
                      "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        result = class_name[np.argmax(prediction)]

    return result

@app.post("/predict-digits/")
async def predict_digits(file: UploadFile = File(...)):
    digit = await predict_model(file, modeldigits, 'digit')
    return JSONResponse(content={"digit": int(digit)})

@app.post("/predict-letters/")
async def predict_letters(file: UploadFile = File(...)):
    letter = await predict_model(file, modelletters, 'letter')
    return JSONResponse(content={"letter": letter})

@app.post("/predict-fashion/")
async def predict_fashion(file: UploadFile = File(...)):
    fashion_item = await predict_model(file, modelfashion, 'fashion')
    return JSONResponse(content={"fashion": fashion_item})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# Load models
modeldigits = load_model('../models/mnist_model_digits.h5')
modelletters = load_model('../models/mnist_model_letters.h5')
modelfashion = load_model('../models/mnist_model_fashion.h5')

# Function to predict
async def predict_model(file: UploadFile, model, response_type: str):
    if not file.filename.endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file format.")
    
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    prediction = model.predict(image_array)
    if response_type == 'digit':
        result = np.argmax(prediction)
    elif response_type == 'letter':
        result = chr(65 + np.argmax(prediction))
    else:  # fashion
        class_name = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
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

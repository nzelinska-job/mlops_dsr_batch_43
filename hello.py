from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO


app = FastAPI()

@app.get("/")
def read_root():
    return "Hello"

async def predict_image(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(BytesIO(image_data))

    input_tensor = preprocess_image(image)
    prediction = predict(input_tensor)
    prediction_label = label_mapping[str(prediction)]
    
    return JSONResponse(content={"prediction": prediction_label})

from fastapi import FastAPI, File, UploadFile, HTTPException  # 1
from fastapi.middleware.cors import CORSMiddleware  # 2
import uvicorn  # 3
import numpy as np  # 4
from io import BytesIO  # 5
from PIL import Image  # 6
import tensorflow as tf  # 7

app = FastAPI()  # 8
print("1")

# List of origins allowed to make requests to this API
origins = [  # 9
    "http://localhost",  # 10
    "http://localhost:3000",  # Add your frontend URL here  # 11
]  # 12
print("2")

app.add_middleware(  # 13
    CORSMiddleware,  # 14
    allow_origins=origins,  # 15
    allow_credentials=True,  # 16
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)  # 17
    allow_headers=["*"],  # Allow all headers  # 18
)  # 19
print("3")

# Load the model
MODEL = tf.keras.models.load_model(r"D:\Derin_Ogrenme\doma\saved_models\models\modelk.keras")  # 20
print("4")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]  # 21
print("5")

@app.get("/")  # 22
async def ping():  # 23
    print("6")
    return "Hello, I am alive"  # 24

def read_file_as_image(data) -> np.ndarray:  # 25
    print("7")
    image = Image.open(BytesIO(data))  # 26
    image = image.resize((224, 224))  # Resize the image to the expected input shape  # 27
    image = np.array(image)  # 28
    print("8")
    return image  # 29

@app.post("/predict")  # 30
async def predict(file: UploadFile = File(...)):  # 31
    print("9")
    image = read_file_as_image(await file.read())  # 32
    img_batch = np.expand_dims(image, 0)  # 33
    print("10")
    
    # Perform prediction
    try:  # 34
        predictions = MODEL(img_batch)  # 35  # Update to use the call method
        print("11")
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]  # 36
        confidence = float(np.max(predictions[0]))  # 37
        print("12")
        return {  # 38
            'class': predicted_class,  # 39
            'confidence': confidence  # 40
        }  # 41
    except Exception as e:  # 42
        print(f"An error occurred: {e}")  # 43
        raise HTTPException(status_code=500, detail="Internal Server Error")  # 44

if __name__ == "__main__":  # 45
    uvicorn.run(app, host="localhost", port=8001)  # 46
    print("13")

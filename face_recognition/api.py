from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from recognizer import FaceRecognizer

app = FastAPI()

@app.post("/compare-face")
async def compare_face(file: UploadFile):
    recognizer = FaceRecognizer()
    recognizer.load_and_compute_known_faces()
    await recognizer.compare_with_known_faces_from_upload(file)
    return recognizer.build_matches_response(1)

@app.post("/clean-training-data")
async def clean_training_data():
    recognizer = FaceRecognizer()
    recognizer.clean_training_data()
    return {"message": "Training data cleaned"}


# Test endpoint
@app.get("/hello")
async def hello_world():
    return {"message": "Hello World!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
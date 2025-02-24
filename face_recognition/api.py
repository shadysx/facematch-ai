from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from recognizer import FaceRecognizer

app = FastAPI()

@app.post("/get-matches-names")
async def compare_face(file: UploadFile):
    recognizer = FaceRecognizer()
    recognizer.load_and_compute_known_faces()
    await recognizer.compare_with_known_faces_from_upload(file)
    return recognizer.build_matches_with_images_response(1)

@app.post("/get-matches-with-images")
async def get_matches_with_images(file: UploadFile):
    recognizer = FaceRecognizer()
    recognizer.load_and_compute_known_faces()
    await recognizer.compare_with_known_faces_from_upload(file)
    return recognizer.build_matches_with_images_response(1)

@app.post("/clean-training-data")
async def clean_training_data():
    recognizer = FaceRecognizer()
    recognizer.clean_training_data()
    return {"message": "Training data cleaned"}

@app.get("/is-cuda-enabled")
async def is_cuda_enabled():
    recognizer = FaceRecognizer()
    return {"cuda_enabled": recognizer.is_cuda_enabled()}

# Test endpoint
@app.get("/hello")
async def hello_world():
    return {"message": "Hello World!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
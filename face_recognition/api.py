from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from recognizer import FaceRecognizer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=False,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

@app.post("/get-matches-names")
async def compare_face(file: UploadFile):
    recognizer = FaceRecognizer()
    if recognizer.is_training:
        return {"message": "Training in progress, please wait..."}
    recognizer.load_and_compute_known_faces()
    await recognizer.compare_with_known_faces_from_upload(file)
    return recognizer.get_n_closest_names_by_distance(5)

@app.post("/get-matches-names-with-images")
async def get_matches_names_with_images(file: UploadFile):
    recognizer = FaceRecognizer()
    if recognizer.is_training:
        return {"message": "Training in progress, please wait..."}
    recognizer.load_and_compute_known_faces()
    await recognizer.compare_with_known_faces_from_upload(file)
    return recognizer.build_n_matches_with_images_response(5)

@app.post("/clean-training-data")
async def clean_training_data():
    recognizer = FaceRecognizer()
    if recognizer.is_training:
        return {"message": "Training in progress, please wait..."}
    recognizer.clean_training_data()
    return {"message": "Training data cleaned"}

@app.get("/is-cuda-enabled")
async def is_cuda_enabled():
    recognizer = FaceRecognizer()
    return {"cuda_enabled": recognizer.is_cuda_enabled()}

@app.get("/is-training")
async def is_training():
    recognizer = FaceRecognizer()
    return {"is_training": recognizer.is_training()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
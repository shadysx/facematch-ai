# Facematch AI Engine
This AI engine is able to find the closest match for an unknown face in a database of known faces.

This is a simple API for face recognition. It uses dlib for face recognition and FastAPI for the API.

## Prerequisites
- GCC 8 (required)
- NVIDIA drivers installed for GPU version ([Installation guide](https://ubuntu.com/server/docs/nvidia-drivers-installation))

## Installation

### CPU Version (without CUDA)
```bash
conda create -n facematch_cpu python=3.8 cmake ipython
conda activate facematch_cpu
conda install -c conda-forge gcc=8.5.0 libxcrypt
conda install dlib fastapi uvicorn
conda install -c menpo opencv
```

### GPU Version (with CUDA)
> **Note**: This version requires manual compilation of dlib with CUDA support.
> - [Compilation tutorial](https://gist.github.com/nguyenhoan1988/ed92d58054b985a1b45a521fcf8fa781)
> - [Dlib source code](https://github.com/davisking/dlib)

```bash
conda create -n facematch_gpu python=3.8 cmake ipython
conda activate facematch_gpu
conda install -c conda-forge gcc=8.5.0 libxcrypt
conda install -c conda-forge cudatoolkit=11.8.0
conda install -c conda-forge cudnn=8.9.2.26
```

## Project Setup

### Dataset Preparation
1. Create a `downloaded_images` folder at the root of the project
2. Add JPG images of people you want to recognize in this folder

### Model Training
1. Place the unknown face image as `unknown_face.jpg` in `/facematch-ai-engine/faces/unknown/`
2. Run the recognizer script:
```bash
cd facematch-ai-engine/face_recognition
python recognizer.py
```
> Note: First run will create `known_faces_descriptors.pkl` at the root folder which will be used for future runs. Delete this file to force a new training.

## API Usage
### You can also run the engine directly from the api.py file, this will expose the engine functions as endpoints.

### Starting the API Server
```bash
cd facematch-ai-engine/face_recognition
python api.py
```

### Available Endpoints
```bash
# Get matches by names
curl -X POST -F "file=@unknown_face.jpg" http://localhost:8000/get-matches-names

# Get matches with images
curl -X POST -F "file=@unknown_face.jpg" http://localhost:8000/get-matches-with-images

# Check if CUDA is enabled
curl -X GET http://localhost:8000/is-cuda-enabled
```


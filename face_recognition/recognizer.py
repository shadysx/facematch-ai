import sys
import os
import dlib
import glob
import pickle
import numpy as np
from fastapi import UploadFile
import cv2

class FaceRecognizer:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor("../models/shape_predictor_5_face_landmarks.dat")
        self.facerec = dlib.face_recognition_model_v1("../models/dlib_face_recognition_resnet_model_v1.dat")
        self.known_faces = {} 
        self.matches_found = []
        self.all_distances = {}  
        self.known_faces_folder = "../downloaded_images"
        self.unknown_face_path = "../faces/unknown/unknown_face.jpg"
        self.descriptors_file = "../known_faces_descriptors.pkl"
        self.threshold = 0.5

    def save_known_faces(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.known_faces, f)
        
    def load_known_faces(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                self.known_faces = pickle.load(f)
                print(f"Loaded {len(self.known_faces)} known faces")
            return True
        return False

    def compute_face_descriptor(self, img_path, label=None):
        img = dlib.load_rgb_image(img_path)
        dets = self.detector(img, 1)
        face_descriptors = [] 
        for k, d in enumerate(dets):
            shape = self.sp(img, d)
            face_descriptor = np.array(self.facerec.compute_face_descriptor(img, shape))
            if label: 
                self.known_faces[f"{label}_{k}"] = face_descriptor
                print(f"Face {k} registered for: {label}")
            face_descriptors.append(face_descriptor)
        return face_descriptors

    def compare_with_known_faces(self, img_path):
        face_descriptor = self.compute_face_descriptor(img_path)
        if face_descriptor is None:
            print("No face detected in the image")
            return
        img_name = os.path.basename(img_path)
        self.all_distances[img_name] = {}
        for name, known_descriptor in self.known_faces.items():
            distance = np.linalg.norm(face_descriptor - known_descriptor)
            self.all_distances[img_name][name] = distance
            if distance < self.threshold:
                self.matches_found.append(name)

    async def compare_with_known_faces_from_upload(self, upload_file: UploadFile):
        # Read the content of the uploaded file
        contents = await upload_file.read()
        # Convert to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Convert BGR (OpenCV) to RGB (dlib)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Detect and calculate the face descriptor
        dets = self.detector(img_rgb, 1)

        if len(dets) == 0:
            print("No face detected in the image")
            return
        # Use the first detected face
        shape = self.sp(img_rgb, dets[0])
        face_descriptor = np.array(self.facerec.compute_face_descriptor(img_rgb, shape))
        unknown_img_name = upload_file.filename
        self.all_distances[unknown_img_name] = {}
        n_known_faces = len(self.known_faces)
        print(f"Comparing with {n_known_faces} known faces")
        for file_path, known_descriptor in self.known_faces.items():
            distance = np.linalg.norm(face_descriptor - known_descriptor)
            self.all_distances[unknown_img_name][file_path] = distance
            if distance < self.threshold:
                self.matches_found.append(file_path)

    def load_and_compute_known_faces(self):
        """Load known faces from file or compute and save them if file doesn't exist."""
        if not self.load_known_faces(self.descriptors_file):
            for root, dirs, files in os.walk(self.known_faces_folder):
                for f in files:
                    if f.endswith('.jpg'):
                        full_path = os.path.join(root, f)
                        relative_path = os.path.relpath(full_path, self.known_faces_folder)
                        name = os.path.splitext(relative_path)[0]
                        self.compute_face_descriptor(full_path, label=name)
            self.save_known_faces(self.descriptors_file)
    
    def get_n_closest_matches(self, n=10):
        matches = []
        for _, distances in self.all_distances.items():
            sorted_distances = sorted(distances.items(), key=lambda x: x[1])
            for file_path, distance in sorted_distances[:n]:
                matches.append({
                    "name": file_path.split('/')[0],
                    "distance": distance,
                    "pictures": []
                })
        return {"matches": matches}
    

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        if os.path.exists("../known_faces_descriptors.pkl"):
            os.remove("../known_faces_descriptors.pkl")
            print("Starting clean")

    if dlib.DLIB_USE_CUDA:
        print("CUDA is enabled")
    else:
        print("CUDA is disabled")

    recognizer = FaceRecognizer()
    recognizer.load_and_compute_known_faces()
    recognizer.compare_with_known_faces(recognizer.unknown_face_path)
    print(recognizer.get_n_closest_matches())
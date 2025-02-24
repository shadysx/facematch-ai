import sys
import os
import dlib
import glob
import pickle
import numpy as np
from fastapi import UploadFile
import cv2
import base64

class FaceRecognizer:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor("../models/shape_predictor_5_face_landmarks.dat")
        self.facerec = dlib.face_recognition_model_v1("../models/dlib_face_recognition_resnet_model_v1.dat")
        self.known_faces = {} 
        # Here are the matches found (Not returned yet)
        self.matches_found = []
        self.threshold = 0.5
        # Here are the distances between the unknown face and the known faces
        self.all_distances = {} # Key is the unknown face, value is a dictionary of known faces and their distances
        self.known_faces_folder = "../downloaded_images"
        # When called from the script
        self.unknown_face_path = "../faces/unknown/unknown_face.jpg"
        # When called from the API
        self.unknown_uploaded_image = ""
        self.descriptors_file = "../known_faces_descriptors.pkl"

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
        try:
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
        except Exception as e:
            print(f"Error computing face descriptor: {e}")
            return None

    def compare_with_known_faces(self, img_path):
        try:
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
        except Exception as e:
            print(f"Error comparing with known faces: {e}")
            return

    async def compare_with_known_faces_from_upload(self, upload_file: UploadFile):
        try:
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
            unknown_uploaded_image = upload_file.filename
            self.all_distances[unknown_uploaded_image] = {}
            n_known_faces = len(self.known_faces)

            print(f"Comparing with {n_known_faces} known faces")
            for file_path, known_descriptor in self.known_faces.items():
                distance = np.linalg.norm(face_descriptor - known_descriptor)
                self.all_distances[unknown_uploaded_image][file_path] = distance
                if distance < self.threshold:
                    self.matches_found.append(file_path)
        except Exception as e:
            print(f"Error comparing with known faces: {e}")
            return

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
    
    def get_n_closest_names_by_distance(self, n=10):
        # First names are the closest matches
        names_by_distance = []
        for _, distances in self.all_distances.items():
            sorted_distances = sorted(distances.items(), key=lambda x: x[1])
            for file_path, distance in sorted_distances[:n]:
                names_by_distance.append(file_path.split('/')[0])
        return names_by_distance
    
    def build_matches_with_images_response(self, n=10):
        names_by_distance = self.get_n_closest_names_by_distance(n)
        matches = []
        for match_name in names_by_distance:

            folder = os.path.join(self.known_faces_folder, match_name)
            jpg_files = glob.glob(os.path.join(folder, "*.jpg"))

            images_from_match = []

            for jpg_file in jpg_files:
                # Read the image and encode it in base64
                image_data = base64.b64encode(open(jpg_file, "rb").read()).decode('utf-8')
                images_from_match.append(image_data)

            matches.append({
                "name": match_name,
                "images": images_from_match
            })
        return matches


    def clean_training_data(self):
        if os.path.exists("../known_faces_descriptors.pkl"):
            os.remove("../known_faces_descriptors.pkl")
    
    def is_cuda_enabled(self):
        return dlib.DLIB_USE_CUDA


if __name__ == "__main__":
    recognizer = FaceRecognizer()
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        recognizer.clean_training_data()
        exit()

    if recognizer.is_cuda_enabled():
        print("CUDA is enabled")
    else:
        print("CUDA is disabled")

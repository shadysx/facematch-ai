import sys
import os
import dlib
import glob
import numpy as np  

class FaceRecognizer:
    def __init__(self, predictor_path, face_rec_model_path):
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor(predictor_path)
        self.facerec = dlib.face_recognition_model_v1(face_rec_model_path)
        # self.win = dlib.image_window()
        self.known_faces = {} 
        self.matches_found = []

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

    def compare_with_known_faces(self, img_path, threshold=0.6):
        face_descriptor = self.compute_face_descriptor(img_path)
        
        if face_descriptor is None:
            print("No face detected in the image")
            return
        
        for name, known_descriptor in self.known_faces.items():
            distance = np.linalg.norm(face_descriptor - known_descriptor)
            print(f"Distance avec {name}: {distance}")
            if distance < threshold:
                print(f"MATCH FOUND! It's probably {name} (distance: {distance:.2f})")
                self.matches_found.append(name)
            else:
                print(f"It's not {name} (distance: {distance:.2f})")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage:\n"
            "   ./face_recognition.py shape_predictor_5_face_landmarks.dat "
            "dlib_face_recognition_resnet_model_v1.dat known_faces_folder unknown_faces_folder")
        exit()


    if dlib.DLIB_USE_CUDA:
        print("CUDA is enabled")
    else:
        print("CUDA is disabled")

    recognizer = FaceRecognizer(sys.argv[1], sys.argv[2])
    known_faces_folder = sys.argv[3]
    unknown_faces_folder = sys.argv[4]

    # First step: register known faces
    # Assume the images are named with the person's name
    for f in glob.glob(os.path.join(known_faces_folder, "*.jpg")):
        name = os.path.basename(f).split('.')[0]  # Nom du fichier sans extension
        recognizer.compute_face_descriptor(f, label=name)

    # Second step: test with unknown faces
    print("\nTesting unknown faces:")
    for f in glob.glob(os.path.join(unknown_faces_folder, "*.jpg")):
        print(f"\nAnalyzing {f}:")
        recognizer.compare_with_known_faces(f)

    # Display matches found
    if recognizer.matches_found:
        print("\nMatches found:")
        for match in recognizer.matches_found:
            print(f"- {match}")

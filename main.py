import sys
import dlib

test = sys.argv[1];

cnn_face_detector = dlib.cnn_face_detection_model_v1(sys.argv[1])
win = dlib.image_window()
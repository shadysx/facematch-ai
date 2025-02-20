import sys
from face_detector import FaceDetector

if len(sys.argv) < 3:
    print(
        "Call this program like this:\n"
        "   ./cnn_face_detector.py mmod_human_face_detector.dat ../examples/faces/*.jpg\n"
        "You can get the mmod_human_face_detector.dat file from:\n"
        "    https://dlib.net/files/mmod_human_face_detector.dat.bz2")
    exit()

detector = FaceDetector(sys.argv[1])

for f in sys.argv[2:]:
    print(f"Processing file: {f}")
    detections, img = detector.detect_faces(f)
    detector.display_results(detections, img)
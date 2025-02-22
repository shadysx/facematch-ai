import dlib
import sys

class FaceDetector:
    def __init__(self, model_path):
        self.detector = dlib.cnn_face_detection_model_v1(model_path)
        self.win = dlib.image_window()

    def detect_faces(self, image_path, upsample_num=1):
        """
        Detect faces in the given image.
        
        Args:
            image_path: Path to the image file
            upsample_num: Number of times to upsample the image (default=1)
            
        Returns:
            tuple: (detections, image)
        """
        img = dlib.load_rgb_image(image_path)
        dets = self.detector(img, upsample_num)
        return dets, img

    def display_results(self, detections, image):
        """Display detection results in a window"""
        print(f"Number of faces detected: {len(detections)}")
        
        for i, d in enumerate(detections):
            print(f"Detection {i}: Left: {d.rect.left()} Top: {d.rect.top()} "
                  f"Right: {d.rect.right()} Bottom: {d.rect.bottom()} "
                  f"Confidence: {d.confidence}")

        rects = dlib.rectangles()
        rects.extend([d.rect for d in detections])

        self.win.clear_overlay()
        self.win.set_image(image)
        self.win.add_overlay(rects)
        dlib.hit_enter_to_continue() 

if __name__ == "__main__":
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

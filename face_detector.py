import dlib

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
import cv2
import os
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Face detector (OpenCV Haar — your working version)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# License-plate detector (pretrained YOLOv8 from HuggingFace, downloads once)
_plate_path = hf_hub_download("Koushim/yolov8-license-plate-detection", "best.pt")
plate_model = YOLO(_plate_path)

def _blur_region(img, x, y, w, h):
    x, y = max(0, x), max(0, y)
    roi = img[y:y+h, x:x+w]
    if roi.size == 0:
        return
    k = max(31, (max(w, h) // 2) | 1)
    img[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (k, k), 0)

def redact(image_path, blur_faces=True, blur_plates=True):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Couldn't read '{image_path}'")
    count = 0

    if blur_faces:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for (x, y, w, h) in face_cascade.detectMultiScale(gray, 1.1, 8, minSize=(40, 40)):
            _blur_region(img, x, y, w, h); count += 1

    if blur_plates:
        results = plate_model(img, conf=0.25)[0]
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            _blur_region(img, x1, y1, x2 - x1, y2 - y1); count += 1

    return img, count

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    result, n = redact("image/test.jpg")
    cv2.imwrite("outputs/redacted.jpg", result)
    print(f"Blurred {n} region(s). Saved outputs/redacted.jpg")
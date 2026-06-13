from ultralytics import YOLOWorld
import numpy as np
import cv2
import os

model = YOLOWorld("yolov8s-world.pt")   # downloads automatically on first run
model.to("cuda")  

def detect_word(image_path, word, conf=0.1):
    model.set_classes([word])                     # tell it what to look for
    results = model(image_path, conf=conf)[0]
    h, w = results.orig_shape
    detections = []
    for i, box in enumerate(results.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        mask = np.zeros((h, w), dtype=bool)        # rectangular mask from the box
        mask[y1:y2, x1:x2] = True
        detections.append({
            "id": i,
            "label": word,
            "box": np.array([x1, y1, x2, y2]),
            "mask": mask,
        })
    return detections

# --- standalone test: draw boxes so you can see what it found ---
if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    image_path = "image/test.jpg"
    word = "bucket"                                # try the bucket the boy is holding!
    dets = detect_word(image_path, word)
    print(f"Found {len(dets)} '{word}'")

    img = cv2.imread(image_path)
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, str(d["id"]), (x1, max(y1 - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite("outputs/detected_word.jpg", img)
    print("Saved outputs/detected_word.jpg")
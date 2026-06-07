from ultralytics import YOLO
import numpy as np
import cv2

model = YOLO("yolo11n-seg.pt")   # downloads automatically on first run

def build_mask(detections, remove_ids, shape):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for d in detections:
        if d["id"] in remove_ids:
            mask[d["mask"]] = 255          # white = erase here
    return cv2.dilate(mask, np.ones((15, 15), np.uint8))   # pad edges a little

def detect(image_path, category="person"):
    results = model(image_path)[0]
    detections = []
    if results.masks is None:
        return detections
    h, w = results.orig_shape
    for i, (m, c, box) in enumerate(zip(results.masks.data,
                                        results.boxes.cls,
                                        results.boxes.xyxy)):
        label = model.names[int(c)]
        if label == category:
            mask_full = cv2.resize(m.cpu().numpy(), (w, h))   # back to original size
            detections.append({
                "id": i,
                "label": label,
                "box": box.cpu().numpy().astype(int),
                "mask": mask_full > 0.5,
            })
    return detections

if __name__ == "__main__":
    image_path = "image/test.jpg"
    dets = detect(image_path, "person")
    print("Detected ids:", [d["id"] for d in dets])

    keep = [0, 1]                                              # the two main kids
    remove_ids = [d["id"] for d in dets if d["id"] not in keep]
    print("Removing ids:", remove_ids)

    img = cv2.imread(image_path)
    mask = build_mask(dets, remove_ids, img.shape)
    cv2.imwrite("result/mask.png", mask)
    print("Saved mask.png — white areas are what gets erased")
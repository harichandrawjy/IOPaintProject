import cv2
import numpy as np
import os
from detect_category import detect

def portrait_blur(image_path, subject="person", blur_strength=51):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Couldn't read '{image_path}'")
    h, w = img.shape[:2]

    dets = detect(image_path, subject)

    # combine all subject masks into one
    mask = np.zeros((h, w), dtype=np.float32)
    for d in dets:
        mask[d["mask"]] = 1.0

    # feather the edges so the transition looks natural
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    mask = np.dstack([mask] * 3)

    # blur the whole image, then composite sharp subject over it
    k = blur_strength | 1                      # force odd kernel
    blurred = cv2.GaussianBlur(img, (k, k), 0)
    result = (img * mask + blurred * (1 - mask)).astype(np.uint8)
    return result, len(dets)

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    result, n = portrait_blur("image/test.jpg", "person")
    cv2.imwrite("outputs/portrait.jpg", result)
    print(f"Found {n} subject(s). Saved outputs/portrait.jpg")
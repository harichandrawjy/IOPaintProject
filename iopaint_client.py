import requests, base64, cv2, numpy as np

URL = "http://localhost:8080/api/v1/inpaint"

def _b64(img):
    return base64.b64encode(cv2.imencode(".png", img)[1]).decode()

def erase(image_path, mask_path, out_path="result.jpg"):
    image = cv2.imread(image_path)
    mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"image not found or unreadable: {image_path}")
    if mask is None:
        raise FileNotFoundError(f"mask not found or unreadable: {mask_path}")
    payload = {
        "image": _b64(image),
        "mask":  _b64(mask),
        "ldm_steps": 20,
        "ldm_sampler": "plms",
        "hd_strategy": "Original",
        "hd_strategy_crop_trigger_size": 800,
        "hd_strategy_crop_margin": 128,
        "hd_strategy_resize_limit": 2048,
    }
    resp = requests.post(URL, json=payload)
    resp.raise_for_status()
    result = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite(out_path, result)
    print("Saved", out_path)
    return result

if __name__ == "__main__":
    erase("image/test.jpg", "mask/mask.png")
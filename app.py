import streamlit as st
import cv2
from detect_category import detect, build_mask
from iopaint_client import erase

st.set_page_config(page_title="AI Photo Cleanup", page_icon="🧹", layout="centered")

st.markdown("""
<style>
.stButton>button { border-radius: 10px; font-weight: 600; padding: 0.5rem 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.title("🧹 AI Photo Cleanup")
st.caption("Category removal — detect people, choose who to remove, erase them.")

if "detections" not in st.session_state:
    st.session_state.detections = None

uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

category = st.selectbox("Category to detect", COCO_CLASSES)

if uploaded is not None:
    image_path = "upload.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded.getbuffer())

    if st.button("1. Detect", type="primary"):
        st.session_state.detections = detect(image_path, category)

    if st.session_state.detections is not None:
        dets = st.session_state.detections
        img = cv2.imread(image_path)

        boxed = img.copy()
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(boxed, str(d["id"]), (x1, max(y1 - 8, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        st.image(cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB),
                 caption=f"Found {len(dets)} '{category}' — note the ID numbers")

        ids = [d["id"] for d in dets]
        remove_ids = st.multiselect("2. Select IDs to REMOVE", options=ids)

        if st.button("3. Remove selected", type="primary"):
            if not remove_ids:
                st.warning("Pick at least one ID to remove.")
            else:
                mask = build_mask(dets, remove_ids, img.shape)
                cv2.imwrite("mask.png", mask)
                try:
                    with st.spinner("Erasing… this can take a few seconds"):
                        result = erase(image_path, "mask.png")
                    st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Result")
                    st.success(f"Removed IDs {remove_ids}")
                    _, buf = cv2.imencode(".jpg", result)
                    st.download_button("Download result", buf.tobytes(),
                                       "result.jpg", "image/jpeg")
                except Exception as e:
                    st.error(f"Erase failed — is IOPaint running on port 8080? ({e})")
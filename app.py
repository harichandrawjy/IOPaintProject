import streamlit as st
import cv2
import os
from detect_category import detect, build_mask, model
from iopaint_client import erase
from privacy import redact

os.makedirs("outputs", exist_ok=True)
COCO_CLASSES = sorted(model.names.values())   # the 80 categories, straight from the model

st.set_page_config(page_title="AI Photo Cleanup", page_icon="🧹", layout="centered")
st.markdown("<style>.stButton>button{border-radius:10px;font-weight:600;padding:0.5rem 1.5rem;}</style>",
            unsafe_allow_html=True)
st.title("🧹 AI Photo Cleanup")

tab_cat, tab_priv = st.tabs(["Category Removal", "Privacy Blur"])

# ---------------- Category Removal ----------------
with tab_cat:
    st.caption("Detect a category, choose which to remove, erase them.")
    if "detections" not in st.session_state:
        st.session_state.detections = None

    uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"], key="cat_up")
    category = st.selectbox("Category to detect", COCO_CLASSES)

    if uploaded is not None:
        image_path = "outputs/upload.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded.getbuffer())

        if st.button("1. Detect", type="primary", key="cat_detect"):
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

            if st.button("3. Remove selected", type="primary", key="cat_remove"):
                if not remove_ids:
                    st.warning("Pick at least one ID to remove.")
                else:
                    mask = build_mask(dets, remove_ids, img.shape)
                    cv2.imwrite("outputs/mask.png", mask)
                    try:
                        with st.spinner("Erasing… this can take a few seconds"):
                            result = erase(image_path, "outputs/mask.png")
                        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Result")
                        st.success(f"Removed IDs {remove_ids}")
                        _, buf = cv2.imencode(".jpg", result)
                        st.download_button("Download result", buf.tobytes(),
                                           "result.jpg", "image/jpeg", key="cat_dl")
                    except Exception as e:
                        st.error(f"Erase failed — is IOPaint running on port 8080? ({e})")

# ---------------- Privacy Blur ----------------
with tab_priv:
    st.caption("Automatically blur faces and license plates. (No IOPaint needed.)")
    p_uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"], key="priv_up")
    c1, c2 = st.columns(2)
    blur_faces = c1.checkbox("Blur faces", value=True)
    blur_plates = c2.checkbox("Blur license plates", value=True)

    if p_uploaded is not None:
        p_path = "outputs/priv_upload.jpg"
        with open(p_path, "wb") as f:
            f.write(p_uploaded.getbuffer())
        if st.button("Blur", type="primary", key="priv_blur"):
            with st.spinner("Detecting and blurring…"):
                result, n = redact(p_path, blur_faces, blur_plates)
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption=f"Blurred {n} region(s)")
            _, buf = cv2.imencode(".jpg", result)
            st.download_button("Download result", buf.tobytes(),
                               "redacted.jpg", "image/jpeg", key="priv_dl")
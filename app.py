import streamlit as st
from PIL import Image
import numpy as np
import os

st.set_page_config(
    page_title="Railway Wildlife Detection",
    layout="centered"
)

st.title("Railway Wildlife Detection")

# ── Safety checks ────────────────────────────────────────
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    st.error("Model file **best.pt** not found in project directory.")
    st.stop()

# ── Lazy model loading ───────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLO model…")
def load_model():
    from ultralytics import YOLO
    return YOLO(MODEL_PATH)

# ── Image input ──────────────────────────────────────────
mode = st.radio(
    "Image source",
    ["Camera", "Upload"],
    horizontal=True
)

image = None

if mode == "Camera":
    cam = st.camera_input("Capture image")
    if cam is not None:
        image = Image.open(cam).convert("RGB")
else:
    up = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "webp"]   # added webp just in case
    )
    if up is not None:
        image = Image.open(up).convert("RGB")

if image is None:
    st.info("Please upload or capture an image to continue.")
    st.stop()

# Show input image immediately (good UX)
st.image(image, caption="Input image", use_column_width=True)

# ── Parameters ───────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    conf = st.slider(
        "Confidence threshold",
        0.05, 0.95, 0.25, 0.05,
        help="Lower = more detections (but more false positives)"
    )
with col2:
    iou = st.slider(
        "IOU threshold (NMS)",
        0.1, 0.9, 0.45, 0.05,
        help="Higher = fewer overlapping boxes"
    )

# ── Detection ────────────────────────────────────────────
if st.button("Run detection", type="primary", use_container_width=True):

    model = load_model()

    with st.spinner("Running object detection…"):
        results = model.predict(
            source=np.array(image),
            conf=conf,
            iou=iou,
            imgsz=640,
            device="cpu",           # change to "0" or "" if you have GPU
            verbose=False
        )

    # Usually only one image → take first result
    res = results[0]

    if len(res.boxes) == 0:
        st.warning("**No wildlife / objects detected** with current settings.")
        st.info("Try lowering the confidence threshold or uploading a clearer image.")
    else:
        # ── Plot only when there ARE boxes ──
        annotated_array = res.plot()                       # returns np.uint8 array (BGR)
        annotated_img = Image.fromarray(annotated_array)   # safe now

        st.success(f"**{len(res.boxes)}** detection{'s' if len(res.boxes) > 1 else ''}")
        st.image(annotated_img, caption="Detection result", use_column_width=True)

        # ── Show details ──
        st.subheader("Detected objects")

        for box in res.boxes:
            cls_id = int(box.cls.item())          # .item() is cleaner
            label = model.names[cls_id]
            score = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist() # or .xywh / .xywhn etc.

            st.write(f"• **{label}**  –  confidence {score:.2%}")
            # Optional: show bounding box coordinates
            # st.caption(f"bbox: ({int(x1)},{int(y1)}) – ({int(x2)},{int(y2)})")

st.markdown("---")
st.caption("App using Ultralytics YOLO • Model: best.pt")

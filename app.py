import streamlit as st
from PIL import Image
import numpy as np
import os

st.set_page_config(
    page_title="Railway Wildlife Detection",
    layout="centered"
)

st.title("Railway Wildlife Detection")

# ---------- Safety checks ----------
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    st.error("Model file best.pt not found in project directory")
    st.stop()

# ---------- Lazy model loading ----------
@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model():
    from ultralytics import YOLO
    return YOLO(MODEL_PATH)

# ---------- Image input ----------
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
        type=["jpg", "jpeg", "png"]
    )
    if up is not None:
        image = Image.open(up).convert("RGB")

if image is None:
    st.info("Please upload or capture an image to continue")
    st.stop()

st.image(
    image,
    caption="Input image",
    width=500
)

# ---------- Parameters ----------
conf = st.slider(
    "Confidence threshold",
    0.05,
    0.95,
    0.25,
    0.05
)

iou = st.slider(
    "IOU threshold",
    0.1,
    0.9,
    0.7,
    0.05
)

# ---------- Detection ----------
if st.button("Run detection", type="primary"):

    model = load_model()

    with st.spinner("Running object detection..."):
        results = model.predict(
            source=np.array(image),
            conf=conf,
            iou=iou,
            imgsz=640,
            device="cpu",
            verbose=False
        )

    res = results[0]

    if res.boxes is None or len(res.boxes) == 0:
        st.warning("No detections found")
        st.stop()

    annotated_img = Image.fromarray(res.plot())

    st.image(
        annotated_img,
        caption="Detected image",
        width=500
    )

    st.subheader("Detections")

    for box in res.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        score = float(box.conf[0])
        st.write(f"{label} ({score:.2f})")

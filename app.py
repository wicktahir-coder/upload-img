import os
import io
import requests
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="Railway Wildlife Detection", layout="centered")

DEPLOY_URL = "https://predict-697f107a5bf07d5df6b6-dproatj77a-as.a.run.app/predict"
API_KEY = os.environ.get("ULTRALYTICS_DEPLOY_API_KEY")

if not API_KEY:
    st.error("Set ULTRALYTICS_DEPLOY_API_KEY in environment variables")
    st.stop()


def draw_boxes(image: Image.Image, predictions: list) -> Image.Image:
    draw = ImageDraw.Draw(image)
    width, height = image.size

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    for pred in predictions:
        box = pred.get("box", {})
        name = pred.get("name", "unknown")
        confidence = pred.get("confidence", 0)

        x1 = int(box.get("x1", 0) * width)
        y1 = int(box.get("y1", 0) * height)
        x2 = int(box.get("x2", 1) * width)
        y2 = int(box.get("y2", 1) * height)

        draw.rectangle([x1, y1, x2, y2], outline="lime", width=4)

        label = f"{name} {confidence:.2f}"
        bbox = draw.textbbox((x1, y1), label, font=font)
        lw = bbox[2] - bbox[0]
        lh = bbox[3] - bbox[1]

        draw.rectangle([x1, y1 - lh - 4, x1 + lw + 4, y1], fill="lime")
        draw.text((x1 + 2, y1 - lh - 2), label, fill="black", font=font)

    return image


st.title("Railway Wildlife Detection")
st.caption("Cloud inference via Ultralytics deployed model")

mode = st.radio("Source", ["Camera", "Upload"], horizontal=True)

image_pil = None
if mode == "Camera":
    cam = st.camera_input("Take photo")
    if cam:
        image_pil = Image.open(cam).convert("RGB")
else:
    up = st.file_uploader("Image", type=["jpg", "jpeg", "png"])
    if up:
        image_pil = Image.open(up).convert("RGB")

if image_pil is None:
    st.stop()

st.subheader("Input")
st.image(image_pil, use_column_width=True)

conf = st.slider("Confidence", 0.05, 0.95, 0.25, 0.05)
iou = st.slider("IOU", 0.10, 0.90, 0.70, 0.05)

if st.button("Run Cloud Detection", type="primary"):
    buf = io.BytesIO()
    image_pil.save(buf, format="JPEG", quality=90)
    buf.seek(0)

    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    files = {
        "file": ("image.jpg", buf, "image/jpeg")
    }

    data = {
        "conf": conf,
        "iou": iou,
        "imgsz": 640
    }

    with st.spinner("Calling Ultralytics deployment..."):
        try:
            response = requests.post(
                DEPLOY_URL,
                headers=headers,
                data=data,
                files=files,
                timeout=90
            )
            response.raise_for_status()
            result = response.json()

            predictions = result.get("predictions") or result.get("results") or []

            if not predictions:
                st.info("No detections found")
                st.json(result)
                st.stop()

            annotated = draw_boxes(image_pil.copy(), predictions)

            st.subheader("Results")
            st.image(annotated, use_column_width=True)

            st.subheader("Detections")
            for pred in predictions:
                name = pred.get("name", "unknown")
                score = pred.get("confidence", 0)
                st.write(f"{name} ({score:.2f})")

            with st.expander("Raw API Response"):
                st.json(result)

        except requests.exceptions.RequestException as e:
            st.error("API request failed")
            if hasattr(e, "response") and e.response is not None:
                st.code(e.response.text)
        except Exception as e:
            st.error(str(e))

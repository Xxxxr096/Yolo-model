import streamlit as st
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("Detection et reconnaisance d'objet avec YOLO")

model = YOLO("yolov8n.pt")
upload_files = st.file_uploader("Choisissez une image: ", type=["jpg", "jpeg", "png"])

if upload_files is not None:
    image = Image.open(upload_files).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption="Image original", use_container_width=True)

    # Inference
    results = model(image_np)
    pred = results[0]
    boxes = pred.boxes

    st.image(
        results[0].plot(), caption="Image avec Détection", use_container_width=True
    )

    st.subheader("Objets détectés : ")
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # coordonnées boite englobante

        confidence = box.conf[0].item()  # Score de confiance

        classes = int(box.cls[0].item())  # classe prédite

        classe_name = model.names[classes]
        st.markdown(
            f"""
                    - **Objet** : {classes}
                    - **Confiance** : {confidence:.2f}
                    - **Boite** : ({x1:.0f},{y1:.0f}) a ({x2:.0f},{y2:.0f})"""
        )

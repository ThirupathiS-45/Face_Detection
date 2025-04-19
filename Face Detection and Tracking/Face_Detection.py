import cv2
import streamlit as st
import numpy as np

# Load Haar cascade
alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)

st.set_page_config(page_title="Real-time Face Detection", layout="centered")
st.title("Real-time Face Detection")

# Center the Start Camera button using columns
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    start = st.button('📷 Start Camera')

FRAME_WINDOW = st.image([])

if start:
    cam = cv2.VideoCapture(0)
    stop_button = st.button('🛑 Stop Camera')

    while cam.isOpened():
        ret, img = cam.read()
        if not ret:
            st.write("Failed to grab frame")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray, 1.3, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # Convert BGR to RGB for Streamlit
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(img_rgb)

        if stop_button:
            break

    cam.release()

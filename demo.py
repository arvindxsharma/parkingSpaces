import cv2
import pickle
import cvzone
import numpy as np
import streamlit as st
import time

# Load parking positions
def load_parking_positions():
    try:
        with open("CarParkPos", 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Parking positions file not found.")
        return []

posList = load_parking_positions()
width, height = 107, 48

def check_parking_space(img, imgPro):
    spaceCounter = 0

    for pos in posList:
        x, y = pos

        imgCrop = imgPro[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)

        if count < 800:
            color = (0, 255, 0)
            thickness = 5
            spaceCounter += 1
        else:
            color = (0, 0, 255)
            thickness = 2

        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1, thickness=2, offset=0, colorR=color)

    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3, thickness=5, offset=20, colorR=(0, 200, 0))

def process_video_file(video_file, frameST):
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        st.error("Error opening video stream or file.")
        return

    frame_rate = 10  # Set the desired frame rate
    prev = 0

    while cap.isOpened():
        time_elapsed = time.time() - prev

        if time_elapsed > 1. / frame_rate:
            prev = time.time()
            ret, img = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
                continue
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
            imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY_INV, 25, 16)
            imgMedian = cv2.medianBlur(imgThreshold, 5)
            kernel = np.ones((3, 3), np.uint8)
            imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

            check_parking_space(img, imgDilate)

            frameST.image(img, channels="BGR")  # Display the image frame

    cap.release()

st.title("Car Parking Space Detection")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    frameST = st.empty()
    process_video_file(uploaded_file, frameST)
else:
    st.info("Please upload a video file to start processing.")

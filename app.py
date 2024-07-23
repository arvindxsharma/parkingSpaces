import cv2
import pickle
import cvzone
import numpy as np
import streamlit as st

# Load parking positions
with open("CarParkPos", 'rb') as f:
    posList = pickle.load(f)

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
        cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1, thickness=2, offset=0,
                           colorR=color)

    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3, thickness=5, offset=20,
                       colorR=(0, 200, 0))

def process_video_file(video_path):
    cap = cv2.VideoCapture(video_path)

    frameST = st.empty()  # Placeholder for video frames

    while True:  # Infinite loop for continuous video playback
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

        frameST.image(img, channels="BGR")

    cap.release()

st.title("Car Parking Space Detection")

# Path to the video file
video_path = 'carPark.mp4'

# Process the predefined video file
process_video_file(video_path)

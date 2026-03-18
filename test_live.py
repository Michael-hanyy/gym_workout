#!/usr/bin/env python3
"""
test_live.py
- Run webcam and classify exercises in real-time
"""

import cv2
import mediapipe as mp
import joblib
import numpy as np

MODEL_PATH = "exercise_classifier.pkl"
clf = joblib.load(MODEL_PATH)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

UPPER_BODY_IDX = [11,12,13,14,15,16]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        lms = results.pose_landmarks.landmark
        row = []

        # build full feature row (132)
        for i in range(33):
            if i in UPPER_BODY_IDX:  # pushup landmarks
                row += [lms[i].x, lms[i].y, lms[i].z, lms[i].visibility]
            else:
                row += [lms[i].x, lms[i].y, lms[i].z, lms[i].visibility]

        row = np.array(row).reshape(1,-1)
        pred = clf.predict(row)[0]

        cv2.putText(frame, f"Exercise: {pred}", (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

        drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Live Exercise Classification - Press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

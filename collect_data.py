#!/usr/bin/env python3
"""
collect_data.py (full update)
- Collect pose data from webcam
- Full-body for squats/jumpingjacks, upper-body for pushups
- Pushups are padded to match full-body features
- Saves to pose_dataset.csv
"""

import cv2
import mediapipe as mp
import csv
import time
import os

# === Config ===
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

exercises = ["squat", "pushup", "jumpingjack"]
samples_per_exercise = 80
CSV_PATH = "pose_dataset.csv"

# Full-body indices: head to ankle landmarks
FULL_BODY_IDX = [i for i in range(33)]
# Upper-body for pushups: shoulders, elbows, wrists
UPPER_BODY_IDX = [11,12,13,14,15,16]

VISIBILITY_THRESHOLD = 0.55
STABLE_FRAMES_FOR_START = 10
MODEL_COMPLEXITY = 2
MIN_DET_CONF = 0.7
MIN_TRACK_CONF = 0.7

# CSV header
if not os.path.exists(CSV_PATH):
    header = []
    for i in range(33):
        header += [f"x_{i}", f"y_{i}", f"z_{i}", f"v_{i}"]
    header.append("label")
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(header)

def landmarks_visibility_ok(landmarks, idx_list, threshold):
    return all(landmarks[i].visibility >= threshold for i in idx_list)

def draw_visibility_feedback(image, landmarks, threshold, idx_list):
    h, w = image.shape[:2]
    for i, lm in enumerate(landmarks):
        x, y = int(lm.x * w), int(lm.y * h)
        color = (0, 255, 0) if lm.visibility >= threshold else (0, 0, 255)
        cv2.circle(image, (x, y), 5, color, -1)
        if i in idx_list and lm.visibility < threshold:
            cv2.putText(image, f"low:{i}", (x+6, y-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

with mp_pose.Pose(model_complexity=MODEL_COMPLEXITY,
                  min_detection_confidence=MIN_DET_CONF,
                  min_tracking_confidence=MIN_TRACK_CONF) as pose, \
     open(CSV_PATH, "a", newline="") as f:

    writer = csv.writer(f)

    for exercise in exercises:
        collect = input(f"\nCollect samples for {exercise}? (y/n) [y]: ").lower()
        if collect == "n":
            continue

        print(f"\n--- Positioning hints for {exercise} ---")
        if exercise == "squat":
            print("Side camera recommended, full body visible.")
            critical_idx = FULL_BODY_IDX
        elif exercise == "pushup":
            print("Camera focused on upper body (shoulders, elbows, wrists).")
            critical_idx = UPPER_BODY_IDX
        else:
            print("Full body visible (jumpingjack).")
            critical_idx = FULL_BODY_IDX

        print("Starting in 4 seconds...")
        time.sleep(4)

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            print("❌ Cannot access webcam.")
            break

        count = 0
        collecting = False
        stable_good_frames = 0
        bad_frame_counter = 0

        while count < samples_per_exercise:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Frame capture failed.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark

                draw_visibility_feedback(frame, lms, VISIBILITY_THRESHOLD, critical_idx)

                if landmarks_visibility_ok(lms, critical_idx, VISIBILITY_THRESHOLD):
                    stable_good_frames += 1
                    bad_frame_counter = 0
                else:
                    stable_good_frames = 0
                    bad_frame_counter += 1

                if stable_good_frames >= STABLE_FRAMES_FOR_START and not collecting:
                    collecting = True
                    print("✅ Stable pose detected — starting collection now.")
                    time.sleep(0.3)

                # save row
                if collecting and landmarks_visibility_ok(lms, critical_idx, VISIBILITY_THRESHOLD):
                    row = []

                    if exercise == "pushup":
                        # upper-body only (6 points x 3 = 18 features)
                        for i in UPPER_BODY_IDX:
                            lm = lms[i]
                            row += [lm.x, lm.y, lm.z, lm.visibility]
                        # pad to full-body size (33 points x 4 = 132)
                        row += [0] * (33*4 - len(row))
                    else:
                        # full-body
                        for lm in lms:
                            row += [lm.x, lm.y, lm.z, lm.visibility]

                    row.append(exercise)
                    writer.writerow(row)
                    count += 1
                    cv2.rectangle(frame, (5,5), (frame.shape[1]-5, frame.shape[0]-5), (0,255,0), 2)

            else:
                stable_good_frames = 0

            status_text = f"{exercise} {count}/{samples_per_exercise}"
            color = (0,255,0) if collecting else (0,0,255)
            cv2.putText(frame, status_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            if bad_frame_counter > 8:
                cv2.putText(frame, "ADJUST CAMERA / LIGHTING / FRAME", (10,70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            cv2.imshow("Collecting Pose Data (Press 'q' to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.04)

        cap.release()
        cv2.destroyAllWindows()
        print(f"Finished collecting {count} samples for {exercise}.\n")

print("Done. Dataset appended to", CSV_PATH)

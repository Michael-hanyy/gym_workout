#!/usr/bin/env python3
"""
smart_gym_trainer_gui_full.py
- GUI front-end for smart_gym_trainer (integrates original rep counters)
- Tutorials played inside Tkinter, webcam preview inside Tkinter
- End session button, calories summary, full diet plan display
"""

import os,json
import time
import math
import threading
import cv2
import joblib
import numpy as np
import mediapipe as mp
from tkinter import Tk, Label, Button, Entry, StringVar, OptionMenu, Text, END, Frame, LEFT, RIGHT, BOTH, TOP
from PIL import Image, ImageTk
from coach import CoachLoginGUI, CoachDashboard, AddExerciseWizard, EditDietGUI


# === CONFIG ===
DISPLAY_WIDTH, DISPLAY_HEIGHT = 640, 480
TUTORIAL_DIR = "tutorials"
EXERCISES = ["squat", "pushup", "jumpingjack"]
MODEL_PATH = "exercise_classifier.pkl"

# Inference / counting params
VISIBILITY_THRESHOLD = 0.45
ANGLE_SMOOTHING_WINDOW = 5
MIN_TIME_BETWEEN_REPS = 0.6  # seconds
MET_VALUES = {"squat": 5.0, "pushup": 8.0, "jumpingjack": 8.0}

# Injury thresholds
KNEE_COLLAPSE_RATIO = 0.75
TORSO_LEAN_ANGLE_DEG = 20
PUSHUP_HIP_ANGLE_DROP_DEG = 12
JACK_ARM_RAISE_Y_DIFF = -0.08
JACK_LEG_OPEN_RATIO = 1.4
EXERCISES_FILE = "exercises.json"  # stores all exercises
DIET_FILE = "diet_plans.json"      # stores diets
# Diet plans (full text)
DEFAULT_EXERCISES = ["squat", "pushup", "jumping_jack"]
DEFAULT_DIETS = {
    "cutting": """CUTTING DIET PLAN
Goal: Lose fat while keeping muscle.
Protein: High (≈2g/kg)
Carbs: Medium–Low
Fats: Moderate

Sample meals:
- Breakfast: Oats + 3 egg whites + 1 whole egg
- Lunch: Chicken breast + rice + vegetables
- Snack: Greek yogurt or protein shake
- Dinner: Fish or chicken + salad
Avoid: Sugary snacks, fried food, soda
""",
    "shredding": """SHREDDING DIET PLAN
Goal: Maximize muscle definition.
Protein: Very high (≈2.2g/kg)
Carbs: Low
Fats: Low–Moderate

Sample meals:
- Breakfast: Boiled eggs + oats
- Lunch: Tuna/chicken + green salad
- Snack: Almonds or yogurt (low-fat)
- Dinner: Fish + broccoli
Avoid: Bread, pasta, sweets
""",
    "bulking": """BULKING DIET PLAN
Goal: Build muscle & strength.
Protein: 1.6–2g/kg
Carbs: High
Fats: Moderate

Sample meals:
- Breakfast: Eggs + oats + peanut butter
- Lunch: Beef or chicken + pasta/rice
- Snack: Nuts or peanut butter sandwich
- Dinner: Salmon or chicken + rice/potatoes
"""
}
# initialize files if missing
if not os.path.exists(EXERCISES_FILE):
    with open(EXERCISES_FILE, "w") as f:
        json.dump(DEFAULT_EXERCISES, f, indent=4)

if not os.path.exists(DIET_FILE):
    with open(DIET_FILE, "w") as f:
        json.dump(DEFAULT_DIETS, f, indent=4)

# load exercises
def load_exercises():
    with open(EXERCISES_FILE) as f:
        return json.load(f)

# save exercises
def save_exercises(ex_list):
    with open(EXERCISES_FILE, "w") as f:
        json.dump(ex_list, f, indent=4)

# load diet plans
def load_diets():
    with open(DIET_FILE) as f:
        return json.load(f)

# save diet plans
def save_diets(diets):
    with open(DIET_FILE, "w") as f:
        json.dump(diets, f, indent=4)

# ---------------- Mediapipe & Model ----------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
POSE = mp_pose.Pose(min_detection_confidence=0.55, min_tracking_confidence=0.55)

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("Loaded model:", MODEL_PATH)
    except Exception as e:
        print("Failed to load model:", e)
        model = None
else:
    print("No trained model found (exercise_classifier.pkl). Continuing without ML verification.")

# ---------------- Utilities (from your working code) ----------------
def calc_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return None
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def dist(a, b):
    a = np.array(a); b = np.array(b)
    return float(np.linalg.norm(a - b))

def landmarks_to_row(landmarks):
    row = []
    for lm in landmarks:
        row += [lm.x, lm.y, lm.z]
    return np.array(row).reshape(1, -1)

def try_find_tutorial_path(ex):
    candidates = [
        os.path.join(TUTORIAL_DIR, f"tutorial_{ex}.mp4"),
        os.path.join(TUTORIAL_DIR, f"{ex}.mp4"),
        os.path.join(TUTORIAL_DIR, f"{ex}_tutorial.mp4"),
        os.path.join(TUTORIAL_DIR, f"tutorial-{ex}.mp4"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def hip_width(landmarks):
    LHIP = [landmarks[23].x, landmarks[23].y]
    RHIP = [landmarks[24].x, landmarks[24].y]
    return abs(LHIP[0] - RHIP[0]) + 1e-6

def shoulder_width(landmarks):
    LSH = [landmarks[11].x, landmarks[11].y]
    RSH = [landmarks[12].x, landmarks[12].y]
    return abs(LSH[0] - RSH[0]) + 1e-6

def torso_angle_deg(landmarks):
    mid_sh = np.array([(landmarks[11].x + landmarks[12].x)/2, (landmarks[11].y + landmarks[12].y)/2])
    mid_hip = np.array([(landmarks[23].x + landmarks[24].x)/2, (landmarks[23].y + landmarks[24].y)/2])
    vec = mid_hip - mid_sh
    vertical = np.array([0.0, 1.0])
    denom = (np.linalg.norm(vec) * np.linalg.norm(vertical)) + 1e-8
    cos = np.clip(np.dot(vec, vertical)/denom, -1.0, 1.0)
    ang = math.degrees(math.acos(cos))
    return ang

# ---------------- Counters (copied exactly) ----------------
class CounterBase:
    def __init__(self, exercise):
        self.exercise = exercise
        self.count = 0
        self.stage = None
        self.last_rep_time = 0
        self.angle_buffer = []

    def _smooth(self, val):
        self.angle_buffer.append(val)
        if len(self.angle_buffer) > ANGLE_SMOOTHING_WINDOW:
            self.angle_buffer.pop(0)
        return float(np.median(self.angle_buffer))

class SquatCounter(CounterBase):
    def __init__(self):
        super().__init__("squat")
        self.down_ratio_thresh = 0.78
        self.up_ratio_thresh = 1.1

    def feed(self, lm):
        hip = [ (lm[23].x + lm[24].x)/2, (lm[23].y + lm[24].y)/2 ]
        knee = [ (lm[25].x + lm[26].x)/2, (lm[25].y + lm[26].y)/2 ]
        ankle = [ (lm[27].x + lm[28].x)/2, (lm[27].y + lm[28].y)/2 ]
        if any(v.visibility < VISIBILITY_THRESHOLD for v in [lm[23], lm[24], lm[25], lm[26], lm[27], lm[28]]):
            return False, None
        r = dist(hip, knee) / (dist(knee, ankle) + 1e-6)
        smooth_r = self._smooth(r)
        now = time.time()
        counted = False
        if smooth_r < self.down_ratio_thresh and self.stage != "down":
            self.stage = "down"
        if smooth_r > self.up_ratio_thresh and self.stage == "down" and (now - self.last_rep_time) > MIN_TIME_BETWEEN_REPS:
            self.count += 1
            self.last_rep_time = now
            self.stage = "up"
            counted = True
        return counted, smooth_r

class PushupCounter(CounterBase):
    def __init__(self):
        super().__init__("pushup")
        self.down_angle_thresh = 95.0
        self.up_angle_thresh = 155.0

    def feed(self, lm):
        if any(v.visibility < VISIBILITY_THRESHOLD for v in [lm[11], lm[12], lm[13], lm[14], lm[15], lm[16]]):
            return False, None
        left_angle = calc_angle([lm[11].x, lm[11].y], [lm[13].x, lm[13].y], [lm[15].x, lm[15].y])
        right_angle = calc_angle([lm[12].x, lm[12].y], [lm[14].x, lm[14].y], [lm[16].x, lm[16].y])
        if left_angle is None or right_angle is None:
            return False, None
        angle = (left_angle + right_angle) / 2.0
        smooth_a = self._smooth(angle)
        now = time.time()
        counted = False
        if smooth_a < self.down_angle_thresh and self.stage != "down":
            self.stage = "down"
        if smooth_a > self.up_angle_thresh and self.stage == "down" and (now - self.last_rep_time) > MIN_TIME_BETWEEN_REPS:
            self.count += 1
            self.last_rep_time = now
            self.stage = "up"
            counted = True
        return counted, smooth_a

class JackCounter(CounterBase):
    def __init__(self):
        super().__init__("jumpingjack")
        self.open_ratio_thresh = 0.45
        self.close_ratio_thresh = 0.28

    def feed(self, lm):
        if any(v.visibility < VISIBILITY_THRESHOLD for v in [lm[11], lm[12], lm[15], lm[16], lm[27], lm[28]]):
            return False, None
        wrist_dist = abs(lm[15].x - lm[16].x)
        shoulder_w = shoulder_width(lm)
        ratio = wrist_dist / (shoulder_w + 1e-6)
        smooth_r = self._smooth(ratio)
        now = time.time()
        counted = False
        if smooth_r > self.open_ratio_thresh and self.stage != "open":
            self.stage = "open"
        if smooth_r < self.close_ratio_thresh and self.stage == "open" and (now - self.last_rep_time) > MIN_TIME_BETWEEN_REPS:
            self.count += 1
            self.last_rep_time = now
            self.stage = "close"
            counted = True
        return counted, smooth_r

# ---------------- Injury check functions (copied) ----------------
def check_squat_injury(landmarks):
    warnings = []
    try:
        knee_x_dist = abs(landmarks[25].x - landmarks[26].x)
        hip_w = hip_width(landmarks)
        if knee_x_dist < (KNEE_COLLAPSE_RATIO * hip_w):
            warnings.append("Knees collapsing inward (valgus) — bend track caution")
        torso_deg = torso_angle_deg(landmarks)
        if torso_deg > TORSO_LEAN_ANGLE_DEG:
            warnings.append("Torso leaning/rounding — keep back straight")
    except Exception:
        pass
    return warnings

def check_pushup_injury(landmarks):
    warnings = []
    try:
        left_sh = [landmarks[11].x, landmarks[11].y]; left_hip = [landmarks[23].x, landmarks[23].y]; left_knee = [landmarks[25].x, landmarks[25].y]
        right_sh = [landmarks[12].x, landmarks[12].y]; right_hip = [landmarks[24].x, landmarks[24].y]; right_knee = [landmarks[26].x, landmarks[26].y]
        left_angle = calc_angle(left_sh, left_hip, left_knee)
        right_angle = calc_angle(right_sh, right_hip, right_knee)
        hip_angle = np.nanmean([left_angle or 180.0, right_angle or 180.0])
        if abs(180.0 - hip_angle) > PUSHUP_HIP_ANGLE_DROP_DEG:
            warnings.append("Hips sagging/raised - keep body straight (plank alignment)")
        l_elbow = calc_angle([landmarks[11].x, landmarks[11].y], [landmarks[13].x, landmarks[13].y], [landmarks[15].x, landmarks[15].y])
        r_elbow = calc_angle([landmarks[12].x, landmarks[12].y], [landmarks[14].x, landmarks[14].y], [landmarks[16].x, landmarks[16].y])
        avg_elbow = np.nanmean([l_elbow or 180.0, r_elbow or 180.0])
        if avg_elbow > 170:
            warnings.append("Elbows almost straight at bottom — reduce range or keep elbows tucked a bit")
    except Exception:
        pass
    return warnings

def check_jack_injury(landmarks):
    warnings = []
    try:
        left_wrist_y = landmarks[15].y
        right_wrist_y = landmarks[16].y
        left_sh_y = landmarks[11].y
        right_sh_y = landmarks[12].y
        avg_wr_y = (left_wrist_y + right_wrist_y) / 2.0
        avg_sh_y = (left_sh_y + right_sh_y) / 2.0
        diff = avg_wr_y - avg_sh_y
        if diff > JACK_ARM_RAISE_Y_DIFF:
            warnings.append("Arms not fully raised overhead")
        ankle_spread = abs(landmarks[27].x - landmarks[28].x)
        sh_w = shoulder_width(landmarks)
        if ankle_spread / (sh_w + 1e-6) < JACK_LEG_OPEN_RATIO:
            warnings.append("Legs not opening wide enough")
    except Exception:
        pass
    return warnings

# ---------------- GUI ----------------
class SmartTrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Gym Trainer")
        self.running=False
        self.cap=None
        self.counter=None
        self.start_time=None
        self.selected_ex = DEFAULT_EXERCISES[0]
        self.weight, self.height = 70.0, 170.0
        self.diet_choice="cutting"
        self.model=model

        self.exercises = load_exercises()
        self.diets = load_diets()

        # Controls
        controls=Frame(root)
        controls.pack(side=TOP, fill=BOTH, padx=8,pady=6)

        Label(controls,text="Weight (kg):").grid(row=0,column=0,sticky="w")
        self.weight_entry=Entry(controls,width=8); self.weight_entry.grid(row=0,column=1); self.weight_entry.insert(0,"70")
        Label(controls,text="Height (cm):").grid(row=0,column=2,sticky="w")
        self.height_entry=Entry(controls,width=8); self.height_entry.grid(row=0,column=3); self.height_entry.insert(0,"170")

        Label(controls,text="Exercise:").grid(row=1,column=0,sticky="w")
        self.ex_var=StringVar(value=self.exercises[0])
        self.ex_menu=OptionMenu(controls,self.ex_var,*self.exercises)
        self.ex_menu.grid(row=1,column=1)

        Label(controls,text="Diet:").grid(row=1,column=2,sticky="w")
        self.diet_var=StringVar(value="cutting")
        self.diet_menu=OptionMenu(controls,self.diet_var,*self.diets.keys())
        self.diet_menu.grid(row=1,column=3)

        self.start_btn=Button(controls,text="Start Session",command=self.start_session)
        self.start_btn.grid(row=2,column=0,columnspan=2,pady=6)
        self.end_btn=Button(controls,text="End Session",command=self.end_session,state="disabled")
        self.end_btn.grid(row=2,column=2,columnspan=2,pady=6)

        self.coach_btn=Button(controls,text="Coach Mode",command=self.open_coach_mode)
        self.coach_btn.grid(row=0,column=4,rowspan=2,padx=10)

        # Display + log
        bottom=Frame(root)
        bottom.pack(side=TOP,fill=BOTH,expand=True,padx=8,pady=6)
        self.display_label=Label(bottom); self.display_label.pack(side=LEFT,padx=6,pady=6)
        self.log_text=Text(bottom,width=48,height=22); self.log_text.pack(side=RIGHT,padx=6,pady=6)

    def log(self,msg):
        now=time.strftime("%H:%M:%S")
        self.log_text.insert(END,f"[{now}] {msg}\n")
        self.log_text.see(END)

    def start_session(self):
        try:self.weight=float(self.weight_entry.get())
        except: self.weight=70.0
        try:self.height=float(self.height_entry.get())
        except: self.height=170.0
        self.selected_ex=self.ex_var.get()
        self.diet_choice=self.diet_var.get()
        self.log(f"Starting: {self.selected_ex}  |  Weight: {self.weight} kg  |  Height: {self.height} cm  |  Diet: {self.diet_choice}")
        self.start_btn.config(state="disabled"); self.end_btn.config(state="normal")
        threading.Thread(target=self.session_thread,daemon=True).start()

    def open_coach_mode(self):
        self.log("Opening Coach Mode...")
        CoachLoginGUI(self.root)

    def launch_coach_dashboard(self, coach_user):
        CoachDashboard(self.root, main_gui=self)

    def session_thread(self):
        # 1) Play tutorial (inside GUI)
        tut_path = try_find_tutorial_path(self.selected_ex)
        if tut_path:
            self.log(f"Playing tutorial: {os.path.basename(tut_path)}  (press End Session to skip)")
            cap_tut = cv2.VideoCapture(tut_path)
            # play tutorial until end or until user hits end_session
            while cap_tut.isOpened() and (not self.running):
                ret, frame = cap_tut.read()
                if not ret:
                    break
                self._show_frame(frame)
                # small sleep to respect video framerate
                time.sleep(1/30.0)
                if not self.end_btn['state'] == "normal":
                    break
            cap_tut.release()
        else:
            self.log("No tutorial found; skipping tutorial.")

        # 2) Countdown (3 seconds)
        for s in range(3, 0, -1):
            frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
            cv2.putText(frame, f"Starting in {s}", (DISPLAY_WIDTH//6, DISPLAY_HEIGHT//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0,0,255), 5, cv2.LINE_AA)
            self._show_frame(frame)
            time.sleep(1.0)

        # 3) Start live webcam session
        self.running = True
        self.start_time = time.time()
        # instantiate the same counters you had
        if self.selected_ex == "squat":
            self.counter = SquatCounter()
        elif self.selected_ex == "pushup":
            self.counter = PushupCounter()
        else:
            self.counter = JackCounter()

        # open camera (try default)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            # try alternate backends or backend flag (mac sometimes needs different)
            self.log("Primary webcam open failed; trying alternate index/backends...")
            # try index 1
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                self.log("Cannot open webcam.")
                self.running = False
                self.start_btn.config(state="normal")
                self.end_btn.config(state="disabled")
                return

        # main loop
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.log("Frame read failed; stopping.")
                break
            # flip to act like mirror (as original)
            frame = cv2.flip(frame, 1)
            # process pose
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = POSE.process(rgb)
            predicted = None
            try:
                if self.model is not None and results.pose_landmarks:
                    row = landmarks_to_row(results.pose_landmarks.landmark)
                    predicted = self.model.predict(row)[0]
            except Exception:
                predicted = None

            counted = False
            last_metric = None
            warnings = []

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                                          connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,200,0), thickness=2))
                counted, last_metric = self.counter.feed(results.pose_landmarks.landmark)
                # injury checks
                if self.selected_ex == "squat":
                    warnings = check_squat_injury(results.pose_landmarks.landmark)
                elif self.selected_ex == "pushup":
                    warnings = check_pushup_injury(results.pose_landmarks.landmark)
                else:
                    warnings = check_jack_injury(results.pose_landmarks.landmark)
                
            # overlay panel (draw on frame)
            # Make overlay box sized based on display width for clarity
            overlay_h = 140
            cv2.rectangle(frame, (0, 0), (DISPLAY_WIDTH, overlay_h), (0,0,0), -1)
            cv2.putText(frame, f"Exercise: {self.selected_ex}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"Reps: {self.counter.count}", (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
            elapsed = int(time.time() - self.start_time)
            cv2.putText(frame, f"Time: {elapsed}s", (10, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            if last_metric is not None:
                if self.selected_ex == "squat":
                    cv2.putText(frame, f"Ratio: {last_metric:.2f}", (300, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)
                elif self.selected_ex == "pushup":
                    cv2.putText(frame, f"Elbow: {last_metric:.1f} deg", (300, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)
                else:
                    cv2.putText(frame, f"Hands ratio: {last_metric:.2f}", (300, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)

            if predicted:
                cv2.putText(frame, f"Model: {predicted}", (10, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)

            if counted:
                # flash +1 near the top-right
                cv2.putText(frame, "+1", (DISPLAY_WIDTH-80, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0,255,0), 4)

            if warnings:
                cv2.putText(frame, f"⚠ {warnings[0]}", (10, overlay_h+28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # ensure frame is the display size (no zoom)
            frame_disp = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            self._show_frame(frame_disp)

        # end of session cleanup
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.running = False
        # compute summary
        duration_s = max(1, int(time.time() - self.start_time)) if self.start_time else 0
        duration_hours = max(1e-6, duration_s) / 3600.0
        calories = MET_VALUES.get(self.selected_ex, 5.0) * self.weight * duration_hours
        self.log(f"Session ended. Duration: {duration_s}s | Reps: {self.counter.count} | Calories: {round(calories,2)} kcal")
        self.log(f"DIET PLAN ({self.diet_choice}):")
        for line in DIET_FILE.get(self.diet_choice, "").splitlines():
            self.log(line)
        # re-enable start button
        self.start_btn.config(state="normal")
        self.end_btn.config(state="disabled")

    def _show_frame(self, frame_bgr):
        """Convert BGR frame (numpy) to Tk image and show."""
        # frame expected to be BGR (OpenCV). Convert properly.
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            # if already RGB or grayscale fallback
            frame_rgb = frame_bgr
        # resize to display dimensions
        frame_rgb = cv2.resize(frame_rgb, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        # set image
        self.display_label.imgtk = imgtk
        self.display_label.config(image=imgtk)
        # update UI
        # Use update_idletasks instead of update to reduce flicker
        self.root.update_idletasks()

    def end_session(self):
        if self.running:
            self.log("End requested by user — stopping session...")
            self.running = False
        else:
            self.log("Session not running.")

# --------- Main ----------
if __name__ == "__main__":
    root = Tk()
    app = SmartTrainerGUI(root)
    root.mainloop()

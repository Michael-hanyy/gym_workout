import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
import cv2
import mediapipe as mp
import csv
import time
import os
import json
import shutil

# =========================
# CONFIG
# =========================
COACH_ACCOUNTS = {"bavly": "1234", "michael": "0000"}

CSV_PATH = "pose_dataset.csv"
EXERCISE_META_PATH = "exercise_meta.json"
DIET_PATH = "diet_plans.json"
INJURY_PATH = "injury_notes.json"
TUTORIALS_DIR = "tutorials"

SAMPLES_PER_EXERCISE = 80
STABLE_FRAMES_FOR_START = 10

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

os.makedirs(TUTORIALS_DIR, exist_ok=True)

# =========================
# UTILS
# =========================
def ensure_files():
    for path, default in [
        (EXERCISE_META_PATH, {}),
        (DIET_PATH, {"cutting": "Calorie deficit, high protein",
                     "bulking": "Calorie surplus, strength focus",
                     "shredding": "Low carb, fat loss"}),
        (INJURY_PATH, {})
    ]:
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump(default, f, indent=4)

    if not os.path.exists(CSV_PATH):
        header = []
        for i in range(33):
            header += [f"x_{i}", f"y_{i}", f"z_{i}", f"v_{i}"]
        header.append("label")
        with open(CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(header)

ensure_files()

# =========================
# FILE HANDLING
# =========================
def load_exercises():
    """Return list of exercises from metadata + default"""
    with open(EXERCISE_META_PATH, "r") as f:
        data = json.load(f)
    default = ["squat", "pushup", "jumping_jack"]
    return default + list(data.keys())

def save_exercise_metadata(name, mode, rule, calories):
    with open(EXERCISE_META_PATH, "r") as f:
        data = json.load(f)
    data[name] = {"mode": mode, "rules": rule, "calories_per_rep": calories}
    with open(EXERCISE_META_PATH, "w") as f:
        json.dump(data, f, indent=4)

def load_diets():
    with open(DIET_PATH, "r") as f:
        return json.load(f)

def save_diets(diets):
    with open(DIET_PATH, "w") as f:
        json.dump(diets, f, indent=4)

def load_injuries():
    with open(INJURY_PATH, "r") as f:
        return json.load(f)

def save_injuries(notes):
    with open(INJURY_PATH, "w") as f:
        json.dump(notes, f, indent=4)

# =========================
# COACH LOGIN
# =========================
class CoachLoginGUI:
    def __init__(self, root, on_update_callback=None):
        self.root = root
        self.on_update_callback = on_update_callback
        self.window = tk.Toplevel(root)
        self.window.title("Coach Login")
        self.window.geometry("300x200")

        tk.Label(self.window, text="Username").pack()
        self.username = tk.Entry(self.window)
        self.username.pack()

        tk.Label(self.window, text="Password").pack()
        self.password = tk.Entry(self.window, show="*")
        self.password.pack()

        tk.Button(self.window, text="Login", command=self.login).pack(pady=10)

    def login(self):
        if self.username.get() in COACH_ACCOUNTS and \
           self.password.get() == COACH_ACCOUNTS[self.username.get()]:
            self.window.destroy()
            CoachDashboard(self.root, self.on_update_callback)
        else:
            messagebox.showerror("Error", "Invalid credentials")

# =========================
# COACH DASHBOARD
# =========================
class CoachDashboard:
    def __init__(self, root, on_update_callback=None):
        self.on_update_callback = on_update_callback
        self.window = tk.Toplevel(root)
        self.window.title("Coach Dashboard")
        self.window.geometry("400x300")

        tk.Button(self.window, text="➕ Add Exercise", width=30, command=self.add_exercise).pack(pady=10)
        tk.Button(self.window, text="🗑 Delete Exercise", width=30, command=self.delete_exercise).pack(pady=10)
        tk.Button(self.window, text="🍽 Edit Diet Plans", width=30, command=self.edit_diet).pack(pady=10)
        tk.Button(self.window, text="⚠ Edit Injury Notes", width=30, command=self.edit_injury).pack(pady=10)
        tk.Button(self.window, text="❌ Close", width=30, command=self.window.destroy).pack(pady=10)

    def add_exercise(self):
        AddExerciseWizard(self.window, self.on_update_callback)

    def delete_exercise(self):
        exercises = list(load_exercises())
        exercises = [ex for ex in exercises if ex not in ["squat", "pushup", "jumping_jack"]]
        if not exercises:
            messagebox.showwarning("Warning", "No custom exercises to delete")
            return
        name = simpledialog.askstring("Delete Exercise", f"Enter exercise name to delete:\n{', '.join(exercises)}")
        if not name or name not in exercises:
            messagebox.showwarning("Warning", "Exercise not found")
            return
        # Remove tutorial if exists
        tut_file = os.path.join(TUTORIALS_DIR, f"{name}.mp4")
        if os.path.exists(tut_file):
            os.remove(tut_file)
        # Remove from metadata
        with open(EXERCISE_META_PATH, "r") as f:
            data = json.load(f)
        if name in data:
            del data[name]
        with open(EXERCISE_META_PATH, "w") as f:
            json.dump(data, f, indent=4)
        messagebox.showinfo("Deleted", f"{name} deleted successfully")
        if self.on_update_callback:
            self.on_update_callback()

    def edit_diet(self):
        EditDietGUI(self.window)

    def edit_injury(self):
        EditInjuryGUI(self.window)

# =========================
# ADD EXERCISE WIZARD
# =========================
class AddExerciseWizard:
    def __init__(self, parent, on_update_callback=None):
        self.on_update_callback = on_update_callback
        self.exercise = simpledialog.askstring("Exercise Name", "Enter exercise name:")
        if not self.exercise:
            return

        self.mode = simpledialog.askstring("Detection Mode", "Type 'angle' or 'ratio':")
        self.rule = simpledialog.askstring("Rules", "Enter angles or ratios (comma separated):")
        self.calories = simpledialog.askfloat("Calories", "Calories per rep:")

        # Tutorial video
        self.tutorial_path = filedialog.askopenfilename(title="Select Tutorial Video", filetypes=[("MP4 files", "*.mp4")])
        if self.tutorial_path:
            shutil.copy(self.tutorial_path, os.path.join(TUTORIALS_DIR, f"{self.exercise}.mp4"))

        self.collect_data()
        save_exercise_metadata(self.exercise, self.mode, self.rule, self.calories)

        # Add empty injury notes
        injuries = load_injuries()
        injuries[self.exercise] = []
        save_injuries(injuries)

        if self.on_update_callback:
            self.on_update_callback()

        messagebox.showinfo("Success", f"{self.exercise} added successfully!")

    def collect_data(self):
        cap = cv2.VideoCapture(0)
        pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)

            collected = 0
            stable = 0
            collecting = False

            while collected < SAMPLES_PER_EXERCISE:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if results.pose_landmarks:
                    stable += 1
                    if stable >= STABLE_FRAMES_FOR_START:
                        collecting = True

                    if collecting:
                        row = []
                        for lm in results.pose_landmarks.landmark:
                            row += [lm.x, lm.y, lm.z, lm.visibility]
                        row.append(self.exercise)
                        writer.writerow(row)
                        collected += 1

                cv2.putText(frame, f"{self.exercise}: {collected}/{SAMPLES_PER_EXERCISE}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.imshow("Collecting Exercise Data", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

# =========================
# EDIT DIET GUI
# =========================
class EditDietGUI:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("Edit Diet Plans")
        self.window.geometry("400x300")

        self.diets = load_diets()
        self.entries = {}

        for k, v in self.diets.items():
            tk.Label(self.window, text=k).pack()
            e = tk.Entry(self.window, width=50)
            e.insert(0, v)
            e.pack()
            self.entries[k] = e

        tk.Button(self.window, text="Save", command=self.save).pack(pady=10)

    def save(self):
        for k, e in self.entries.items():
            self.diets[k] = e.get()
        save_diets(self.diets)
        messagebox.showinfo("Saved", "Diet plans updated")

# =========================
# EDIT INJURY GUI
# =========================
class EditInjuryGUI:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("Edit Injury Notes")
        self.window.geometry("400x300")

        self.notes = load_injuries()
        self.ex_var = tk.StringVar(value=list(self.notes.keys())[0])
        tk.OptionMenu(self.window, self.ex_var, *self.notes.keys()).pack(pady=5)

        self.text_entry = tk.Text(self.window, width=50, height=10)
        self.text_entry.pack(pady=5)
        self.load_notes()

        tk.Button(self.window, text="Save", command=self.save).pack(pady=10)
        self.ex_var.trace("w", lambda *args: self.load_notes())

    def load_notes(self):
        ex = self.ex_var.get()
        self.text_entry.delete("1.0", tk.END)
        if ex in self.notes:
            self.text_entry.insert(tk.END, "\n".join(self.notes[ex]))

    def save(self):
        ex = self.ex_var.get()
        self.notes[ex] = self.text_entry.get("1.0", tk.END).splitlines()
        save_injuries(self.notes)
        messagebox.showinfo("Saved", f"Injury notes for {ex} updated")

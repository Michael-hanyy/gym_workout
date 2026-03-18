import cv2
import time
import os

TUTORIAL_DIR = "tutorials"
EXERCISES = ["squat", "pushup", "jumpingjack"]
RECORD_SECONDS = 7   # Duration per tutorial video
COUNTDOWN = 3       # Seconds before recording starts


def countdown_display(seconds):
    """Display a countdown overlay using the camera feed."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ ERROR: Camera failed to open!")
        return False

    for i in range(seconds, 0, -1):
        ret, frame = cap.read()
        if not ret:
            print("❌ ERROR: Failed to read frame")
            break

        # Draw countdown number
        cv2.putText(frame, f"Starting in {i}...",
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 0, 255), 3)

        cv2.imshow("Tutorial Countdown", frame)
        cv2.waitKey(1000)  # wait EXACTLY 1 second

    cap.release()
    cv2.destroyWindow("Tutorial Countdown")
    return True


def record_video(filename, duration=7):
    """Record the tutorial video."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ ERROR: Camera failed to open!")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))

    print(f"🎥 Recording started: {filename}")
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ ERROR: Failed to read frame")
            break

        out.write(frame)
        cv2.imshow("Recording Tutorial", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("⏹ Manual stop.")
            break

        if (time.time() - start_time) >= duration:
            print("⏹ Finished recording.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return True



def main():
    if not os.path.exists(TUTORIAL_DIR):
        os.makedirs(TUTORIAL_DIR)

    print("=== TUTORIAL VIDEO RECORDER ===")
    print("You will get a 3-second countdown before recording.\n")

    for ex in EXERCISES:
        print(f"Ready to record: {ex.upper()}")
        input("Press ENTER to begin countdown...")

        print("⏳ Starting countdown...")
        countdown_display(COUNTDOWN)

        filename = os.path.join(TUTORIAL_DIR, f"{ex}.mp4")
        success = record_video(filename, RECORD_SECONDS)

        if success:
            print(f"✅ Saved tutorial for {ex}: {filename}\n")
        else:
            print(f"❌ Failed to record tutorial for {ex}\n")

    print("🎉 All tutorials recorded successfully!")


if __name__ == "__main__":
    main()

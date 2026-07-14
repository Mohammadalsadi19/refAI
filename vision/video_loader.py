import cv2
from pathlib import Path

VIDEO_DIR = Path("videos")

# ابحث عن أول فيديو داخل جميع الأقسام
video_files = list(VIDEO_DIR.glob("*/*.mp4"))

if not video_files:
    raise FileNotFoundError("No videos found.")

video_path = video_files[0]

print("Loading:", video_path)

cap = cv2.VideoCapture(str(video_path))

while True:

    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("RefAI Video", frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
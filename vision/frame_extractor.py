import cv2
from pathlib import Path

VIDEO_DIR = Path("videos")
FRAME_DIR = Path("frames")

FRAME_STEP = 5

for video in VIDEO_DIR.glob("*/*.mp4"):

    category = video.parent.name
    video_name = video.stem

    save_dir = FRAME_DIR / category / video_name
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))

    frame_id = 0
    saved = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % FRAME_STEP == 0:

            filename = save_dir / f"{saved:04d}.jpg"

            cv2.imwrite(str(filename), frame)

            saved += 1

        frame_id += 1

    cap.release()

    print(f"{video_name} -> {saved} frames")
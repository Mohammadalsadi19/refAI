from pathlib import Path
import json

import cv2
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"
FRAME_DIR = Path("frames")
TRACKS_DIR = Path("tracks")

PERSON_CLASS_ID = 0
BALL_CLASS_ID = 32  # COCO 'sports ball' — a generic approximation.
# For a real football, this is unreliable (small/fast-moving object,
# easily confused with other round shapes). Fine-tuning YOLO on your own
# labeled football frames will matter a lot here — flagging as future work.

model = YOLO(MODEL_PATH)


def track_video_frames(frame_folder):
    """Runs YOLO tracking (with persistent IDs) over a sorted frame
    sequence for one video clip. Returns a flat list of per-frame
    detections with consistent track_id across frames."""
    frame_paths = sorted(frame_folder.glob("*.jpg"))
    if not frame_paths:
        return []

    tracks = []
    for frame_id, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[PERSON_CLASS_ID, BALL_CLASS_ID],
        verbose=False,
    )
        result = results[0]

        if result.boxes is None or result.boxes.id is None:
            continue  # nothing tracked in this frame

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for box, track_id, cls, conf in zip(boxes, ids, classes, confidences):

            x1, y1, x2, y2 = box

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            width = x2 - x1
            height = y2 - y1
            area = width * height

            tracks.append({
                "frame_id": frame_id,
                "track_id": int(track_id),
                "class_name": "ball" if cls == BALL_CLASS_ID else "person",
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "centroid": [float(cx), float(cy)],
                "width": float(width),
                "height": float(height),
                "area": float(area),
                "confidence": float(conf)
            })

    return tracks


def main():
    if not FRAME_DIR.exists():
        raise FileNotFoundError(f"{FRAME_DIR} not found — run frame_extractor.py first.")

    for category_dir in sorted(FRAME_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        for video_dir in sorted(category_dir.iterdir()):
            if not video_dir.is_dir():
                continue

            print(f"Tracking {category_dir.name}/{video_dir.name} ...")
            tracks = track_video_frames(video_dir)

            out_path = TRACKS_DIR / category_dir.name / f"{video_dir.name}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(tracks, f, indent=2)

            print(f"  -> {len(tracks)} track points saved to {out_path}")


if __name__ == "__main__":
    main()
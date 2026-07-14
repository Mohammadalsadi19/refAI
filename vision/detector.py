from ultralytics import YOLO
import cv2
from pathlib import Path

model = YOLO("yolov8n.pt")

FRAME_DIR = Path("frames")

image_files = list(FRAME_DIR.glob("*/*/*.jpg"))

if not image_files:
    raise FileNotFoundError("No frames found.")

image = image_files[0]

print("Testing:", image)

results = model(str(image))

annotated = results[0].plot()

cv2.imshow("YOLO Detection", annotated)

cv2.waitKey(0)

cv2.destroyAllWindows()
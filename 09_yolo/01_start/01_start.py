from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

model = YOLO('yolov8n.pt')
results = model.predict(
    source='./ultralytics-main/ultralytics/assets/bus.jpg', 
    save=True,
    conf=0.05,
    project=PROJECT_ROOT / 'runs',
    name='predict',
    exist_ok=True)
print(results)

from typing import List
import numpy as np
from ultralytics import YOLO

YOLO_MODEL_PATH = "app/models/yolo_model.pt"

model = YOLO(YOLO_MODEL_PATH)

def detect_logo_YOLO(image: np.ndarray) -> List[List[float]]:
    to_return = []
    results = model.predict(image)
    for box in results[0].boxes:
        to_return.append(box.xyxy.tolist()[0])
    return to_return
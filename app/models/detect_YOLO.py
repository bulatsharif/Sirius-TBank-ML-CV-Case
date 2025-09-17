from typing import List, Optional
import numpy as np
from ultralytics import YOLO
import logging

YOLO_MODEL_PATH = "app/models/yolo_model.pt"

logger = logging.getLogger(__name__)

_cached_model: Optional[YOLO] = None
_model_load_error: Optional[Exception] = None

def _get_model() -> YOLO:
    global _cached_model, _model_load_error
    if _model_load_error is not None:
        raise RuntimeError("Model unavailable")
    if _cached_model is None:
        try:
            _cached_model = YOLO(YOLO_MODEL_PATH)
        except Exception as exc:
            _model_load_error = exc
            logger.exception("Failed to load YOLO model from %s", YOLO_MODEL_PATH)
            raise RuntimeError("Model load failed") from exc
    return _cached_model

def detect_logo_YOLO(image: np.ndarray) -> List[List[float]]:
    model = _get_model()
    results = model.predict(image)
    return [box.xyxy.tolist()[0] for box in results[0].boxes]
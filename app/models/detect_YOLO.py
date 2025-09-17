from typing import List, Optional
import numpy as np
from ultralytics import YOLO
import logging
from pathlib import Path
from app.core.config import YOLO_MODEL_PATH

logger = logging.getLogger(__name__)

_cached_model: Optional[YOLO] = None
_model_load_error: Optional[Exception] = None

def _get_model() -> YOLO:
    # Функция, чтобы кешировать модель, и не загружать каждый раз
    global _cached_model, _model_load_error
    if _model_load_error is not None:
        raise RuntimeError("Model unavailable")
    if _cached_model is None:
        try:
            model_path = Path(YOLO_MODEL_PATH)
            _cached_model = YOLO(str(model_path))
        except Exception as exc:
            _model_load_error = exc
            logger.exception("Не получилось загрузить модель, путь: %s", YOLO_MODEL_PATH)
            raise RuntimeError("Не получилось загрузить модель") from exc
    return _cached_model

def detect_logo_YOLO(image: np.ndarray) -> List[List[float]]:
    model = _get_model()
    results = model.predict(image, verbose=False)
    return [box.xyxy.tolist()[0] for box in results[0].boxes]
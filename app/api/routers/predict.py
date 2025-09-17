import cv2
import numpy as np
import logging
from fastapi import APIRouter, HTTPException, File, UploadFile
from app.schemas.predict import DetectionResponse, ErrorResponse, Detection, BoundingBox
from app.models.detect_YOLO import detect_logo_YOLO
from app.core.config import ALLOWED_IMAGE_CONTENT_TYPES

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post(
    "/detect",
    response_model=DetectionResponse,
    responses={
        400: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def detect_logo(file: UploadFile = File(...)):
    """
    Детекция логотипа Т-банка на изображении

    Args:
        file: Загружаемое изображение (JPEG, PNG, BMP, WEBP)

    Returns:
        DetectionResponse: Результаты детекции с координатами найденных логотипов
    """
    if file.content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Ваш формат изображения не поддерживается; Поддерживаемые форматы: JPEG, PNG, BMP, WEBP")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Файл пустой")

    nparr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  

    if image is None:
        raise HTTPException(status_code=400, detail="Неверный формат изображения")

    try:
        bboxes = detect_logo_YOLO(image)
        detections = []
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # Вероятно, проблема с загрузкой модели или ее доступностью
        logger.error("Модель недоступна: %s", e)
        raise HTTPException(status_code=503, detail="Модель недоступна, попробуйте позже")
    except Exception:
        logger.exception("Сбой детекции")
        raise HTTPException(status_code=500, detail="Сбой детекции")

    if not bboxes:
        return DetectionResponse(detections=[])

    for bbox in bboxes:
        x_min = int(min(bbox[0], bbox[2]))
        y_min = int(min(bbox[1], bbox[3]))
        x_max = int(max(bbox[0], bbox[2]))
        y_max = int(max(bbox[1], bbox[3]))

        detections.append(Detection(bbox=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)))

    response = DetectionResponse(detections=detections)
    return response
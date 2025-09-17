import cv2
import numpy as np
import logging
from fastapi import APIRouter, HTTPException, File, UploadFile
from app.schemas.predict import DetectionResponse, ErrorResponse, Detection, BoundingBox
from app.models.detect_YOLO import detect_logo_YOLO

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
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=415, detail="Unsupported media type")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File is empty")

    nparr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    # image has shape (height, width, channels); height/width not used below

    try:
        bboxes = detect_logo_YOLO(image)
        detections = []
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # Likely model loading or availability problem
        logger.error("Model unavailable: %s", e)
        raise HTTPException(status_code=503, detail="Model unavailable. Try later.")
    except Exception:
        logger.exception("Detection failed")
        raise HTTPException(status_code=500, detail="Detection failed")

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
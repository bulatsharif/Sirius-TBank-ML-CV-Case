import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, File, UploadFile
from app.schemas.predict import DetectionResponse, ErrorResponse, Detection, BoundingBox
from app.models.detect_YOLO import detect_logo_YOLO

router = APIRouter()

@router.post("/detect", response_model=DetectionResponse, responses={400: {"model": ErrorResponse}})
async def detect_logo(file: UploadFile = File(...)):
    """
    Детекция логотипа Т-банка на изображении

    Args:
        file: Загружаемое изображение (JPEG, PNG, BMP, WEBP)

    Returns:
        DetectionResponse: Результаты детекции с координатами найденных логотипов
    """
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
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

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
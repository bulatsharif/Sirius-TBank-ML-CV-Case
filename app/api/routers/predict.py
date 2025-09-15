import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, File, UploadFile
from app.schemas.predict import DetectionResponse, ErrorResponse, Detection, BoundingBox
from app.models.model import detect_logo_SIFT_RANSAC

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
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    height, width = image.shape

    try:
        bboxes = detect_logo_SIFT_RANSAC(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

    if not bboxes:
        return DetectionResponse(detections=[])

    x_coords = [p[0] for p in bboxes]
    y_coords = [p[1] for p in bboxes]

    x_min = int(max(0, min(x_coords)))
    y_min = int(max(0, min(y_coords)))
    x_max = int(min(width, max(x_coords)))
    y_max = int(min(height, max(y_coords)))

    response = DetectionResponse(
        detections=[
            Detection(bbox=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max))
        ]
    )
    return response
from fastapi import APIRouter, HTTPException, File, UploadFile
from app.schemas.predict import DetectionResponse, ErrorResponse

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
    
    # TODO: handle different file types
    return {"detections": []}
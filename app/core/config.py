from pathlib import Path
from typing import Set


# Здесь будут храниться константы и общая конфигурация для приложения


BASE_DIR = Path(__file__).resolve().parent.parent


# Путь к модели YOLO внутри контейнера/репозитория
YOLO_MODEL_PATH: str = str(BASE_DIR / "models" / "yolo_model.pt")


# Поддерживаемые типы изображений для API
ALLOWED_IMAGE_CONTENT_TYPES: Set[str] = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
}



from fastapi.testclient import TestClient
import cv2
import numpy as np
from app.main import app


client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def _encode_image(ext: str, color: tuple = (0, 255, 0)) -> bytes:
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[:] = color
    success, buf = cv2.imencode(ext, img)
    assert success
    return buf.tobytes()


def test_detect_accepts_supported_types(monkeypatch):
    from app.api.routers import predict as predict_module

    def fake_detect_logo_YOLO(image):
        return []

    monkeypatch.setattr(predict_module, "detect_logo_YOLO", fake_detect_logo_YOLO)

    files = [
        ("/detect_jpeg", "image/jpeg", ".jpg"),
        ("/detect_png", "image/png", ".png"),
        ("/detect_webp", "image/webp", ".webp"),
        ("/detect_bmp", "image/bmp", ".bmp"),
    ]

    for _, content_type, ext in files:
        data = _encode_image(ext)
        r = client.post(
            "/api/v1/detect",
            files={"file": (f"test{ext}", data, content_type)},
        )
        assert r.status_code == 200
        assert r.json()["detections"] == []



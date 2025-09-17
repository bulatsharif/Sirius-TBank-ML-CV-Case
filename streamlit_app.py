import streamlit as st
import cv2
import numpy as np
from PIL import Image
from app.models.detect_YOLO import detect_logo_YOLO
import time

st.title("Детекция логотипа Т-банка")

uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "png", "webp", "bmp"])

if uploaded_file is not None:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        if opencv_image is None:
            st.error("Неверный формат изображения.")
        else:
            st.image(opencv_image, channels="BGR", caption="Загруженное изображение")
            st.write("Детекция логотипа...")
            try:
                # Keep BGR consistently (same as API)
                start_time = time.perf_counter()
                bboxes = detect_logo_YOLO(opencv_image)
                end_time = time.perf_counter()
                elapsed_ms = (end_time - start_time) * 1000.0
            except Exception:
                st.error("Сбой детекции. Попробуйте другое изображение или перезагрузите приложение.")
                bboxes = []

            if bboxes:
                img_with_box = opencv_image.copy()
                st.write("Логотип найден!")
                for bbox in bboxes:
                    x_min = min(bbox[0], bbox[2])
                    y_min = min(bbox[1], bbox[3])
                    x_max = max(bbox[0], bbox[2])
                    y_max = max(bbox[1], bbox[3])
                    x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
                    p1 = (x_min, y_min)
                    p2 = (x_max, y_max)
                    p3 = (x_max, y_min)
                    p4 = (x_min, y_max)
                    pts = np.array([p1, p3, p2, p4], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    img_with_box = cv2.polylines(img_with_box, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
                st.image(img_with_box, channels="BGR", caption="Обработанное изображение")
                st.caption(f"Время инференса: {elapsed_ms:.1f} мс")
            else:
                st.write("Логотип не найден.")
    except Exception:
        st.error("Ошибка обработки файла.")

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from app.models.model import detect_logo_SIFT_RANSAC, TBANK_LOGO

st.title("Детекция логотипа Т-банка")

uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "png", "webp", "bmp"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    
    st.image(opencv_image, channels="BGR", caption="Загруженное изображение")

    st.write("Детекция логотипа...")
    
    gray_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    bboxes = detect_logo_SIFT_RANSAC(gray_image)
    

    if bboxes:
        st.write("Логотип найден!")
        
        pts = np.array(bboxes, np.int32)
        pts = pts.reshape((-1, 1, 2))
        img_with_box = cv2.polylines(opencv_image.copy(), [pts], isClosed=True, color=(0, 255, 0), thickness=3)

        st.image(img_with_box, channels="BGR", caption="Обработанное изображение")
    else:
        st.write("Логотип не найден.")

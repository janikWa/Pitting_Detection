import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os 
from utils import * 
import torchvision.transforms as transforms

BASE = os.getcwd()


# Streamlit app
def main():

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    st.set_page_config(
        page_title="WBK Pitting Detection",
        page_icon= "random",
        layout="wide",
    )

    logo = r"/Users/janikwahrheit/Library/CloudStorage/OneDrive-Persönlich/01_Studium/01_Bachelor/06. SS_2025/Pitting_Detection/images/wbk.png"



    st.markdown("""
    <style>
    #stDecoration {
        background-image: linear-gradient(90deg, #d6d6d6, #26ab71) !important;
        height: 2px !important; 
    }
    </style>
    """, unsafe_allow_html=True)


    st.title("🔩Pitting Detection")
    st.logo(logo, icon_image=logo, size="large")

    #model = load_model()‚

    with st.sidebar: 

        user = st.text_input("E-Mail Adresse", help="E-Mail Adresse für automatische Bewarnung")
        run = st.toggle("Start Video Stream")

    visualizer = st.pills("Visualisiertung", ["Bounding-Box", "Heatmap"], selection_mode="single", default="Bounding-Box")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        FRAME_WINDOW = st.image([], use_container_width=True)

    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        prediction = predict_pitting(None, frame)
        frame = draw_prediction_box(frame, prediction)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with col2:
            FRAME_WINDOW.image(frame)

    cap.release()

if __name__ == '__main__':
    main()

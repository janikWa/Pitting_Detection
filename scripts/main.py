import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os 
from utils import * 
import torchvision.transforms as transforms
import time 


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
RAW = os.path.join(BASE_DIR, "images", "cache", "raw.png")
HEATMAP = os.path.join(BASE_DIR, "images", "cache", "heatmap.png")


# Streamlit app
def main():

    
    EMAIL_COOLDOWN = 300
    last_email_time = 0

    model = load_model("medium")

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

    with st.sidebar: 

        user = st.text_input("E-Mail Adresse", help="E-Mail Adresse für automatische Bewarnung")
        run = st.toggle("Start Video Stream")

    visualizer = st.pills("Visualisiertung", ["Bounding-Box", "Heatmap"], selection_mode="single", default="Bounding-Box")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        FRAME_WINDOW = st.image([], use_container_width=True)
        prediction_text = st.empty()  # nur einmal vor der Schleife

    cap = cv2.VideoCapture(0)  # einmal vor der Schleife öffnen

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        img_pil = Image.fromarray(frame)
        img_tensor = transform(img_pil)
        overlay, prediction = get_heatmap_overlay(model, img_tensor)

        if prediction == 1:
            label = "Pitting erkannt!"
            current_time = time.time()
            if current_time - last_email_time > EMAIL_COOLDOWN:
                cv2.imwrite(RAW, frame)
                cv2.imwrite(HEATMAP, overlay)
                send_warning(2)
                last_email_time = current_time

        else:
            label = "Kein Pitting erkannt"

        frame = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        #frame = cv2.resize(frame, (640, 480))

        with col2:
            FRAME_WINDOW.image(frame)
            prediction_text.markdown(
                f"<h4 style='text-align:center; color:{'red' if prediction == 1 else 'green'}'>{label}</h4>",
                unsafe_allow_html=True
            )

    cap.release() 


if __name__ == '__main__':
    main()

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os 

BASE = os.getcwd()

#load best model 
def load_model():
    import joblib
    model = joblib.load("your_model.pkl")
    return model

# dummy 
def predict_pitting(model, frame):
    return np.random.choice(["pitting", "no_pitting"])

#box 
def draw_prediction_box(frame, prediction):
    height, width, _ = frame.shape
    color = (0, 255, 0) if prediction == "no_pitting" else (255, 0, 0)
    thickness = 4
    box_start = (50, 50)
    box_end = (width - 50, height - 50)
    cv2.rectangle(frame, box_start, box_end, color, thickness)
    cv2.putText(frame, prediction.upper(), (box_start[0], box_start[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame

# Streamlit app
def main():

    st.markdown("""
    <style>
    #stDecoration {
        background-image: linear-gradient(90deg, #d6d6d6, #26ab71) !important;
        height: 2px !important; 
    }
    </style>
    """, unsafe_allow_html=True)

    logo = r"/Users/janikwahrheit/Library/CloudStorage/OneDrive-Persönlich/01_Studium/01_Bachelor/06. SS_2025/Pitting_Detection/images/wbk.png"


    st.title("🔩Pitting Detection")
    st.logo(logo, icon_image=logo, size="large")

    #model = load_model()‚

    with st.sidebar: 

        user = st.text_input("E-Mail Adresse", help="E-Mail Adresse für automatische Bewarnung")
        run = st.toggle("Start Video Stream")

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)  
    while run:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        prediction = predict_pitting(None, frame)
        frame = draw_prediction_box(frame, prediction)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    cap.release()

if __name__ == '__main__':
    main()

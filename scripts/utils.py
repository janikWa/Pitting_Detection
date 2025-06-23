import numpy as np
import cv2


#load best model 
def load_model():
    import joblib
    model = joblib.load("model.pkl")
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
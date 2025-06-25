import numpy as np
import cv2
import smtplib
from email.message import EmailMessage
from datetime import datetime
from pathlib import Path
import mimetypes
import os
import matplotlib.pyplot as plt 
import numpy as np
import os
import cv2
import torch
import matplotlib.pyplot as plt 
from torchvision import models, transforms


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
RAW = os.path.join(BASE_DIR, "images", "cache", "raw.png")
HEATMAP = os.path.join(BASE_DIR, "images", "cache", "heatmap.png")
LOGO = os.path.join(BASE_DIR, "images", "wbk.png")


#load best model 
def load_model(transformation: str = None): 
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2 classes
    filename = f"resnet18_pitting_not_simplified_{transformation}.pth" if transformation else "resnet18_pitting_not_simplified.pth"
    load_path = os.path.join(BASE_DIR, "models", filename)
    model.load_state_dict(torch.load(load_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) 
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

def send_warning(machine):

    msg = EmailMessage()
    msg["Subject"] = "🚨 WARNUNG: Pitting festgestellt"
    msg["From"] = "warnung@wbk.de"
    msg["To"] = "leiter@produktion.de"

    now = datetime.now()
    date_str = now.strftime("%d.%m.%Y") 
    time_str = now.strftime("%H:%M")

    msg.set_content(
        f"Es wurde ein Defekt an Maschine {machine} am {date_str} um {time_str} festgestellt."
    )

    msg.add_alternative(f"""\
    <html>
      <body style="font-family: Arial, sans-serif; background-color: #f5f5f5; padding: 20px;">
        <div style="background-color: #fff; padding: 20px; border-radius: 8px; border: 1px solid #ddd;">
          <img src="cid:logo" style="height: 50px;" alt="WBK Logo" />
          <h2 style="color: #c62828;">⚠️ Pitting-Alarm</h2>
          <p>Es wurde ein <strong>Defekt an Maschine {machine}</strong> festgestellt.</p>
          <p><strong>Datum:</strong> {date_str}<br>
             <strong>Uhrzeit:</strong> {time_str}</p>

          <h3>Bilder des Defekts:</h3>
          <p><strong>Originalaufnahme:</strong><br>
             <img src="cid:rawimg" style="max-width: 100%; border: 1px solid #ccc;"/></p>
          <p><strong>Heatmap:</strong><br>
             <img src="cid:heatmap" style="max-width: 100%; border: 1px solid #ccc;"/></p>
        </div>
      </body>
    </html>
    """, subtype='html')

    def attach_image(cid_name, file_path):
        path = Path(file_path)
        if path.exists():
            mime_type, _ = mimetypes.guess_type(path)
            maintype, subtype = mime_type.split("/")
            with open(path, "rb") as img:
                msg.get_payload()[1].add_related(
                    img.read(), maintype=maintype, subtype=subtype, cid=cid_name
                )

    # Logo
    attach_image("logo", LOGO)
    
    # Raw-Bild
    attach_image("rawimg", RAW)

    # Heatmap
    attach_image("heatmap", HEATMAP)

    logo_path = Path("/Users/janikwahrheit/Library/CloudStorage/OneDrive-Persönlich/01_Studium/01_Bachelor/06. SS_2025/Pitting_Detection/images/wbk.png")
    maintype, subtype = mimetypes.guess_type(logo_path)[0].split("/")

    with open(logo_path, "rb") as img:
        msg.get_payload()[1].add_related(img.read(), maintype=maintype, subtype=subtype, cid="logo")

    with smtplib.SMTP("sandbox.smtp.mailtrap.io", 2525) as server:
        server.starttls()
        server.login("8f10d22cc724e5", "9c84a2795ef13a")
        server.send_message(msg)

    print("Mail gesendet")
    print()


def get_heatmap_overlay(model, img_tensor):
    gradients = []

    if img_tensor.shape != (1, 3, 224, 224):
        if img_tensor.shape == (3, 224, 224):
            image_tensor_batched = img_tensor.unsqueeze(0)
        else:
            raise ValueError(f"Unerwartete Bildgröße: {img_tensor.shape}")
    else:
        image_tensor_batched = img_tensor

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    unnormalized_image = img_tensor * std[:, None, None] + mean[:, None, None]
    image_np = unnormalized_image.permute(1, 2, 0).numpy()

    def save_gradient(grad):
        gradients.append(grad)

    activations = []
    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    hook = model.layer4.register_forward_hook(forward_hook)

    output = model(image_tensor_batched)
    pred_class = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, pred_class].backward()

    grads_val = gradients[0][0].cpu().detach().numpy()
    acts_val = activations[0][0].cpu().detach().numpy()

    weights = np.mean(grads_val, axis=(1, 2))
    cam = np.zeros(acts_val.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts_val[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cam - cam.min()
    cam = cam / cam.max()
    cam = cv2.resize(cam, (224, 224))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    image_uint8 = (image_np * 255).astype(np.uint8)
    overlay = heatmap * 0.4 + image_uint8 * 0.6
    overlay = np.uint8(overlay)

    hook.remove()
    return overlay, pred_class


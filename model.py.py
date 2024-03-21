import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import cv2
import numpy as np
from PIL import Image
import zipfile

with zipfile.ZipFile("examples.zip", "r") as zip_ref:
    zip_ref.extractall(".")

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).eval()

model = InceptionResnetV1(
    pretrained="vggface2",
    classify=True,
    num_classes=1,
    device=DEVICE
)

checkpoint = torch.load("resnetinceptionv1_epoch_32.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

def predict(input_image_path):
    """Predict whether the image contains a real or fake face"""
    image = cv2.imread(input_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    image = Image.fromarray(image)
    
    face = mtcnn(image)
    if face is None:
        return "No face detected"
    
    face = face.unsqueeze(0)  # add the batch dimension
    face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)
    face = face.to(DEVICE, dtype=torch.float32) / 255.0
    
    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))
        prediction = "Real" if output.item() < 0.5 else "Fake"

    return prediction

# Example usage:
input_image_path = "realimg.jpg"
prediction_result = predict(input_image_path)
print("Prediction:", prediction_result)

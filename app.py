from flask import Flask, request, render_template
from ultralytics import YOLO
import torch
import torch.nn as nn
from PIL import Image
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO
yolo_model = YOLO("best.pt")

# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,16,3,2,1), nn.ReLU(),
            nn.Conv2d(16,32,3,2,1), nn.ReLU(),
            nn.Conv2d(32,64,3,2,1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64,32,3,2,1,1), nn.ReLU(),
            nn.ConvTranspose2d(32,16,3,2,1,1), nn.ReLU(),
            nn.ConvTranspose2d(16,3,3,2,1,1), nn.Sigmoid()
        )

    def forward(self,x):
        return self.decoder(self.encoder(x))

model_ae = Autoencoder()
model_ae.load_state_dict(torch.load("autoencoder.pth", map_location="cpu"))
model_ae.eval()

def predict(image_path):
    results = yolo_model.predict(image_path, conf=0.25)
    
    if len(results[0].boxes) > 0:
        return "BAD WELD", "Defect detected"
    else:
        return "GOOD WELD", "No defect detected"

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        result, explanation = predict(path)

        return f"""
        <h2>{result}</h2>
        <p>{explanation}</p>
        <img src="/{path}" width="300">
        <br><a href="/">Upload Another</a>
        """

    return """
    <h2>Upload Weld Image</h2>
    <form method="POST" enctype="multipart/form-data">
    <input type="file" name="file" accept="image/*" capture="camera">
    <button type="submit">Check Weld</button>
    </form>
    """

if __name__ == "__main__":
    app.run()
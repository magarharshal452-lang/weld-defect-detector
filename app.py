import streamlit as st
from PIL import Image
import torch
from ultralytics import YOLO
import numpy as np

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Weld Defect Detection", layout="centered")

st.title("🔍 Weld Defect Detection System")

# -------------------- LOAD MODELS --------------------

@st.cache_resource
def load_models():
    # Load YOLO
    yolo_model = YOLO("best.pt")
    yolo_model.to("cpu")

    # Autoencoder
    class Autoencoder(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(3,16,3,2,1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16,32,3,2,1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32,64,3,2,1),
                torch.nn.ReLU()
            )

            self.decoder = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(64,32,3,2,1,1),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(32,16,3,2,1,1),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(16,3,3,2,1,1),
                torch.nn.Sigmoid()
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    autoencoder = Autoencoder()

    state_dict = torch.load("autoencoder.pth", map_location="cpu", weights_only=False)
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()

    return yolo_model, autoencoder


yolo_model, autoencoder = load_models()

# -------------------- UI --------------------

uploaded_file = st.file_uploader("Upload Weld Image", type=["jpg", "png", "jpeg"])

# -------------------- MAIN LOGIC --------------------

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ---------- YOLO DETECTION ----------
    results = yolo_model.predict(image, imgsz=256, conf=0.3)
    boxes = results[0].boxes

    detected_classes = []

    if boxes is not None and len(boxes) > 0:
        labels = results[0].names
        detected_classes = [labels[int(cls)] for cls in boxes.cls]

        st.image(results[0].plot(), caption="Detection Result")

    # ---------- AUTOENCODER ----------
    img = image.resize((128,128))
    img = np.array(img) / 255.0
    img = torch.tensor(img).permute(2,0,1).unsqueeze(0).float()

    with torch.no_grad():
        recon = autoencoder(img)

    loss = torch.mean((img - recon)**2).item()

    # ---------- FINAL DECISION ----------
    defect_keywords = ["defect", "bad", "crack", "porosity"]

    is_defect_yolo = any(
        any(keyword in cls.lower() for keyword in defect_keywords)
        for cls in detected_classes
    )

    if is_defect_yolo or loss > 0.006:
        final_result = "❌ DEFECTIVE WELD"
    else:
        final_result = "✅ GOOD WELD"

    # ---------- OUTPUT ----------
    st.subheader("Result")
    st.success(final_result)

    st.subheader("Anomaly Score")
    st.write(f"{loss:.6f}")

    st.subheader("Explanation")

    if is_defect_yolo:
        st.write("Detected using YOLO object detection.")
    elif loss > 0.006:
        st.write("Detected using anomaly detection (autoencoder).")
    else:
        st.write("No defect detected.")

    # Debug
    with st.expander("Debug Info"):
        st.write("Detected Classes:", detected_classes)

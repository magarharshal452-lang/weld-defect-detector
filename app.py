import streamlit as st
from PIL import Image
import torch
from ultralytics import YOLO
import numpy as np

# Load YOLO
yolo_model = YOLO("best.pt")

# ... (rest of your Autoencoder class) ...

autoencoder = Autoencoder()
autoencoder.load_state_dict(torch.load("autoencoder.pth", map_location="cpu"))
autoencoder.eval()

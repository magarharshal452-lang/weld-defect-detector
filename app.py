import streamlit as st
from PIL import Image
import torch
from ultralytics import YOLO
import numpy as np

# Load YOLO
yolo_model = YOLO("best.pt")

# ✅ ADD THIS FULL CLASS (DO NOT REMOVE)

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

    def forward(self,x):
        return self.decoder(self.encoder(x))

# --------------------

autoencoder = Autoencoder()
autoencoder.load_state_dict(torch.load("autoencoder.pth", map_location="cpu"))
autoencoder.eval()

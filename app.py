# Ensure these match your uploaded file names exactly
yolo_model = YOLO("best.pt") 

# And for the autoencoder:
autoencoder.load_state_dict(torch.load("autoencoder.pth", map_location="cpu"))

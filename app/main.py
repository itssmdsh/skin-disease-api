import os
import gdown
from fastapi import FastAPI, UploadFile
import torch
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO

# Initialize FastAPI app
app = FastAPI()

# Path where the model will be saved
model_path = "app/model.pth"
model_url = "https://www.dropbox.com/scl/fi/mu7vcde9i971765otbv9y/model.pth?rlkey=aknqcedutttfj37n5q35kj3eg&st=ys7g0nvb&dl=1"  # Dropbox direct link

# Check if the model exists, otherwise download it
if not os.path.exists(model_path):
    print("Downloading model from Dropbox...")
    gdown.download(model_url, model_path, quiet=False)
    print("Model downloaded.")

# Load the PyTorch model
model = torch.load(model_path)
model.eval()

# Define image transform (resize and normalize to match the model's requirements)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224 (change if needed)
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization (adjust if needed)
])

# Prediction endpoint
@app.post("/predict/")
async def predict(image: UploadFile):
    image_data = await image.read()
    img = Image.open(BytesIO(image_data))  # Open the image from bytes
    img = transform(img).unsqueeze(0)  # Add batch dimension for the model

    # Make prediction
    with torch.no_grad():
        outputs = model(img)  # Get model outputs
        _, predicted = torch.max(outputs, 1)  # Get the class with the highest score

    return {"prediction": predicted.item()}  # Return the predicted class index

import os
import torch
from fastapi import FastAPI, File, UploadFile
from torchvision import models, transforms
from PIL import Image
import requests
import torch.nn.functional as F

app = FastAPI()

# Define the download function
def download_model_from_dropbox(url: str, save_path: str):
    print("Downloading model from Dropbox...")
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print("Model downloaded.")

# Load the model directly as ResNet18 (no wrapper)
def load_model(model_path: str):
    num_classes = 6
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# Define the image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Model config
model_url = "https://www.dropbox.com/scl/fi/mu7vcde9i971765otbv9y/model.pth?rlkey=aknqcedutttfj37n5q35kj3eg&st=ys7g0nvb&dl=1"
model_path = "model.pth"

# Download and load the model
if not os.path.exists(model_path):
    download_model_from_dropbox(model_url, model_path)

model = load_model(model_path)

# Class names in order
class_names = ['Acne', 'Eczema', 'Psoriasis', 'Warts', 'SkinCancer', 'Unknown_Normal']

# Prediction endpoint with confidence percentages for each class
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Open the image
    image = Image.open(file.file).convert("RGB")
    image = transform(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        
        # Apply softmax to get probabilities for each class
        probabilities = F.softmax(outputs, dim=1)
        
        # Get the class with the highest probability
        _, predicted = torch.max(probabilities, 1)
        
        # Prepare the prediction
        prediction = class_names[predicted.item()]
        
        # Convert probabilities to a dictionary for each class with its percentage
        class_probabilities = {class_names[i]: round(probabilities[0][i].item() * 100, 2) for i in range(len(class_names))}
        
    return {"prediction": prediction, "confidence_percentages": class_probabilities}

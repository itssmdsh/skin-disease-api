import os
import torch
from fastapi import FastAPI, File, UploadFile
from torchvision import models, transforms
from PIL import Image
import requests

# Create FastAPI app
app = FastAPI()

# Define the download function
def download_model_from_dropbox(url: str, save_path: str):
    # Download the model from Dropbox
    print("Downloading model from Dropbox...")
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print("Model downloaded.")

# Define model class (custom SkinDiseaseModel)
class SkinDiseaseModel(torch.nn.Module):
    def __init__(self, num_classes=6):
        super(SkinDiseaseModel, self).__init__()
        self.resnet = models.resnet18(pretrained=False)  # Don't load pretrained weights here
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, num_classes)  # Replace final layer

    def forward(self, x):
        return self.resnet(x)

# Load the model function
def load_model(model_path: str):
    # Initialize model
    model = SkinDiseaseModel(num_classes=6)

    # Load the state_dict of the pretrained ResNet18 model
    resnet18 = models.resnet18(pretrained=True)
    pretrained_dict = resnet18.state_dict()

    # Get the model state_dict
    model_dict = model.state_dict()

    # Filter out the layers we don't want to load from pretrained model (excluding 'fc' layer)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k != 'fc.weight' and k != 'fc.bias'}

    # Update the model state dict with the pretrained weights
    model_dict.update(pretrained_dict)

    # Load the state dict into the model
    model.load_state_dict(model_dict)

    # Load the model weights saved in model.pth
    model.load_state_dict(torch.load(model_path))
    
    # Set the model to evaluation mode
    model.eval()

    return model

# Define the image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the model path and Dropbox URL
model_url = "https://www.dropbox.com/scl/fi/mu7vcde9i971765otbv9y/model.pth?rlkey=aknqcedutttfj37n5q35kj3eg&st=ys7g0nvb&dl=1"
model_path = "model.pth"

# Download the model if not already downloaded
if not os.path.exists(model_path):
    download_model_from_dropbox(model_url, model_path)

# Load the model
model = load_model(model_path)

# Define the prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Open the uploaded image
    image = Image.open(file.file)

    # Preprocess the image
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # Map the predicted label to the corresponding skin disease
    class_names = ['Acne', 'Eczema', 'Psoriasis', 'Warts', 'SkinCancer', 'Unknown_Normal']
    prediction = class_names[predicted.item()]

    return {"prediction": prediction}

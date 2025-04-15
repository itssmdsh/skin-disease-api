import os
import gdown
from fastapi import FastAPI, UploadFile, File
import torch
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO

app = FastAPI()

# Google Drive model ID and local path
MODEL_ID = "1Q5sJGHHq1x-LWM0wr7Bj9prny196XoH9"
MODEL_PATH = "model.pth"

# Download model from Google Drive if it doesn't exist
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        gdown.download(id=MODEL_ID, output=MODEL_PATH, quiet=False)
        print("Model downloaded.")

download_model()

# Define your actual model class below
# Replace this dummy with your real architecture
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = torch.nn.Linear(512, 10)  # Example layer

    def forward(self, x):
        return self.fc(x)

# Load model
model = MyModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust to your model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

# Prediction endpoint
@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    image_data = await image.read()
    img = Image.open(BytesIO(image_data)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        _, prediction = torch.max(output, 1)

    return {"prediction": prediction.item()}

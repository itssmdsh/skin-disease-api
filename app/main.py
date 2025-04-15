import os
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import gdown

# Initialize FastAPI app
app = FastAPI()

# Model config
MODEL_PATH = "app/model.pth"
MODEL_URL = "https://www.dropbox.com/scl/fi/mu7vcde9i971765otbv9y/model.pth?rlkey=aknqcedutttfj37n5q35kj3eg&st=ys7g0nvb&dl=1"
CLASS_NAMES = ['Acne', 'Eczema', 'Psoriasis', 'Warts', 'SkinCancer', 'Unknown_Normal']

# Download model if not present
os.makedirs("app", exist_ok=True)
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Dropbox...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    print("Model downloaded.")

# Define model architecture
class SkinDiseaseModel(nn.Module):
    def __init__(self):
        super(SkinDiseaseModel, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(CLASS_NAMES))

    def forward(self, x):
        return self.model(x)

# Load model
model = SkinDiseaseModel()
state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
model.load_state_dict(state_dict)
model.eval()

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.get("/")
def read_root():
    return {"message": "Skin Disease Classifier is up and running 🚀"}

@app.post("/predict/")
async def predict(image: UploadFile):
    try:
        contents = await image.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = CLASS_NAMES[predicted.item()]

    return JSONResponse(content={"prediction": predicted_class})

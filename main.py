from fastapi import FastAPI, Query
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
import io

app = FastAPI()

# Load MobileNetV2 model
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()  # Set to evaluation mode

# ImageNet animal class range (0-999 labels)
ANIMAL_CLASSES = set(range(151, 296))  # Includes most mammals, birds, fish, etc.

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for MobileNetV2
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def is_animal(image_url: str) -> bool:
    try:
        # Download the image
        response = requests.get(image_url, timeout=5)
        response.raise_for_status()  # Raise error for invalid responses
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception:
        return False  # If the image cannot be loaded, return False

    # Preprocess image
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Predict class
    with torch.no_grad():
        output = model(image)
        predicted_class = output.argmax().item()

    # Check if the predicted class is an animal
    return predicted_class in ANIMAL_CLASSES

@app.get("/validate-pet-image/")
async def validate_pet_image(image_url: str = Query(..., title="Image URL")):
    result = is_animal(image_url)
    return {"is_animal": result}

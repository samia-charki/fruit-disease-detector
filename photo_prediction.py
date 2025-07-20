import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os

# 1. تحميل أسماء الكلاسات
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

# 2. إعداد ResNet18
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_indices))
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model = model.to(device)
model.eval()

# 3. نفس الترانسفورم اللي درناه فالتدريب
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 4. دالة التنبؤ بصورة وحدة
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    predicted_class = idx_to_class[predicted.item()]
    return predicted_class, confidence.item() * 100

# 5. تجربة على جميع الصور داخل مجلد
def show_prediction(image_path):
    try:
        predicted_class, confidence = predict_image(image_path)
        print(f"📷 {os.path.basename(image_path)} → {predicted_class} ({confidence:.2f}%)")
    except Exception as e:
        print(f"❌ Error: {e}")

# 🔄 قراءة جميع الصور من مجلد "photos"
folder_path = "photos"
image_extensions = [".jpg", ".jpeg", ".png"]

image_paths = [
    os.path.join(folder_path, filename)
    for filename in os.listdir(folder_path)
    if any(filename.lower().endswith(ext) for ext in image_extensions)
]

# 🔍 عرض التنبؤات
for path in image_paths:
    show_prediction(path)

show_prediction("spliit_dataset/train/Banana__Healthy/Banana__Healthy_augmented_3.jpg")


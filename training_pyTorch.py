import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import numpy as np

print("✅ OK - Start...")

# 1. التحويلات
train_transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 2. تحميل البيانات
train_dataset = datasets.ImageFolder('spliit_dataset/train', transform=train_transform)
val_dataset = datasets.ImageFolder('spliit_dataset/val', transform=val_transform)
test_dataset = datasets.ImageFolder('spliit_dataset/test_final', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. حفظ أسماء الكلاسات
class_indices = train_dataset.class_to_idx
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f)

# 4. إعداد الموديل
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_indices))
model = model.to(device)

# 5. حساب class weights
labels = [label for _, label in train_dataset.samples]
class_counts = Counter(labels)
num_samples = sum(class_counts.values())
class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]
class_weights = torch.FloatTensor(class_weights).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("🧠 Using device:", device)  # cuda or cpu

# 6. تدريب الموديل
num_epochs = 4
patience = 2
best_val_loss = float('inf')
counter = 0
early_stop = False

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

print("🚀 Start training loop...")

for epoch in range(num_epochs):
    if early_stop:
        print(f"🛑 Early Stop at epoch {epoch}")
        break

    print(f"🟢 Epoch {epoch+1}/{num_epochs} — Training...")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)

    print("🔍 Validation...")
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    print(f"📊 Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(f"✅ Accuracy — Train: {train_accuracy:.2f}% | Validation: {val_accuracy:.2f}%")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print("💾 Best model saved!")
    else:
        counter += 1
        if counter >= patience:
            early_stop = True

# 7. الحفظ النهائي
torch.save(model.state_dict(), 'last.pth')
print("✅ Training Done!")

# 8. رسم منحنيات الخسارة والدقة
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Curve')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("loss_accuracy_curves.png")
plt.show()

# 9. تقييم نهائي على test_final
print("\n📌 Final Evaluation on Test Set:")

y_true = []
y_pred = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_indices.keys()))

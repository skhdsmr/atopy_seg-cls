import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
import timm

# ============================================================
# 0. Create save folder (local)
# ============================================================
SAVE_DIR = './SkinDisNet_Models'
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"📍 가중치 저장 경로: {os.path.abspath(SAVE_DIR)}")

# ============================================================
# 1. Hyperparameter configuration
# ============================================================
DATA_DIR      = './SkinDisNet_balanced'
NUM_CLASSES   = 5
BATCH_SIZE    = 8
ACCUM_STEPS   = 4         # effective batch size = 8 × 4 = 32
NUM_EPOCHS    = 150
LEARNING_RATE = 1e-4
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"사용 디바이스 : {DEVICE}")

# ============================================================
# 2. Transform definition (keep 512×512 original resolution)
# ============================================================
base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================================================
# 3. Dataset and DataLoader
# ============================================================
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=base_transform)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'),   transform=base_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"클래스 목록             : {train_dataset.classes}")
print(f"훈련 데이터셋(Train) 수 : {len(train_dataset)}장")
print(f"검증 데이터셋(Valid) 수 : {len(val_dataset)}장")
print(f"Batch size             : {BATCH_SIZE}")
print(f"Accum steps            : {ACCUM_STEPS}")
print(f"유효 Batch size         : {BATCH_SIZE * ACCUM_STEPS}")

# ============================================================
# 4. Model definition
# ============================================================
print("\nMobileNetV4-Medium pretrained 모델 로드 중...")
model = timm.create_model('mobilenetv4_conv_medium', pretrained=True, num_classes=NUM_CLASSES)

# Light freezing strategy to prevent overfitting
for param in model.parameters():
    param.requires_grad = False

for param in model.conv_head.parameters():
    param.requires_grad = True
for param in model.classifier.parameters():
    param.requires_grad = True

model = model.to(DEVICE)
print(f"출력 클래스 수 : {NUM_CLASSES} → {train_dataset.classes}")

# ============================================================
# 5. Training configuration
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
scaler    = GradScaler('cuda')

# ============================================================
# 6. Training / validation functions
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, accum_steps):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad()

    for step, (images, labels) in enumerate(tqdm(loader, desc='  Train', leave=False)):
        images, labels = images.to(device), labels.to(device)

        with autocast('cuda'):
            outputs = model(images)
            loss    = criterion(outputs, labels) / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accum_steps * images.size(0)
        correct      += (outputs.argmax(dim=1) == labels).sum().item()
        total        += labels.size(0)

    if (step + 1) % accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='  Valid', leave=False):
            images, labels = images.to(device), labels.to(device)

            with autocast('cuda'):
                outputs = model(images)
                loss    = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)

            running_loss += loss.item() * images.size(0)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / total
    val_acc  = correct / total
    val_f1   = f1_score(all_labels, all_preds, average='macro')

    return val_loss, val_acc, val_f1

# ============================================================
# 7. Training loop and weight saving
# ============================================================
best_val_acc = 0.0
best_val_f1  = 0.0

acc_path = os.path.join(SAVE_DIR, 'best_mobilenetv4_acc.pth')
f1_path  = os.path.join(SAVE_DIR, 'best_mobilenetv4_f1.pth')

history = {'train_loss': [], 'train_acc': [],
           'val_loss':   [], 'val_acc':   [], 'val_f1': []}

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\nEpoch [{epoch:02d}/{NUM_EPOCHS}]  lr={scheduler.get_last_lr()[0]:.2e}")

    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, scaler, DEVICE, ACCUM_STEPS
    )
    val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, DEVICE)

    scheduler.step()

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)

    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"  Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc*100:.2f}% | Valid F1: {val_f1:.4f}")

    # Save based on best val_acc
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), acc_path)
        print(f"  ✅ Best Acc 모델 저장 완료 (val_acc: {best_val_acc*100:.2f}%)")

    # Save based on best val_f1
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), f1_path)
        print(f"  ✅ Best F1  모델 저장 완료 (val_f1: {best_val_f1:.4f})")

print(f"\n✅ 전체 학습 완료")
print(f"  ⭐ 최종 저장 위치: {os.path.abspath(SAVE_DIR)}")
print(f"  ➡️ Best Val Acc 가중치 파일: {acc_path}")
print(f"  ➡️ Best Val F1  가중치 파일: {f1_path}")

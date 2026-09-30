import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.amp import autocast
from tqdm import tqdm
import timm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
)

# ============================================================
# 1. Configuration (matched to the training code)
# ============================================================
DATA_DIR    = '../SkinDisNet_balanced'
WEIGHT_PATH = './best_mobilenetv4.pth'     # weight file to evaluate
NUM_CLASSES = 5
BATCH_SIZE  = 8
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"사용 디바이스 : {DEVICE}")
print(f"가중치 파일   : {WEIGHT_PATH}")

# ============================================================
# 2. Transform (same as training — keep 512x512 original resolution)
# ============================================================
base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================================================
# 3. Test dataset / DataLoader
# ============================================================
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'),
                                     transform=base_transform)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

class_names = test_dataset.classes
print(f"클래스 목록           : {class_names}")
print(f"테스트 데이터셋(Test) 수 : {len(test_dataset)}장")

# ============================================================
# 4. Model definition and weight loading
# ============================================================
print("\nMobileNetV4-Medium 모델 로드 중...")
model = timm.create_model('mobilenetv4_conv_medium', pretrained=False,
                          num_classes=NUM_CLASSES)

state_dict = torch.load(WEIGHT_PATH, map_location=DEVICE)
# Handle the case where it was saved as a wrapper dict (containing a 'state_dict' key)
if isinstance(state_dict, dict) and 'state_dict' in state_dict:
    state_dict = state_dict['state_dict']
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()
print("가중치 로드 완료")

# ============================================================
# 5. Evaluation
# ============================================================
criterion = nn.CrossEntropyLoss()
running_loss, total = 0.0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc='Test'):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        if DEVICE.type == 'cuda':
            with autocast('cuda'):
                outputs = model(images)
                loss    = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss    = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        total        += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

test_loss = running_loss / total
test_acc  = accuracy_score(all_labels, all_preds)
test_f1   = f1_score(all_labels, all_preds, average='macro')

# ============================================================
# 6. Print results
# ============================================================
print("\n" + "=" * 50)
print("테스트 결과")
print("=" * 50)
print(f"Test Loss      : {test_loss:.4f}")
print(f"Test Accuracy  : {test_acc * 100:.2f}%")
print(f"Test Macro F1  : {test_f1:.4f}")

print("\n[ 클래스별 리포트 ]")
cm = confusion_matrix(all_labels, all_preds)
per_class_recall = recall_score(all_labels, all_preds, average=None)
per_class_f1     = f1_score(all_labels, all_preds, average=None)
print(f"{'class':>10}  {'recall':>9}  {'f1-score':>9}")
for i, c in enumerate(class_names):
    print(f"{c:>10}  {per_class_recall[i] * 100:>8.2f}%  {per_class_f1[i]:>9.4f}")

print("\n[ Confusion Matrix ]  (행=실제, 열=예측)")
header = "actual\\pred  " + "  ".join(f"{c:>7}" for c in class_names)
print(header)
for i, row in enumerate(cm):
    print(f"{class_names[i]:>10}  " + "  ".join(f"{v:>7d}" for v in row))
print("\n\n\n\n\n\n\n\n\n")

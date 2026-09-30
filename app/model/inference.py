import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import timm

# ============================================================
# 1. Configuration
# ============================================================
WEIGHT_PATH = './best_mobilenetv4.pth'
NUM_CLASSES = 5
# Class order as sorted alphabetically by ImageFolder (must match training)
CLASS_NAMES = ['CD', 'EC', 'OTHERS', 'SC', 'TC']
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Normalization values (same as training)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]


# ============================================================
# 2. Grad-CAM implementation (using forward / backward hooks)
# ============================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        # Save the target layer's output (activations) and the gradient w.r.t. that output
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)               # [1, num_classes]
        probs  = F.softmax(output, dim=1)

        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        # Backpropagate w.r.t. the target class score
        score = output[0, class_idx]
        score.backward()

        # Per-channel weights = spatial average (GAP) of the gradient
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)

        # Upsample to the input image size
        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                            mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to 0~1 (clip at the top 1% to emphasize lesion-area contrast)
        cam -= cam.min()
        vmax = np.percentile(cam, 99)
        if vmax > 0:
            cam = np.clip(cam / vmax, 0, 1)

        return cam, class_idx, probs.detach().cpu().numpy()[0]


# ============================================================
# 3. Model loading
# ============================================================
def load_model():
    model = timm.create_model('mobilenetv4_conv_medium', pretrained=False,
                              num_classes=NUM_CLASSES)
    state_dict = torch.load(WEIGHT_PATH, map_location=DEVICE)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model


# ============================================================
# 4. Preprocessing (same as training — keep original resolution, no resize)
# ============================================================
def preprocess(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
    tensor = transform(img).unsqueeze(0).to(DEVICE)   # [1, 3, H, W]
    return img, tensor


# ============================================================
# 5. Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='MobileNetV4 단일 이미지 추론 + Grad-CAM')
    parser.add_argument('image', help='추론할 이미지 파일 경로')
    parser.add_argument('--out', default=None, help='시각화 결과 저장 경로 (기본: <이미지명>_gradcam.png)')
    parser.add_argument('--alpha', type=float, default=0.5, help='히트맵 오버레이 투명도 (0~1)')
    parser.add_argument('--thr', type=float, default=0.2,
                        help='이 값 미만의 활성도 영역은 히트맵 없이 원본 유지 (0~1)')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"[오류] 이미지 파일을 찾을 수 없습니다: {args.image}")
        sys.exit(1)

    print(f"사용 디바이스 : {DEVICE}")
    model = load_model()
    print("모델 로드 완료")

    img_pil, input_tensor = preprocess(args.image)

    # Target layer: the last block before pooling, where spatial resolution remains
    # (conv_head is applied 'after' global pooling, so it is 1x1 with no spatial info -> becomes uniform everywhere)
    target_layer = model.blocks[-1]

    gradcam = GradCAM(model, target_layer)
    cam, pred_idx, probs = gradcam(input_tensor)

    pred_name = CLASS_NAMES[pred_idx]
    print("\n" + "=" * 40)
    print(f"예측 클래스 : {pred_name}  (확률 {probs[pred_idx] * 100:.2f}%)")
    print("-" * 40)
    print("클래스별 확률:")
    for i, name in enumerate(CLASS_NAMES):
        bar = '#' * int(probs[i] * 30)
        print(f"  {name:>7} : {probs[i] * 100:6.2f}%  {bar}")
    print("=" * 40)

    # ---------- Visualization ----------
    img_np = np.array(img_pil) / 255.0         # [H, W, 3], 0~1
    heatmap = cm.jet(cam)[..., :3]             # apply colormap [H, W, 3], 0~1

    # Vary blend strength by activation (cam).
    #  - below thr (faint blue area) -> alpha 0 -> original kept as-is
    #  - at or above thr (green/yellow/red) -> blend heatmap proportional to cam value
    cam_w = np.clip((cam - args.thr) / (1.0 - args.thr), 0, 1)   # [H, W], 0~1
    alpha_map = (args.alpha * cam_w)[..., None]                  # [H, W, 1]
    overlay = (1 - alpha_map) * img_np + alpha_map * heatmap
    overlay = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_np)
    axes[0].set_title('Original')
    axes[1].imshow(overlay)
    axes[1].set_title(f'Overlay\nPred: {pred_name} ({probs[pred_idx]*100:.1f}%)')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()

    out_path = args.out
    if out_path is None:
        base = os.path.splitext(os.path.basename(args.image))[0]
        out_path = f'{base}_gradcam.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n시각화 결과 저장: {os.path.abspath(out_path)}")
    plt.show()


if __name__ == '__main__':
    main()

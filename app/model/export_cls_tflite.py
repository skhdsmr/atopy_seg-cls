"""분류 체크포인트(.pt) -> 온디바이스용 TFLite 변환 (app 폴더 자립형).

경로:  PyTorch(MultiTaskNet) -> ONNX(고정 1x3xNxN, NCHW, 단일 concat 출력)
       -> onnx2tf -> TFLite(NHWC 입력, [1,H,W,3] float32).

세그멘테이션용 export_tflite.py 와 짝을 이루는 '분류' 버전이다.
  - 입력 체크포인트: app/model/checkpoint/classification/*.pt
  - 출력 tflite:     app/dermascan-app/assets/models/
  - 앱 레지스트리:   app/dermascan-app/src/ml/models.cls.gen.js  (분류 전용, 세그와 분리)

멀티태스크(공유 인코더 + 5개 순서형 등급 head: severity(IGA)/erythema/papulation/
excoriation/lichenification)를 고정 순서로 이어붙여 단일 출력 [1, sum(sizes)] 로 낸다.
앱은 meta.tasks 의 offset/size 로 잘라 태스크별 softmax -> 등급을 얻는다.

전처리 규약(학습/서버와 동일): Resize((imgsz,imgsz)) -> ImageNet 정규화.

준비(변환 툴체인, 최초 1회): pip install -r app/requirements-convert.txt

사용:
  # 폴더 내 모든 분류 .pt 를 일괄 변환
  python app/model/export_cls_tflite.py --batch
  # 단일 변환
  python app/model/export_cls_tflite.py --ckpt app/model/checkpoint/classification/mnv3s_224_0.2.pt
"""
import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from cls_arch import TASKS, TASK_NAMES, TASK_TITLE, MultiTaskNet, ConcatHead

# app/model/export_cls_tflite.py -> APP_DIR = app/
APP_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = APP_DIR / "model"
DEFAULT_CKPT_DIR = MODEL_DIR / "checkpoint" / "classification"
DEFAULT_OUT = APP_DIR / "dermascan-app" / "assets" / "models"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ("model", "state_dict", "model_state_dict", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _head_sizes(sd):
    """state_dict 의 heads.<task>.weight 로부터 {task: n_classes} 추출."""
    sizes = {}
    for k, v in sd.items():
        if k.startswith("heads.") and k.endswith(".weight"):
            sizes[k.split(".")[1]] = int(v.shape[0])
    return sizes


def _tasks_meta(head_sizes):
    """고정 순서(TASK_NAMES)로 offset/size/labels 를 담은 태스크 메타 리스트를 만든다."""
    order = [t for t in TASK_NAMES if t in head_sizes]
    # 혹시 TASK_NAMES 에 없는 head 가 있으면 뒤에 덧붙인다(안전)
    order += [t for t in head_sizes if t not in order]
    tasks, off = [], 0
    for t in order:
        size = head_sizes[t]
        en, ko = TASK_TITLE.get(t, (t, t))
        labels = TASKS.get(t, [str(i) for i in range(size)])
        tasks.append({"name": t, "title": en, "title_ko": ko,
                      "labels": labels, "offset": off, "size": size})
        off += size
    return tasks, order, off


def load_cls_model(ckpt_path, imgsz_override=None):
    """체크포인트 -> (concat_model, imgsz, tasks_meta, total_out, backbone_name)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = (ckpt.get("args") if isinstance(ckpt, dict) else None) or {}
    sd = _extract_state_dict(ckpt)
    backbone = ckpt.get("model_name") if isinstance(ckpt, dict) else None
    if not backbone:
        raise SystemExit(f"model_name 없음: {ckpt_path} (분류 체크포인트가 맞는지 확인)")
    head_sizes = _head_sizes(sd)
    if not head_sizes:
        raise SystemExit(f"heads.* 가중치를 찾을 수 없습니다: {ckpt_path}")
    tasks, order, total = _tasks_meta(head_sizes)
    imgsz = int(imgsz_override or a.get("imgsz") or 224)
    embed_dim = int(a.get("embed_dim") or 512)
    dropout = float(a.get("dropout") or 0.5)

    net = MultiTaskNet(backbone, {t: head_sizes[t] for t in order},
                       embed_dim=embed_dim, dropout=dropout, pretrained=False)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing:
        print(f"    [warn] missing keys {len(missing)} (예: {missing[:2]})")
    if unexpected:
        print(f"    [warn] unexpected keys {len(unexpected)} (예: {unexpected[:2]})")
    net.eval()
    model = ConcatHead(net, order).eval()
    return model, imgsz, tasks, total, backbone


def convert_one(ckpt_path, out_dir, name, imgsz_override=None):
    """단일 분류 체크포인트 -> <name>.tflite + <name>.json (float32 I/O). 반환: meta dict."""
    import tensorflow as tf
    import onnx2tf

    work = Path(tempfile.mkdtemp(prefix="cls_export_"))
    try:
        model, imgsz, tasks, total, backbone = load_cls_model(ckpt_path, imgsz_override)
        print(f"[{name}] {backbone}  imgsz={imgsz}  tasks={[t['name'] for t in tasks]}  out={total}")

        onnx_path = work / "model.onnx"
        torch.onnx.export(
            model, torch.zeros(1, 3, imgsz, imgsz), str(onnx_path),
            input_names=["input"], output_names=["logits"],
            opset_version=13, do_constant_folding=True, dynamic_axes=None,
        )
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(work / "tf"),
            copy_onnx_input_output_names_to_tflite=True,
            non_verbose=True,
        )
        # react-native-fast-tflite 는 Float32Array 만 입력 가능 -> float32 I/O 모델 필수.
        f32 = next((work / "tf").glob("*_float32.tflite"), None)
        if f32 is None:
            raise RuntimeError(f"float32 tflite 미생성: {list((work / 'tf').glob('*.tflite'))}")
        dst = out_dir / f"{name}.tflite"
        shutil.copy2(f32, dst)

        # 입출력 dtype 가드
        _it = tf.lite.Interpreter(model_path=str(dst)); _it.allocate_tensors()
        idt = _it.get_input_details()[0]["dtype"].__name__
        odt = _it.get_output_details()[0]["dtype"].__name__
        if idt != "float32" or odt != "float32":
            raise RuntimeError(f"입출력이 float32 아님(input={idt}, output={odt})")

        meta = {
            "name": name, "task": "classification", "backbone": backbone,
            "imgsz": imgsz, "num_outputs": total, "tasks": tasks,
            "mean": IMAGENET_MEAN.tolist(), "std": IMAGENET_STD.tolist(),
            "layout": "NHWC", "input": [1, imgsz, imgsz, 3], "io_dtype": "float32",
            "activation": "softmax_per_task",
        }
        (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"[{name}] -> {dst.name} ({dst.stat().st_size/1e6:.1f} MB) + {name}.json")
        return meta
    finally:
        shutil.rmtree(work, ignore_errors=True)


def write_cls_registry(out_dir, app_dir):
    """assets/models 의 json 중 task=='classification' 만 골라 분류 전용 레지스트리 생성.
    앱은 src/ml/models.cls.gen.js 에서 분류 모델을 고른다(선택은 modelSelectCls.js)."""
    names = []
    for jp in sorted(out_dir.glob("*.json")):
        if not (out_dir / f"{jp.stem}.tflite").is_file():
            continue
        try:
            if json.loads(jp.read_text()).get("task") == "classification":
                names.append(jp.stem)
        except Exception:
            pass
    reg = app_dir / "dermascan-app" / "src" / "ml" / "models.cls.gen.js"
    lines = [
        "// ============================================================",
        "// AUTO-GENERATED by app/model/export_cls_tflite.py — 직접 수정하지 마세요.",
        "// 번들에 포함된 온디바이스 '분류' 모델 목록. 선택은 src/ml/modelSelectCls.js.",
        "// ============================================================",
        "export const CLS_MODELS = {",
    ]
    for n in names:
        lines.append(
            f'  {json.dumps(n)}: {{ meta: require("../../assets/models/{n}.json"), '
            f'model: require("../../assets/models/{n}.tflite") }},')
    lines += [
        "};",
        f"export const CLS_MODEL_KEYS = {json.dumps(names)};",
        f"export const DEFAULT_CLS_MODEL = {json.dumps(names[0]) if names else 'null'};",
        "",
    ]
    reg.write_text("\n".join(lines))
    print(f"[registry] {reg}  ({len(names)}개: {', '.join(names) or '없음'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", action="store_true",
                    help="--ckpt-dir 의 모든 .pt 를 각각 변환 (파일명이 모델 이름)")
    ap.add_argument("--ckpt-dir", default=str(DEFAULT_CKPT_DIR),
                    help="--batch 대상 폴더 (기본 app/model/checkpoint/classification)")
    ap.add_argument("--ckpt", default=None, help="단일 변환할 체크포인트(.pt)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="tflite 출력 폴더")
    ap.add_argument("--name", default=None, help="단일 변환 산출 이름(기본: 체크포인트 파일명)")
    ap.add_argument("--imgsz", type=int, default=None, help="입력 크기 override")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.batch:
        pts = sorted(Path(args.ckpt_dir).glob("*.pt"))
        if not pts:
            raise SystemExit(f"체크포인트 없음: {args.ckpt_dir}/*.pt")
        ok, fail = [], []
        for p in pts:
            try:
                convert_one(str(p), out_dir, p.stem, imgsz_override=args.imgsz)
                ok.append(p.stem)
            except Exception as e:
                msg = (str(e).splitlines() or [type(e).__name__])[0][:160]
                fail.append((p.stem, msg))
                print(f"[skip] {p.stem}: {msg}")
        write_cls_registry(out_dir, APP_DIR)
        print(f"\n[batch] 성공 {len(ok)}: {', '.join(ok)}")
        if fail:
            print("[batch] 건너뜀 " + "; ".join(f"{n}({m})" for n, m in fail))
        total = sum((out_dir / f"{n}.tflite").stat().st_size for n in ok) / 1e6
        print(f"[batch] 번들 tflite 총 ~{total:.0f} MB (그대로 APK 크기에 반영됨)")
        return

    ckpt = args.ckpt
    if not ckpt or not os.path.isfile(ckpt):
        raise SystemExit(f"체크포인트를 찾을 수 없습니다: {ckpt}")
    convert_one(ckpt, out_dir, args.name or Path(ckpt).stem, imgsz_override=args.imgsz)
    write_cls_registry(out_dir, APP_DIR)


if __name__ == "__main__":
    main()

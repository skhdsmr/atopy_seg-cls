"""세그멘테이션 체크포인트(.pth) -> 온디바이스용 TFLite 변환 (app 폴더 자립형).

경로:  PyTorch(smp) -> ONNX(고정 1x3xNxN, NCHW) -> onnx2tf -> TFLite(NHWC 입력).

이 스크립트는 app/ 하위에서 완결된다:
  - 입력 체크포인트: app/model/ 안의 .pth (기본 checkpoint_best.pth)
  - 출력 tflite:     app/dermascan-app/assets/models/
segmentation/ 등 상위 폴더에 의존하지 않는다(smp 만 있으면 동작).

앱(react-native-fast-tflite)은 NHWC float 입력 [1, imgsz, imgsz, 3] 을 받으므로
onnx2tf 가 자동으로 넣어주는 NCHW->NHWC 변환을 그대로 이용한다.

전처리 규약(학습/서버와 동일):
  Resize((imgsz, imgsz)) -> ImageNet 정규화 -> 모델 -> sigmoid -> thr(0.5)

준비(변환 툴체인, 최초 1회):
  pip install -r app/requirements-convert.txt

사용:
  # 기본: app/model/checkpoint_best.pth -> app/dermascan-app/assets/models/lesion_seg*.tflite
  python app/model/export_tflite.py --fp16

  # 다른 체크포인트/샘플 지정
  python app/model/export_tflite.py --ckpt app/model/my_ckpt.pth \
      --fp16 --sample /path/to/any.png
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import segmentation_models_pytorch as smp

# app/model/export_tflite.py -> APP_DIR = app/
APP_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = APP_DIR / "model"
DEFAULT_OUT = APP_DIR / "dermascan-app" / "assets" / "models"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_SMP_ARCH = {
    "unet": "unet", "unetpp": "unetplusplus", "unetplusplus": "unetplusplus",
    "manet": "manet", "deeplabv3": "deeplabv3", "deeplabv3plus": "deeplabv3plus",
    "fpn": "fpn", "pspnet": "pspnet", "linknet": "linknet", "pan": "pan",
}


def _norm_arch(name):
    return _SMP_ARCH.get(str(name).lower().replace("+", "plus").replace("-", ""),
                         str(name).lower())


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ("model", "state_dict", "model_state_dict", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _out_classes(sd):
    for k in ("segmentation_head.0.weight", "final.weight", "head.weight"):
        if k in sd:
            return int(sd[k].shape[0])
    return 1


def load_seg_model(ckpt_path):
    """체크포인트 args(있으면) 또는 파일명에서 arch/encoder 추론. 반환 (model, imgsz, nclass, arch, encoder)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = (ckpt.get("args") if isinstance(ckpt, dict) else None) or {}
    sd = _extract_state_dict(ckpt)
    nclass = _out_classes(sd) or 1
    if a.get("encoder") and a.get("decoder"):
        arch, encoder = _norm_arch(a["decoder"]), a["encoder"]
    else:
        toks = Path(ckpt_path).stem.split("_")
        if toks and toks[-1] in ("best", "last"):
            toks = toks[:-1]
        if len(toks) < 2:
            raise SystemExit(
                f"인코더/디코더를 알 수 없습니다: {ckpt_path}\n"
                "  args 없는 체크포인트면 파일명을 '<decoder>_<encoder>_best.pth' 형식으로 두거나\n"
                "  --arch / --encoder / --imgsz 로 직접 지정하세요.")
        arch, encoder = _norm_arch(toks[0]), "_".join(toks[1:])
    imgsz = int(a.get("imgsz", 512))
    model = smp.create_model(arch, encoder_name=encoder, encoder_weights=None,
                             in_channels=3, classes=nclass)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, imgsz, nclass, arch, encoder


def preprocess(sample_path, imgsz):
    from PIL import Image
    img = Image.open(sample_path).convert("RGB").resize((imgsz, imgsz), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    nchw = np.transpose(arr, (2, 0, 1))[None]
    return np.ascontiguousarray(nchw, dtype=np.float32)


import re


def _imgsz_from_name(name, default=512):
    """파일명 끝의 숫자를 입력크기로 사용(예: unext_512 -> 512). 없으면 default."""
    m = re.search(r"(\d+)\s*$", name)
    return int(m.group(1)) if m else default


def _is_unext(a, sd):
    """UNeXt(커스텀 아키텍처) 여부. args.arch 또는 state_dict 키(encoder1/final)로 판별."""
    if str((a or {}).get("arch", "")).lower() in ("unext", "unet_ext"):
        return True
    return "encoder1.weight" in sd and "final.weight" in sd


def _build_unext(sd, imgsz, nclass):
    """app/model/unext_arch.py 의 UNext 를 만들어 state_dict 로드."""
    import importlib.util
    p = Path(__file__).resolve().parent / "unext_arch.py"
    spec = importlib.util.spec_from_file_location("unext_arch", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.UNext(num_classes=nclass, input_channels=3, img_size=imgsz)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def _parity(model, onnx_path, tflite_path, sample, imgsz):
    """torch / onnxruntime / tflite 출력 일치 검증(옵션)."""
    import tensorflow as tf
    import onnxruntime as ort
    x_nchw = preprocess(sample, imgsz)
    with torch.no_grad():
        t_prob = 1.0 / (1.0 + np.exp(-model(torch.from_numpy(x_nchw)).numpy()))[0, 0]
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    o_prob = 1.0 / (1.0 + np.exp(-sess.run(None, {"input": x_nchw})[0]))[0, 0]
    interp = tf.lite.Interpreter(model_path=str(tflite_path)); interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    x_nhwc = np.ascontiguousarray(np.transpose(x_nchw, (0, 2, 3, 1)), dtype=np.float32)
    interp.set_tensor(inp["index"], x_nhwc); interp.invoke()
    l_prob = 1.0 / (1.0 + np.exp(-np.squeeze(interp.get_tensor(out["index"]))))

    def iou(a, b, thr=0.5):
        a, b = a >= thr, b >= thr
        u = np.logical_or(a, b).sum()
        return float(np.logical_and(a, b).sum() / u) if u else 1.0
    print(f"    parity: torch↔onnx {np.abs(t_prob-o_prob).max():.2e}  "
          f"torch↔tflite {np.abs(t_prob-l_prob).max():.2e}  IoU {iou(t_prob,l_prob):.4f}")


def convert_one(ckpt_path, out_dir, name, sample=None, arch=None, encoder=None, imgsz_override=None):
    """단일 체크포인트 -> <name>.tflite + <name>.json (float32 I/O). 반환: meta dict. 실패 시 예외."""
    import json
    import tensorflow as tf
    import onnx2tf

    work = Path(tempfile.mkdtemp(prefix="seg_export_"))
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        a = (ckpt.get("args") if isinstance(ckpt, dict) else None) or {}
        sd = _extract_state_dict(ckpt)
        nclass = _out_classes(sd) or 1

        if str(arch).lower() == "unext" or _is_unext(a, sd):
            # UNeXt(커스텀). args 없으면 파일명 끝 숫자를 입력크기로 사용(예: unext_512 -> 512).
            imgsz = imgsz_override or int(a.get("imgsz") or _imgsz_from_name(name, 512))
            if imgsz % 32:
                raise RuntimeError(f"UNeXt 입력크기는 32의 배수여야 함(현재 {imgsz}). --imgsz 로 지정하세요.")
            model = _build_unext(sd, imgsz, nclass)
            arch_, enc_ = "unext", "unext"
        elif arch and encoder:
            # args 없는 smp state_dict 를 사용자가 직접 지정한 경우
            arch_, enc_ = _norm_arch(arch), encoder
            imgsz = imgsz_override or int(a.get("imgsz") or 512)
            model = smp.create_model(arch_, encoder_name=enc_, encoder_weights=None,
                                     in_channels=3, classes=nclass)
            model.load_state_dict(sd, strict=False)
            model.eval()
        else:
            model, imgsz, nclass, arch_, enc_ = load_seg_model(ckpt_path)
            if imgsz_override:
                imgsz = imgsz_override
        print(f"[{name}] {arch_}/{enc_}  classes={nclass}  imgsz={imgsz}")

        # PyTorch -> ONNX -> onnx2tf -> TFLite
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
        # 반드시 float32 I/O 모델 사용(react-native-fast-tflite 는 Float32Array 만 입력 가능;
        # float16 I/O 모델을 쓰면 바이트 크기 불일치로 앱이 네이티브 크래시).
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
            "name": name, "imgsz": imgsz, "nclass": nclass, "arch": arch_, "encoder": enc_,
            "mean": IMAGENET_MEAN.tolist(), "std": IMAGENET_STD.tolist(),
            "layout": "NHWC", "input": [1, imgsz, imgsz, 3], "io_dtype": "float32",
            "activation": "sigmoid" if nclass <= 1 else "softmax_fg", "threshold": 0.5,
        }
        (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2))
        print(f"[{name}] -> {dst.name} ({dst.stat().st_size/1e6:.1f} MB) + {name}.json")

        if sample and os.path.isfile(sample):
            _parity(model, onnx_path, dst, sample, imgsz)
        return meta
    finally:
        shutil.rmtree(work, ignore_errors=True)


def write_registry(out_dir, app_dir):
    """assets/models 의 (json+tflite) 쌍을 스캔해 src/ml/models.gen.js 레지스트리를 생성한다.
    앱은 이 레지스트리에서 모델을 고른다(선택은 src/ml/modelSelect.js)."""
    import json
    # 세그멘테이션 레지스트리: 분류(task=='classification') 항목은 제외
    # (분류는 export_cls_tflite.py 가 models.cls.gen.js 로 따로 관리).
    names = []
    for jp in sorted(out_dir.glob("*.json")):
        if not (out_dir / f"{jp.stem}.tflite").is_file():
            continue
        try:
            if json.loads(jp.read_text()).get("task") == "classification":
                continue
        except Exception:
            pass
        names.append(jp.stem)
    reg = app_dir / "dermascan-app" / "src" / "ml" / "models.gen.js"
    lines = [
        "// ============================================================",
        "// AUTO-GENERATED by app/model/export_tflite.py — 직접 수정하지 마세요.",
        "// 번들에 포함된 온디바이스 모델 목록. 사용할 모델 선택은 src/ml/modelSelect.js.",
        "// ============================================================",
        "export const MODELS = {",
    ]
    for n in names:
        lines.append(
            f'  {json.dumps(n)}: {{ meta: require("../../assets/models/{n}.json"), '
            f'model: require("../../assets/models/{n}.tflite") }},')
    lines += [
        "};",
        f"export const MODEL_KEYS = {json.dumps(names)};",
        f"export const DEFAULT_MODEL = {json.dumps(names[0]) if names else 'null'};",
        "",
    ]
    reg.write_text("\n".join(lines))
    print(f"[registry] {reg}  ({len(names)}개: {', '.join(names) or '없음'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", action="store_true",
                    help="--ckpt-dir 의 모든 .pth 를 각각 변환 (파일명이 모델 이름)")
    ap.add_argument("--ckpt-dir", default=str(MODEL_DIR / "checkpoint" / "segmentation"),
                    help="--batch 대상 폴더 (기본 app/model/checkpoint/segmentation)")
    ap.add_argument("--ckpt", default=None, help="단일 변환할 체크포인트(.pth)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="tflite 출력 폴더")
    ap.add_argument("--name", default=None, help="단일 변환 산출 이름(기본: 체크포인트 파일명)")
    ap.add_argument("--sample", default=None, help="검증용 샘플 이미지(옵션)")
    ap.add_argument("--imgsz", type=int, default=None, help="입력 크기 override")
    ap.add_argument("--arch", default=None, help="args 없는 체크포인트용 디코더")
    ap.add_argument("--encoder", default=None, help="args 없는 체크포인트용 인코더")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.batch:
        pths = sorted(Path(args.ckpt_dir).glob("*.pth"))
        if not pths:
            raise SystemExit(f"체크포인트 없음: {args.ckpt_dir}/*.pth")
        ok, fail = [], []
        for p in pths:
            try:
                convert_one(str(p), out_dir, p.stem, sample=args.sample,
                            arch=args.arch, encoder=args.encoder, imgsz_override=args.imgsz)
                ok.append(p.stem)
            except Exception as e:
                msg = (str(e).splitlines() or [type(e).__name__])[0][:160]
                fail.append((p.stem, msg))
                print(f"[skip] {p.stem}: {msg}")
        write_registry(out_dir, APP_DIR)
        print(f"\n[batch] 성공 {len(ok)}: {', '.join(ok)}")
        if fail:
            print("[batch] 건너뜀 " + "; ".join(f"{n}({m})" for n, m in fail))
        total = sum((out_dir / f"{n}.tflite").stat().st_size for n in ok) / 1e6
        print(f"[batch] 번들 tflite 총 ~{total:.0f} MB (그대로 APK 크기에 반영됨)")
        return

    # 단일 변환
    ckpt = args.ckpt or str(MODEL_DIR / "checkpoint_best.pth")
    if not os.path.isfile(ckpt):
        raise SystemExit(f"체크포인트를 찾을 수 없습니다: {ckpt}")
    convert_one(ckpt, out_dir, args.name or Path(ckpt).stem, sample=args.sample,
                arch=args.arch, encoder=args.encoder, imgsz_override=args.imgsz)
    write_registry(out_dir, APP_DIR)


if __name__ == "__main__":
    main()

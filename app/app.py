"""
아토피 분할·분류 테스트셋 대시보드 (3분할).

  왼쪽  : test 데이터셋 목록 (썸네일/선택, 밀집·희소 필터)
  중간  : 선택 이미지의 분할 결과(예측·정답 오버레이) + 중증도 + 증상 등급
  오른쪽: 성능 누적(accumulation) — 전체 pooled 최종 지표(F1·Precision·Recall·IoU·Dice)
          + '지금까지 본 이미지'까지의 누적 + 밀집/희소 그룹별 지표

무거운 계산(모델 추론)은 하지 않는다. eval_precompute.py 가 미리 만든
outputs/eval_meta.json + 예측/정답 마스크(PNG)를 읽어 '표시'만 한다.

준비:
  python3 app/eval_precompute.py            # outputs/ 생성
실행:
  streamlit run app/app.py
"""

import json
import math
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent   # app/ 상위 = repo 루트
OUT = Path(__file__).resolve().parent / "outputs"   # 항상 app/outputs

# 결과 변형(단추로 전환). eval_precompute.py 가 outputs/<variant>/ 에 미리 생성.
VARIANTS = ["face", "lesion"]
VARIANT_KO = {"face": "얼굴 (face)", "lesion": "병변 (lesion)"}

TASK_KO = {
    "severity": "중증도 (IGA)",
    "erythema": "홍반 (Erythema)",
    "papulation": "구진 (Papulation)",
    "excoriation": "찰상 (Excoriation)",
    "lichenification": "태선화 (Lichenification)",
}

# 업로드/카메라 이미지 실시간 추론용 (classification/dataset.py 의 TASKS 와 동일 순서)
TASKS = {
    "severity":        ["Clear", "Almost Clear", "Mild", "Moderate", "Severe"],
    "erythema":        ["None", "Mild", "Moderate", "Severe"],
    "papulation":      ["None", "Mild", "Moderate", "Severe"],
    "excoriation":     ["None", "Mild", "Moderate", "Severe"],
    "lichenification": ["None", "Mild", "Moderate", "Severe"],
}
NUM_CLASSES = {t: len(v) for t, v in TASKS.items()}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

st.set_page_config(page_title="아토피 테스트셋 대시보드", page_icon="🩺", layout="wide")

# 페이지 전체에서 희미한(회색) 글씨를 검정으로 — 캡션·지표 라벨/값·위젯 라벨·도움말
st.markdown(
    """
    <style>
      [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *,
      [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *,
      [data-testid="stMetricValue"], [data-testid="stMetricValue"] *,
      [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
      .stMarkdown small, small { color: #000 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 로딩 유틸
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_meta(variant: str):
    path = OUT / variant / "eval_meta.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_rgb(rel_path: str):
    return np.array(Image.open(ROOT / rel_path).convert("RGB"))


@st.cache_data(show_spinner=False)
def load_mask(variant: str, rel_path: str):
    """outputs/<variant>/ 기준 상대경로의 마스크를 0/1 (H,W) 로."""
    return (np.array(Image.open(OUT / variant / rel_path).convert("L")) > 127).astype(np.uint8)


def overlay(img: np.ndarray, pred: np.ndarray | None, gt: np.ndarray | None,
            show_pred: bool, show_gt: bool, alpha: float = 0.45):
    """예측=빨강, 정답=초록 반투명 오버레이. 겹치면 노랑처럼 보인다."""
    base = img.astype(np.float32)
    out = base.copy()
    if show_gt and gt is not None:
        m = (gt > 0)[..., None]
        green = np.zeros_like(base); green[..., 1] = 255
        out = np.where(m, (1 - alpha) * out + alpha * green, out)
    if show_pred and pred is not None:
        m = (pred > 0)[..., None]
        red = np.zeros_like(base); red[..., 0] = 255
        out = np.where(m, (1 - alpha) * out + alpha * red, out)
    return out.clip(0, 255).astype(np.uint8)


def pooled_metrics(records):
    """레코드 리스트의 tp/fp/fn 을 합쳐 pooled 지표 계산."""
    tp = sum(r.get("tp", 0) for r in records)
    fp = sum(r.get("fp", 0) for r in records)
    fn = sum(r.get("fn", 0) for r in records)
    eps = 1e-7
    f1 = (2 * tp) / (2 * tp + fp + fn + eps)
    return {
        "f1": f1, "dice": f1,
        "iou": tp / (tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "n": len(records), "tp": tp, "fp": fp, "fn": fn,
    }


def metric_table(m):
    c1, c2 = st.columns(2)
    c1.metric("IoU", f"{m['iou']:.4f}")
    c2.metric("Dice", f"{m['dice']:.4f}")


def _qwk(yt, yp, k):
    """Quadratic Weighted Kappa (classification/metrics.py 와 동일 정의)."""
    O = np.zeros((k, k))
    for t, p in zip(yt, yp):
        O[t, p] += 1
    if O.sum() == 0:
        return 0.0
    w = np.array([[((i - j) ** 2) / ((k - 1) ** 2 + 1e-9) for j in range(k)]
                  for i in range(k)])
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (w * E).sum()
    return float(1 - (w * O).sum() / den) if den >= 1e-9 else 0.0


def cls_scores(records, task, num_classes):
    """레코드들의 (gt_idx, pred idx) 로 순서형 지표 계산. GT 없으면 None."""
    pairs = [(r["cls"][task]["gt_idx"], r["cls"][task]["idx"])
             for r in records
             if r.get("cls", {}).get(task, {}).get("gt_idx") is not None
             and "idx" in r["cls"][task]]
    if not pairs:
        return None
    yt = np.array([g for g, _ in pairs])
    yp = np.array([p for _, p in pairs])
    return {
        "n": len(pairs),
        "qwk": _qwk(yt, yp, num_classes),
        "acc": float((yt == yp).mean()),
        "acc1": float((np.abs(yt - yp) <= 1).mean()),
        "mae": float(np.abs(yt - yp).mean()),
    }


def cls_metric_block(records, tasks_meta):
    """IGA 중증도 QWK·±1 Acc·Acc + 증상 4종 QWK 를 표시."""
    sev = cls_scores(records, "severity", len(tasks_meta["severity"]))
    if sev is None:
        return
    st.markdown(f"**IGA 중증도 (분류 · n={sev['n']})**")
    c1, c2, c3 = st.columns(3)
    c1.metric("QWK", f"{sev['qwk']:.3f}")
    c2.metric("±1 Acc", f"{sev['acc1'] * 100:.1f}%")
    c3.metric("Acc", f"{sev['acc'] * 100:.1f}%")
    st.caption(f"MAE {sev['mae']:.3f} 등급")

    syms = ["erythema", "papulation", "excoriation", "lichenification"]
    parts = []
    for t in syms:
        s = cls_scores(records, t, len(tasks_meta[t]))
        if s:
            parts.append(f"{TASK_KO[t].split(' ')[0]} {s['qwk']:.2f}")
    if parts:
        st.caption("증상 QWK · " + " · ".join(parts))


# ---------------------------------------------------------------------------
# 실시간 추론(업로드/카메라 이미지) — 미리 계산된 test 셋과 달리 모델을 직접 돌린다.
#   분할 모델은 현재 변형(meta.seg_ckpt), 분류 모델은 공유(meta.cls_ckpt)를 사용.
# ---------------------------------------------------------------------------
def _load_py(path, name):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@st.cache_resource(show_spinner="분할 모델 로드 중…")
def _seg_model(seg_ckpt_rel):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mod = _load_py(ROOT / "segmentation/encoder_decoder/model.py", "dash_seg_def")
    ckpt = torch.load(ROOT / seg_ckpt_rel, map_location=device)
    a = ckpt.get("args", {})
    seg = mod.build_model(encoder_name=a.get("encoder", "tu-hrnet_w18"),
                          encoder_weights=None, decoder=a.get("decoder", "unet"),
                          num_classes=1, in_channels=3)
    seg.load_state_dict(ckpt["model"])
    seg.eval().to(device)
    return seg, int(a.get("imgsz", 512)), device


@st.cache_resource(show_spinner="분류 모델 로드 중…")
def _cls_model(cls_ckpt_rel):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mod = _load_py(ROOT / "classification/mobilenet/model.py", "dash_cls_def")
    ckpt = torch.load(ROOT / cls_ckpt_rel, map_location=device)
    a = ckpt.get("args", {})
    cls = mod.MultiTaskNet(ckpt.get("model_name", "mobilenetv4_conv_medium"),
                           NUM_CLASSES, embed_dim=a.get("embed_dim", 512),
                           dropout=a.get("dropout", 0.3))
    cls.load_state_dict(ckpt["model"])
    cls.eval().to(device)
    return cls, int(a.get("imgsz", 384)), device


def _preprocess(pil, size, device):
    import torch  # noqa: F401
    from torchvision import transforms as T
    tf = T.Compose([T.Resize((size, size)), T.ToTensor(),
                    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return tf(pil).unsqueeze(0).to(device)


def infer_seg(pil, seg_ckpt_rel, thr=0.5):
    import torch
    import torch.nn.functional as F
    seg, imgsz, device = _seg_model(seg_ckpt_rel)
    W, H = pil.size
    with torch.no_grad():
        prob = torch.sigmoid(seg(_preprocess(pil, imgsz, device)))
        prob = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)
        prob = prob[0, 0].cpu().numpy()
    return (prob >= thr).astype(np.uint8)


def infer_cls(pil, cls_ckpt_rel):
    import torch
    import torch.nn.functional as F
    cls, imgsz, device = _cls_model(cls_ckpt_rel)
    with torch.no_grad():
        out = cls(_preprocess(pil, imgsz, device))
    res = {}
    for t, names in TASKS.items():
        probs = F.softmax(out[t][0], dim=0).cpu().numpy()
        i = int(probs.argmax())
        res[t] = {"label": names[i], "idx": i, "conf": float(probs[i]),
                  "probs": [float(p) for p in probs], "names": names}
    return res


def _nice_axis(top_val):
    """최고값보다 '한 눈금 위'에서 끝나는 (domain_top, tick_values) 반환.

    예) 최고값 0.68 → 눈금 0·20·40·60·80%, 상단 80%에서 끝남.
    확률이므로 domain 은 100% 를 넘지 않도록 제한한다.
    """
    if top_val <= 0:
        return 1.0, [0.0, 0.5, 1.0]
    raw = top_val / 4.0                                  # 눈금 4칸 목표
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    domain_top = min((math.floor(top_val / step) + 1) * step, 1.0)
    n = int(round(domain_top / step))
    ticks = [round(step * i, 6) for i in range(n + 1)]
    return domain_top, ticks


def prob_chart(names, probs, height=170, x_size=15, y_size=11):
    """등급별 확률 막대그래프(민트 막대·그리드 없음).

    x축 라벨은 0도(가로)·크게, y축은 최고값보다 한 눈금 위에서 끝나게 잡는다.
    """
    names = list(names)
    probs = [float(p) for p in probs]
    domain_top, ticks = _nice_axis(max(probs))
    df = pd.DataFrame({"등급": names, "확률": probs})
    return (
        alt.Chart(df)
        .mark_bar(color="#5EEAD4")
        .encode(
            x=alt.X("등급:N", sort=names,
                    scale=alt.Scale(paddingInner=0.4, paddingOuter=0.3),
                    axis=alt.Axis(labelAngle=0, labelFontSize=x_size, labelLimit=1000,
                                  title=None, ticks=False, domainColor="#3a3f46")),
            y=alt.Y("확률:Q", scale=alt.Scale(domain=[0, domain_top], nice=False),
                    axis=alt.Axis(title=None, format=".0%", grid=False, values=ticks,
                                  labelFontSize=y_size, domain=False, ticks=False)),
            tooltip=["등급", alt.Tooltip("확률:Q", format=".1%")],
        )
        .properties(height=height)
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor="#000000")
    )


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
st.title("🩺 아토피 병변 분할 · 중증도 분류 — 테스트셋 대시보드")
st.caption("연구/교육용 데모이며 의료 진단이 아닙니다. 지표는 pooled(픽셀 누적) 방식입니다.")

# --- 결과 데이터셋 전환(단추): face / lesion ---
available = [v for v in VARIANTS if (OUT / v / "eval_meta.json").exists()]
if not available:
    st.warning(
        "사전 평가 결과가 없습니다. 먼저 실행하세요:\n\n"
        "```bash\npython3 app/eval_precompute.py        # face·lesion 모두 생성\n```"
    )
    st.stop()

if hasattr(st, "segmented_control"):
    variant = st.segmented_control(
        "결과 데이터셋", available, format_func=lambda v: VARIANT_KO[v],
        default=available[0], key="result_variant") or available[0]
else:
    variant = st.radio("결과 데이터셋", available, key="result_variant",
                       format_func=lambda v: VARIANT_KO[v], horizontal=True)

meta = load_meta(variant)

has_seg = meta.get("seg_ckpt") is not None
has_cls = meta.get("cls_ckpt") is not None

# 사용자가 추가한(업로드/촬영) 이미지 — 세션에 유지. GT 가 없어 지표엔 반영되지 않는다.
user_items = st.session_state.setdefault("user_items", [])
user_records = [{"user": True, "case": it["name"], "img_arr": it["img"],
                 "cls": it["cls"], "group": None} for it in user_items]

# 삭제된 test 이미지 키(변형별). 삭제 시 GT 이미지가 빠지면 지표도 함께 갱신된다.
deleted = st.session_state.setdefault("deleted", set())
visible_test = [r for r in meta["images"]
                if f"test:{variant}:{r['case']}" not in deleted]

all_imgs = visible_test + user_records

# --- 사이드바: 필터 & 표시 옵션 ---
with st.sidebar:
    st.header("설정")
    st.caption(f"결과 `{VARIANT_KO[variant]}` · 데이터셋 `{meta['dataset']}` · "
               f"split `{meta['split']}` · {meta['n_images']}장 · thr={meta['thr']}")
    group_sel = st.multiselect("도메인 필터 (GT 전경비율 기준)",
                               ["dense", "sparse"], default=["dense", "sparse"],
                               help=f"fg_thresh={meta.get('fg_thresh')} 기준 밀집/희소")
    show_pred = st.checkbox("예측 마스크(빨강)", value=True)
    show_gt = st.checkbox("정답 마스크(초록)", value=True)
    alpha = st.slider("오버레이 투명도", 0.1, 0.9, 0.45, 0.05)
    n_del = sum(1 for k in deleted if k.startswith(f"test:{variant}:"))
    if n_del and st.button(f"🔄 삭제 되돌리기 ({n_del}건 복구)"):
        for k in [k for k in deleted if k.startswith(f"test:{variant}:")]:
            deleted.discard(k)
        st.rerun()

# test 이미지는 도메인 필터 적용, 사용자 추가 이미지는 항상 목록에 유지
test_imgs = [r for r in visible_test if r.get("group") in group_sel] or visible_test
imgs = test_imgs + user_records

left, mid, right = st.columns([1.1, 2.2, 1.4], gap="large")

# ===========================================================================
# 왼쪽: 테스트 데이터셋 목록
# ===========================================================================
with left:
    n_user = len(user_records)
    st.subheader(f"📁 Test 데이터셋 ({len(imgs)}장"
                 + (f", 추가 {n_user}장" if n_user else "") + ")")

    # --- 촬영 / 불러오기: 이미지를 목록에 추가(추론은 선택 시 실시간) ---
    st.markdown("### 📷 촬영 / 불러오기")
    st.caption("업로드/촬영 이미지를 test 목록에 추가합니다. 정답(GT)이 없어 "
               "성능 지표에는 반영되지 않습니다.")
    up_file = st.file_uploader("이미지 불러오기", type=["png", "jpg", "jpeg"],
                               key="user_upload")
    cam_file = None
    if st.checkbox("카메라 촬영 열기", key="user_camera_on"):
        st.caption("⚠️ 카메라는 브라우저 보안상 **localhost 또는 HTTPS** 에서만 켜집니다. "
                   "IP 주소로 http 접속 시엔 차단됩니다.")
        cam_file = st.camera_input("촬영", key="user_camera", label_visibility="collapsed")
    src = cam_file or up_file
    if src is not None:
        pend = Image.open(src).convert("RGB")
        st.image(pend, caption="추가할 이미지", use_container_width=True)
        if st.button("➕ test 데이터셋에 추가", key="add_user", use_container_width=True):
            try:
                cres = infer_cls(pend, meta["cls_ckpt"]) if has_cls else None
            except Exception as e:  # noqa: BLE001
                st.error(f"분류 추론 실패: {e}")
                st.stop()
            n = st.session_state.get("user_counter", 0) + 1
            st.session_state["user_counter"] = n
            user_items.append({"id": n, "name": f"업로드_{n}",
                               "img": np.array(pend), "cls": cres})
            st.session_state[f"img_sel_{variant}"] = len(imgs)   # 새 항목(마지막) 자동 선택
            st.rerun()

    st.divider()
    if not imgs:
        st.info("표시할 이미지가 없습니다. 사이드바의 ‘삭제 되돌리기’로 복구하세요.")
        st.stop()

    labels = []
    for r in imgs:
        sev = r.get("cls", {}).get("severity", {}).get("label", "-")
        if r.get("user"):
            labels.append(f"📷 {r['case']}  ·  {sev}")
        else:
            tag = "🔴" if r.get("group") == "dense" else "🟡"
            labels.append(f"{tag} {r['case']}  ·  {sev}")

    # 저장된 선택 인덱스가 목록 축소로 범위를 벗어나면 보정
    sel_key = f"img_sel_{variant}"
    if st.session_state.get(sel_key, 0) >= len(imgs):
        st.session_state[sel_key] = len(imgs) - 1

    idx = st.radio("이미지 선택", range(len(imgs)), key=sel_key,
                   format_func=lambda i: labels[i], label_visibility="collapsed")
    cur = imgs[idx]

    if cur.get("user"):
        st.image(cur["img_arr"], caption=f"📷 {cur['case']}", use_container_width=True)
        st.caption("업로드/촬영 이미지 · 정답(GT) 없음")
    else:
        st.image(load_rgb(cur["image"]), caption=cur["case"], use_container_width=True)
        st.caption(f"그룹: **{cur.get('group')}** · GT 전경 {cur['fg_gt'] * 100:.2f}%"
                   + (f" · 예측 전경 {cur['fg_pred'] * 100:.2f}%" if "fg_pred" in cur else ""))

    # --- 삭제: 선택 이미지(원본 100장 포함)를 목록에서 제거 ---
    # 라디오(sel_key)는 이미 생성됐으므로 여기서 sel_key 를 바꾸면 안 된다.
    # 삭제 후 rerun 하면 라디오 직전의 범위 보정 로직이 인덱스를 안전하게 맞춘다.
    if st.button("🗑 이 이미지 삭제", key=f"del_{variant}", use_container_width=True):
        if cur.get("user"):
            st.session_state["user_items"] = [it for it in user_items
                                              if it["name"] != cur["case"]]
        else:
            deleted.add(f"test:{variant}:{cur['case']}")
        st.rerun()

# ===========================================================================
# 중간: 분할 결과 + 중증도 + 증상
# ===========================================================================
with mid:
    user_mode = cur.get("user", False)
    st.subheader("🔬 분할 결과" + (" · 📷 업로드/촬영" if user_mode else ""))
    if user_mode:
        img = cur["img_arr"]
        try:
            pred = (infer_seg(Image.fromarray(img), meta["seg_ckpt"],
                              meta.get("thr", 0.5)) if has_seg else None)
        except Exception as e:  # noqa: BLE001
            st.error(f"실시간 추론 실패: {e}\n\n"
                     "torch / segmentation-models-pytorch / timm 설치를 확인하세요.")
            st.stop()
        gt = None
        cls = cur.get("cls")
        per_image = None
        st.caption(f"업로드/촬영 이미지 · 분할 `{VARIANT_KO[variant]}` 모델 · "
                   "정답(GT) 없음 → 실시간 추론 결과")
    else:
        img = load_rgb(cur["image"])
        pred = load_mask(variant, cur["pred_mask"]) if ("pred_mask" in cur and has_seg) else None
        gt = load_mask(variant, cur["gt_mask"]) if "gt_mask" in cur else None
        cls = cur.get("cls")
        per_image = cur.get("per_image")

    c1, c2 = st.columns(2)
    c1.image(img, caption="원본", use_container_width=True)
    ov_cap = "오버레이 (빨강=예측)" if gt is None else "오버레이 (빨강=예측, 초록=정답)"
    c2.image(overlay(img, pred, gt, show_pred, show_gt, alpha),
             caption=ov_cap, use_container_width=True)

    if per_image:
        m1, m2 = st.columns(2)
        m1.metric("IoU", f"{per_image['iou']:.3f}")
        m2.metric("Dice", f"{per_image['dice']:.3f}")

    st.divider()
    st.subheader("🏷️ 중증도 · 증상 등급")
    if cls:
        def gt_line(r):
            g = r.get("gt_label")
            if not g:
                return None
            return f"정답(GT): {g} " + ("✅" if g == r["label"] else "❌")

        sev = cls["severity"]
        st.metric(TASK_KO["severity"], sev["label"], f"conf {sev['conf'] * 100:.0f}%")
        if gt_line(sev):
            st.caption(gt_line(sev))
        cols = st.columns(4)
        for c, t in zip(cols, ["erythema", "papulation", "excoriation", "lichenification"]):
            r = cls[t]
            c.metric(TASK_KO[t], r["label"], f"{r['conf'] * 100:.0f}%")
            if gt_line(r):
                c.caption(gt_line(r))
        with st.expander("클래스별 확률 분포", expanded=True):
            sev = cls["severity"]
            st.markdown(f"**{TASK_KO['severity']}**")
            _, igc, _ = st.columns([1, 3, 1])       # 가운데 좁은 폭으로 밀집 배치
            with igc:
                st.altair_chart(
                    prob_chart(sev["names"], sev["probs"],
                               height=250, x_size=15, y_size=12),
                    use_container_width=True)

            st.markdown("**증상별 등급**")
            syms = ["erythema", "papulation", "excoriation", "lichenification"]
            for row in (syms[:2], syms[2:]):           # 2×2 배치
                rc = st.columns(2)
                for c, t in zip(rc, row):
                    r = cls[t]
                    with c:
                        st.caption(TASK_KO[t])
                        st.altair_chart(
                            prob_chart(r["names"], r["probs"],
                                       height=210, x_size=14, y_size=11),
                            use_container_width=True)
    else:
        st.info("분류 예측이 eval_meta 에 없습니다 (분류 가중치 없이 평가됨).")

# ===========================================================================
# 오른쪽: 성능 누적 (accumulation)
# ===========================================================================
with right:
    st.subheader("📊 성능 평가 (누적)")
    st.caption("정답(GT)이 있는 이미지에 대해서만 집계합니다. "
               "업로드/촬영 이미지는 반영되지 않습니다.")
    if not (has_seg or has_cls):
        st.info("가중치 없이 평가되어 지표가 없습니다.")
    else:
        # 지표는 GT(분할=tp 존재) 가 있는 이미지에 대해서만 계산 → 사용자 추가분 자동 제외
        seg_all = [r for r in all_imgs if "tp" in r]
        st.markdown(f"**전체 test셋 최종 (pooled · GT {len(seg_all)}장)**")
        if has_seg:
            st.caption("분할 (Segmentation)")
            metric_table(pooled_metrics(seg_all))
        if has_cls:
            cls_metric_block(all_imgs, meta["tasks"])

        st.divider()
        st.markdown(f"**누적: 목록 1 ~ {idx + 1}번째까지**")
        window = imgs[: idx + 1]
        seg_win = [r for r in window if "tp" in r]
        if has_seg:
            running = pooled_metrics(seg_win)
            st.caption(f"분할 · GT {running['n']}장 픽셀 누적 · TP={running['tp']:,} "
                       f"FP={running['fp']:,} FN={running['fn']:,}")
            metric_table(running)
        if has_cls:
            cls_metric_block(window, meta["tasks"])

        if has_seg:
            st.divider()
            st.markdown("**분할 도메인 그룹별 (전체 test셋)**")
            for g in ["dense", "sparse"]:
                recs = [r for r in all_imgs if r.get("group") == g]
                if not recs:
                    continue
                gm = pooled_metrics(recs)
                tag = "🔴 밀집(dense)" if g == "dense" else "🟡 희소(sparse)"
                st.write(f"{tag} · {gm['n']}장")
                st.caption(f"IoU {gm['iou']:.3f} · Dice {gm['dice']:.3f}")

# ===========================================================================
# 3분할 경계를 드래그해서 좌/중/우 칸 너비를 조절하는 기능.
#   Streamlit 은 컬럼 리사이즈를 기본 지원하지 않으므로, 부모 문서(DOM)에 접근해
#   컬럼 사이에 드래그 핸들을 끼워 넣고 flex-grow 비율을 조정한다. 조정한 비율은
#   window.parent 전역에 저장해 rerun(위젯 조작) 후에도 유지한다.
# ===========================================================================
_RESIZE_JS = """
<script>
(function () {
  const doc = window.parent.document;
  const KEY = "__atopy_col_ratios__";

  function findBlock() {
    const blocks = doc.querySelectorAll('[data-testid="stHorizontalBlock"]');
    for (const b of blocks) {
      const cols = b.querySelectorAll(
        ':scope > [data-testid="stColumn"], :scope > [data-testid="column"]');
      if (cols.length === 3) return { block: b, cols: Array.from(cols) };
    }
    return null;
  }

  function apply() {
    const found = findBlock();
    if (!found) return;
    const { block, cols } = found;
    let ratios = window.parent[KEY] || [1.1, 2.2, 1.4];
    window.parent[KEY] = ratios;

    if (block.dataset.resizerReady === "1") {
      cols.forEach((c, i) => { c.style.flex = ratios[i] + " 1 0%"; });
      return;
    }
    block.dataset.resizerReady = "1";
    block.style.alignItems = "stretch";
    cols.forEach((c, i) => {
      c.style.flex = ratios[i] + " 1 0%";
      c.style.minWidth = "0";
    });

    for (let i = 0; i < cols.length - 1; i++) {
      const left = cols[i], right = cols[i + 1];
      const handle = doc.createElement("div");
      handle.style.cssText =
        "flex:0 0 10px;align-self:stretch;cursor:col-resize;" +
        "display:flex;align-items:center;justify-content:center;z-index:10;";
      const bar = doc.createElement("div");
      bar.style.cssText =
        "width:3px;height:100%;border-radius:3px;background:rgba(130,130,130,.35);" +
        "transition:background .15s;";
      handle.appendChild(bar);
      left.after(handle);

      let dragging = false, startX = 0, startL = 0, startR = 0, startW = 1;
      const paint = () => bar.style.background =
        dragging ? "rgba(80,140,255,.9)" : "rgba(130,130,130,.35)";
      handle.addEventListener("mouseenter", () => {
        if (!dragging) bar.style.background = "rgba(80,140,255,.6)"; });
      handle.addEventListener("mouseleave", paint);
      handle.addEventListener("mousedown", (e) => {
        dragging = true;
        startX = e.clientX;
        startL = parseFloat(left.style.flexGrow) || ratios[i];
        startR = parseFloat(right.style.flexGrow) || ratios[i + 1];
        startW = left.getBoundingClientRect().width +
                 right.getBoundingClientRect().width || 1;
        doc.body.style.userSelect = "none";
        paint();
        e.preventDefault();
      });
      doc.addEventListener("mousemove", (e) => {
        if (!dragging) return;
        const total = startL + startR;
        const dr = ((e.clientX - startX) / startW) * total;
        let nl = startL + dr, nr = startR - dr;
        const minR = 0.25 * total;
        if (nl < minR || nr < minR) return;
        left.style.flex = nl + " 1 0%";
        right.style.flex = nr + " 1 0%";
        ratios[i] = nl; ratios[i + 1] = nr;
        window.parent[KEY] = ratios;
      });
      doc.addEventListener("mouseup", () => {
        if (dragging) { dragging = false; doc.body.style.userSelect = ""; paint(); }
      });
    }
  }

  apply();
  new MutationObserver(apply).observe(doc.body, { childList: true, subtree: true });
})();
</script>
"""
components.html(_RESIZE_JS, height=0)

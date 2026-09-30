// ============================================================
//                ▼▼▼  사용할 온디바이스 모델 선택  ▼▼▼
//
// 여기 값만 바꾸고 앱을 리로드하면 그 모델로 추론합니다(재빌드 불필요).
// 이름 = app/model/checkpoint 의 파일명(확장자 제외) = assets/models 의 tflite 이름.
//
// 선택 가능한 목록은 자동생성된 models.gen.js 의 MODEL_KEYS 를 보세요.
// 예: 'effb0_unetpp_256', 'effb0_unetpp_384', 'effb0_unetpp_512'
//
// 주의: 여기 적은 모델이 번들에 실제로 있어야 합니다.
//       (없으면 models.gen.js 의 DEFAULT_MODEL 로 자동 대체)
//       새 체크포인트를 추가하려면:  python app/model/export_tflite.py --batch
// ============================================================
export const ACTIVE_MODEL = 'effb0_unetpp_384';

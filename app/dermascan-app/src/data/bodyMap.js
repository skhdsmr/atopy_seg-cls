// 2D body map definition (axis 1: body part metadata)
// Radiation dermatitis = fixed irradiation site / uremic pruritus = register multiple scratch-prone sites

export const CONDITIONS = [
  { key: 'uremic', label: '요독성 소양증', hint: '투석·말기신부전 가려움. 찰상 부위를 등록하세요.' },
  { key: 'radiation', label: '방사선 피부염', hint: '방사선 조사 부위(고정)를 등록하세요.' },
  { key: 'other', label: '기타', hint: '' },
];

export const BODY_VIEWS = [
  { key: 'front', label: '앞면' },
  { key: 'back', label: '뒷면' },
];

// Front/back body part lists (selected via tabs)
export const BODY_REGIONS = {
  front: ['얼굴', '목', '가슴', '복부', '어깨', '상완', '전완', '손', '허벅지', '정강이', '발등'],
  back: ['뒤통수', '목뒤', '등 상부', '등 하부', '허리', '엉덩이', '종아리', '발뒤꿈치'],
};

export const SIDES = [
  { key: 'L', label: '좌' },
  { key: 'R', label: '우' },
  { key: 'C', label: '중앙' },
];

export const conditionLabel = (key) =>
  (CONDITIONS.find((c) => c.key === key) || {}).label || '기타';

export const sideLabel = (key) =>
  (SIDES.find((s) => s.key === key) || {}).label || '';

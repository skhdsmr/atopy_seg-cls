// Demo mock data (used until the real server is integrated)

export const MOCK_HOSPITALS = [
  {
    id: 'h1',
    name: '서울 피부과의원',
    dept: '피부과',
    doctor: '김민준 의사',
    distanceKm: 0.3,
    rating: 4.8,
    status: '진료 중',
    address: '강남구 테헤란로 123',
    phone: '02-1234-5678',
    registered: true,
  },
  {
    id: 'h2',
    name: '청담 메디컬센터',
    dept: '피부과·성형외과',
    doctor: '이수진 의사',
    distanceKm: 0.8,
    rating: 4.6,
    status: '진료 중',
    address: '강남구 청담동 45',
    phone: '02-2345-6789',
    registered: false,
  },
  {
    id: 'h3',
    name: '강남 피부클리닉',
    dept: '피부과',
    doctor: '박지영 의사',
    distanceKm: 1.2,
    rating: 4.4,
    status: '진료 종료',
    address: '강남구 역삼동 67',
    phone: '02-3456-7890',
    registered: false,
  },
];

export const MOCK_CHATS = [
  {
    id: 'c1',
    doctor: '김민준 의사',
    hospital: '서울 피부과의원',
    avatar: '김',
    lastMessage: '내일 진료실에서 뵙겠습니다.',
    time: '10분 전',
    unread: 2,
    messages: [
      { id: 'm1', text: '안녕하세요, 사진 잘 받았습니다.', senderRole: 'doctor', ts: '오전 9:10' },
      { id: 'm2', text: '증상이 언제부터 있으셨나요?', senderRole: 'doctor', ts: '오전 9:11' },
      { id: 'm3', text: '3일 전부터 가렵기 시작했어요.', senderRole: 'patient', ts: '오전 9:20' },
      { id: 'm4', text: '내일 진료실에서 뵙겠습니다.', senderRole: 'doctor', ts: '오전 9:25' },
    ],
  },
  {
    id: 'c2',
    doctor: '이수진 의사',
    hospital: '청담 메디컬센터',
    avatar: '이',
    lastMessage: '사진 잘 받았습니다. 확인 후 연락드리겠습니다.',
    time: '어제',
    unread: 0,
    messages: [
      { id: 'm1', text: '사진 잘 받았습니다. 확인 후 연락드리겠습니다.', senderRole: 'doctor', ts: '어제' },
    ],
  },
];

// Records for calendar display (key: YYYY-MM-DD)
export const MOCK_RECORDS = {
  '2026-06-03': { severity: '경증', note: '' },
  '2026-06-07': { severity: '중등도', note: '' },
  '2026-06-10': { severity: '중등도', note: '' },
  '2026-06-14': { severity: '중증', note: '' },
  '2026-06-17': { severity: '중등도', note: '' },
  '2026-06-20': { severity: '경증', note: '' },
  '2026-06-24': { severity: '경증', note: '거의 회복. 피부결 개선 중.', photo: null },
};

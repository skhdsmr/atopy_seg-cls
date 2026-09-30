// App-wide color/spacing tokens (based on the mockup green tone)
export const colors = {
  primary: '#2F6B5A',
  primaryDark: '#245446',
  primarySoft: '#E7F0EC',
  naver: '#5BB98C',
  kakao: '#FAE100',
  kakaoText: '#3C1E1E',
  google: '#FFFFFF',

  bg: '#FFFFFF',
  surface: '#F4F7F5',
  card: '#FFFFFF',
  border: '#E3E9E6',

  text: '#1F2B27',
  textSub: '#7A8A84',
  textMuted: '#9AA8A2',

  // Severity
  mild: '#3FB07A',     // mild (green)
  moderate: '#E8973A', // moderate (orange)
  severe: '#E25555',   // severe (red)

  warning: '#F6A609',
  warningBg: '#FFF6E6',
  white: '#FFFFFF',
};

export const spacing = { xs: 4, sm: 8, md: 16, lg: 24, xl: 32 };

export const radius = { sm: 8, md: 12, lg: 16, xl: 24, pill: 999 };

export const severityColor = (level) => {
  switch (level) {
    case '경증': return colors.mild;
    case '중등도': return colors.moderate;
    case '중증': return colors.severe;
    default: return colors.textSub;
  }
};

import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useAuth } from '../context/AuthContext';
import { colors, radius, spacing } from '../theme';

export default function LoginScreen({ navigation }) {
  const { startSocialLogin } = useAuth();
  const [busy, setBusy] = useState(null);

  const handle = async (provider) => {
    setBusy(provider);
    try {
      const res = await startSocialLogin(provider);
      // New signups go to the role select screen; existing accounts log in automatically
      if (res.isNew) {
        navigation.navigate('RoleSelect', { provider });
      }
    } finally {
      setBusy(null);
    }
  };

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.content}>
        {/* Logo / brand */}
        <View style={styles.logoBox}>
          <Ionicons name="medkit" size={40} color={colors.white} />
        </View>
        <Text style={styles.brand}>DermaScan</Text>
        <Text style={styles.tagline}>AI 기반 병변 진단 & 병원 연계 서비스</Text>

        {/* Stats */}
        <View style={styles.stats}>
          <Stat value="98.2%" label="진단 정확도" />
          <Divider />
          <Stat value="3,400+" label="협력 병원" />
          <Divider />
          <Stat value="52만+" label="진단 건수" />
        </View>
      </View>

      {/* Social login */}
      <View style={styles.bottom}>
        <Text style={styles.socialLabel}>소셜 계정으로 시작하기</Text>

        <SocialButton
          provider="google"
          label="Google로 로그인"
          bg={colors.white}
          textColor={colors.text}
          bordered
          busy={busy === 'google'}
          onPress={() => handle('google')}
          icon={<Ionicons name="logo-google" size={20} color="#EA4335" />}
        />
        <SocialButton
          provider="naver"
          label="네이버로 로그인"
          bg={colors.naver}
          textColor={colors.white}
          busy={busy === 'naver'}
          onPress={() => handle('naver')}
          icon={<Text style={[styles.iconText, { color: colors.white }]}>N</Text>}
        />
        <SocialButton
          provider="kakao"
          label="카카오로 로그인"
          bg={colors.kakao}
          textColor={colors.kakaoText}
          busy={busy === 'kakao'}
          onPress={() => handle('kakao')}
          icon={<Ionicons name="chatbubble" size={18} color={colors.kakaoText} />}
        />

        <Text style={styles.terms}>
          로그인 시 <Text style={styles.link}>이용약관</Text> 및{' '}
          <Text style={styles.link}>개인정보처리방침</Text>에 동의합니다
        </Text>
      </View>
    </SafeAreaView>
  );
}

function Stat({ value, label }) {
  return (
    <View style={styles.stat}>
      <Text style={styles.statValue}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </View>
  );
}
const Divider = () => <View style={styles.vline} />;

function SocialButton({ label, bg, textColor, bordered, icon, busy, onPress }) {
  return (
    <TouchableOpacity
      style={[styles.social, { backgroundColor: bg }, bordered && styles.socialBorder]}
      onPress={onPress}
      disabled={busy}
      activeOpacity={0.85}
    >
      <View style={styles.socialIcon}>{busy ? <ActivityIndicator color={textColor} /> : icon}</View>
      <Text style={[styles.socialText, { color: textColor }]}>{label}</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bg },
  content: { flex: 1, alignItems: 'center', justifyContent: 'center', paddingHorizontal: spacing.lg },
  logoBox: {
    width: 84, height: 84, borderRadius: 24, backgroundColor: colors.primary,
    alignItems: 'center', justifyContent: 'center', marginBottom: spacing.md,
  },
  brand: { fontSize: 30, fontWeight: '800', color: colors.text },
  tagline: { fontSize: 14, color: colors.primary, marginTop: 6 },
  stats: { flexDirection: 'row', alignItems: 'center', marginTop: spacing.xl },
  stat: { alignItems: 'center', paddingHorizontal: spacing.md },
  statValue: { fontSize: 22, fontWeight: '800', color: colors.primary },
  statLabel: { fontSize: 12, color: colors.textSub, marginTop: 2 },
  vline: { width: 1, height: 32, backgroundColor: colors.border },
  bottom: { paddingHorizontal: spacing.lg, paddingBottom: spacing.lg },
  socialLabel: { textAlign: 'center', color: colors.textMuted, fontSize: 13, marginBottom: spacing.md },
  social: {
    flexDirection: 'row', alignItems: 'center', borderRadius: radius.md,
    paddingVertical: 16, paddingHorizontal: 18, marginBottom: 12,
  },
  socialBorder: { borderWidth: 1, borderColor: colors.border },
  socialIcon: { width: 24, alignItems: 'center' },
  iconText: { fontWeight: '900', fontSize: 16 },
  socialText: { flex: 1, textAlign: 'center', fontSize: 16, fontWeight: '700', marginRight: 24 },
  terms: { textAlign: 'center', color: colors.textMuted, fontSize: 12, marginTop: spacing.sm },
  link: { color: colors.primary, textDecorationLine: 'underline' },
});

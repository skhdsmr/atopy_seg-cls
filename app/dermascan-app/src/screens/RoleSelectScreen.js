import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useAuth } from '../context/AuthContext';
import { colors, radius, spacing } from '../theme';

const PROVIDER_LABEL = { google: 'Google', naver: '네이버', kakao: '카카오' };

export default function RoleSelectScreen({ navigation, route }) {
  const { provider } = route.params;
  const { completeSignup } = useAuth();
  const [role, setRole] = useState(null);
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    if (!role) return;
    setBusy(true);
    try {
      await completeSignup(provider, role); // signup complete → switches to main automatically
    } catch (e) {
      setBusy(false);
    }
  };

  return (
    <SafeAreaView style={styles.safe}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} hitSlop={10}>
          <Ionicons name="arrow-back" size={24} color={colors.text} />
        </TouchableOpacity>
      </View>

      <View style={styles.body}>
        <Text style={styles.step}>회원가입 · 1/1</Text>
        <Text style={styles.title}>어떤 유형으로{'\n'}가입하시나요?</Text>
        <Text style={styles.sub}>
          {PROVIDER_LABEL[provider] ?? '소셜'} 계정으로 처음 가입합니다.{'\n'}
          가입 후에는 로그인 시 자동으로 적용됩니다.
        </Text>

        <View style={styles.cards}>
          <RoleCard
            active={role === 'patient'}
            onPress={() => setRole('patient')}
            icon="person"
            title="환자로 가입"
            desc="병변을 진단받고, 병원·기록·상담을 이용해요"
          />
          <RoleCard
            active={role === 'doctor'}
            onPress={() => setRole('doctor')}
            icon="medkit"
            title="의사로 가입"
            desc="환자 상담 요청을 받고 채팅으로 관리해요"
          />
        </View>
      </View>

      {/* Bottom confirm button */}
      <View style={styles.footer}>
        <TouchableOpacity
          style={[styles.cta, (!role || busy) && { opacity: 0.5 }]}
          onPress={submit}
          disabled={!role || busy}
          activeOpacity={0.85}
        >
          {busy ? (
            <ActivityIndicator color={colors.white} />
          ) : (
            <Text style={styles.ctaText}>
              {role === 'doctor' ? '의사로 시작하기' : role === 'patient' ? '환자로 시작하기' : '유형을 선택하세요'}
            </Text>
          )}
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

function RoleCard({ active, onPress, icon, title, desc }) {
  return (
    <TouchableOpacity
      style={[styles.card, active && styles.cardActive]}
      onPress={onPress}
      activeOpacity={0.85}
    >
      <View style={[styles.cardIcon, active && styles.cardIconActive]}>
        <Ionicons name={icon} size={26} color={active ? colors.white : colors.primary} />
      </View>
      <View style={{ flex: 1 }}>
        <Text style={[styles.cardTitle, active && { color: colors.primary }]}>{title}</Text>
        <Text style={styles.cardDesc}>{desc}</Text>
      </View>
      <Ionicons
        name={active ? 'checkmark-circle' : 'ellipse-outline'}
        size={24}
        color={active ? colors.primary : colors.border}
      />
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bg },
  header: { paddingHorizontal: spacing.md, paddingVertical: spacing.sm },
  body: { flex: 1, paddingHorizontal: spacing.lg, paddingTop: spacing.md },
  step: { fontSize: 13, fontWeight: '700', color: colors.primary, marginBottom: 8 },
  title: { fontSize: 26, fontWeight: '800', color: colors.text, lineHeight: 34 },
  sub: { fontSize: 14, color: colors.textSub, marginTop: 12, lineHeight: 21 },
  cards: { marginTop: spacing.xl, gap: 14 },
  card: {
    flexDirection: 'row', alignItems: 'center', gap: 14,
    borderWidth: 1.5, borderColor: colors.border, borderRadius: radius.lg,
    padding: spacing.md, backgroundColor: colors.card,
  },
  cardActive: { borderColor: colors.primary, backgroundColor: colors.primarySoft },
  cardIcon: {
    width: 52, height: 52, borderRadius: 26, backgroundColor: colors.primarySoft,
    alignItems: 'center', justifyContent: 'center',
  },
  cardIconActive: { backgroundColor: colors.primary },
  cardTitle: { fontSize: 17, fontWeight: '800', color: colors.text },
  cardDesc: { fontSize: 13, color: colors.textSub, marginTop: 4, lineHeight: 19 },
  footer: { paddingHorizontal: spacing.lg, paddingBottom: spacing.lg },
  cta: {
    backgroundColor: colors.primary, borderRadius: radius.md,
    paddingVertical: 17, alignItems: 'center',
  },
  ctaText: { color: colors.white, fontSize: 16, fontWeight: '700' },
});

import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { Card } from '../components/ui';
import { useAuth } from '../context/AuthContext';
import { colors, radius, spacing } from '../theme';

const MENU = [
  { icon: 'notifications-outline', title: '알림 설정', sub: '병원 방문 · 진단 결과' },
  { icon: 'time-outline', title: '방문 예약 알람', sub: '6월 28일 예약 중' },
  { icon: 'card-outline', title: '결제 내역', sub: '이번 달 채팅 3,500원' },
  { icon: 'person-outline', title: '개인정보 관리', sub: '약관 · 데이터 설정' },
];

export default function ProfileScreen() {
  const { user, signOut, resetAccounts } = useAuth();

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView contentContainerStyle={styles.scroll}>
        <Text style={styles.h1}>프로필</Text>

        {/* User card */}
        <View style={styles.userCard}>
          <View style={styles.avatar}>
            <Ionicons name="person" size={28} color={colors.white} />
          </View>
          <View style={{ flex: 1 }}>
            <Text style={styles.name}>{user?.name ?? '사용자'}</Text>
            <Text style={styles.email}>{user?.email}</Text>
            <View style={styles.roleChip}>
              <Text style={styles.roleChipText}>{user?.role === 'doctor' ? '의사 계정' : '환자 계정'}</Text>
            </View>
          </View>
        </View>

        {/* Stats (per role) */}
        <View style={styles.stats}>
          {user?.role === 'doctor' ? (
            <>
              <Stat value={`${user?.patients ?? 0}명`} label="담당 환자" />
              <Stat value={`${user?.consults ?? 0}건`} label="상담 건수" />
              <Stat value={user?.hospital ?? '-'} label="소속 병원" />
            </>
          ) : (
            <>
              <Stat value={`${user?.diagnoses ?? 0}회`} label="진단 횟수" />
              <Stat value={`${user?.hospitals ?? 0}곳`} label="등록 병원" />
              <Stat value={`${user?.streak ?? 0}일`} label="연속 기록" />
            </>
          )}
        </View>

        {/* Menu */}
        {MENU.map((m) => (
          <Card key={m.title} style={styles.menuItem}>
            <View style={styles.menuIcon}>
              <Ionicons name={m.icon} size={20} color={colors.primary} />
            </View>
            <View style={{ flex: 1 }}>
              <Text style={styles.menuTitle}>{m.title}</Text>
              <Text style={styles.menuSub}>{m.sub}</Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color={colors.textMuted} />
          </Card>
        ))}

        <TouchableOpacity style={styles.logout} onPress={signOut}>
          <Text style={styles.logoutText}>로그아웃</Text>
        </TouchableOpacity>

        {/* Dev: reset signup history → the role select screen shows again on next login */}
        <TouchableOpacity style={styles.devReset} onPress={resetAccounts}>
          <Text style={styles.devResetText}>가입 유형 초기화 (개발용)</Text>
        </TouchableOpacity>
      </ScrollView>
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

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bg },
  scroll: { padding: spacing.md, paddingBottom: spacing.xl },
  h1: { fontSize: 24, fontWeight: '800', color: colors.text, marginBottom: spacing.md },
  userCard: { flexDirection: 'row', alignItems: 'center', gap: 14, backgroundColor: colors.primary, borderRadius: radius.lg, padding: spacing.md },
  avatar: { width: 56, height: 56, borderRadius: 28, backgroundColor: 'rgba(255,255,255,0.2)', alignItems: 'center', justifyContent: 'center' },
  name: { fontSize: 18, fontWeight: '800', color: colors.white },
  email: { fontSize: 13, color: 'rgba(255,255,255,0.85)', marginTop: 2 },
  roleChip: { alignSelf: 'flex-start', backgroundColor: 'rgba(255,255,255,0.2)', paddingHorizontal: 10, paddingVertical: 3, borderRadius: radius.pill, marginTop: 6 },
  roleChipText: { color: colors.white, fontSize: 12, fontWeight: '700' },
  stats: { flexDirection: 'row', gap: 10, marginVertical: spacing.md },
  stat: { flex: 1, backgroundColor: colors.card, borderRadius: radius.md, borderWidth: 1, borderColor: colors.border, paddingVertical: 16, alignItems: 'center' },
  statValue: { fontSize: 18, fontWeight: '800', color: colors.text },
  statLabel: { fontSize: 12, color: colors.textSub, marginTop: 2 },
  menuItem: { flexDirection: 'row', alignItems: 'center', gap: 12, marginBottom: 10 },
  menuIcon: { width: 40, height: 40, borderRadius: 20, backgroundColor: colors.primarySoft, alignItems: 'center', justifyContent: 'center' },
  menuTitle: { fontSize: 15, fontWeight: '700', color: colors.text },
  menuSub: { fontSize: 13, color: colors.textSub, marginTop: 2 },
  logout: { alignItems: 'center', paddingVertical: 16, marginTop: spacing.sm, backgroundColor: colors.surface, borderRadius: radius.md },
  logoutText: { fontSize: 15, fontWeight: '700', color: colors.textSub },
  devReset: { alignItems: 'center', paddingVertical: 12, marginTop: spacing.sm },
  devResetText: { fontSize: 12, color: colors.textMuted, textDecorationLine: 'underline' },
});

import React, { useState } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Alert, Linking } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { Card } from '../components/ui';
import { registerHospital, addAppointment } from '../api/client';
import { scheduleVisitReminder } from '../utils/notifications';
import { colors, radius, spacing } from '../theme';

// Demo available booking dates (in production, fetched from the hospital server)
const SLOTS = [
  { label: '6/28 (일)', date: new Date(2026, 5, 28, 10, 0) },
  { label: '6/29 (월)', date: new Date(2026, 5, 29, 10, 0) },
  { label: '6/30 (화)', date: new Date(2026, 5, 30, 10, 0) },
];

export default function HospitalDetailScreen({ navigation, route }) {
  const { hospital } = route.params;
  const [registered, setRegistered] = useState(hospital.registered);
  const [selected, setSelected] = useState(null);

  const toggleRegister = async () => {
    const next = !registered;
    setRegistered(next);
    await registerHospital(hospital.id, next);
  };

  const book = async (slot) => {
    setSelected(slot.label);
    await addAppointment(hospital.id, slot.label);   // record the booking on the server
    const id = await scheduleVisitReminder(hospital.name, slot.date);  // local visit reminder
    Alert.alert(
      '방문 예약 완료',
      id
        ? `${slot.label} 방문 알람이 설정되었습니다.`
        : `${slot.label} 예약되었습니다. (알림 권한 미허용)`
    );
  };

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} hitSlop={10}>
          <Ionicons name="arrow-back" size={24} color={colors.text} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>{hospital.name}</Text>
        <View style={{ width: 24 }} />
      </View>

      <ScrollView contentContainerStyle={styles.scroll}>
        {/* Map placeholder */}
        <View style={styles.map}>
          <Ionicons name="location" size={28} color={colors.primary} />
          <Text style={styles.mapText}>지도 연동 예정</Text>
        </View>

        {/* Basic info */}
        <Card style={styles.infoCard}>
          <View style={styles.infoTop}>
            <Text style={styles.dept}>{hospital.dept}</Text>
            {hospital.rating != null && (
              <View style={styles.ratingRow}>
                <Ionicons name="star" size={16} color="#F2B705" />
                <Text style={styles.rating}>{hospital.rating}</Text>
              </View>
            )}
          </View>
          {!!hospital.address && <Text style={styles.address}>{hospital.address}</Text>}
          {!!hospital.doctor && (
            <View style={styles.infoLine}>
              <Ionicons name="person-outline" size={16} color={colors.textSub} />
              <Text style={styles.infoText}>담당의: {hospital.doctor}</Text>
            </View>
          )}
          {!!hospital.phone && (
            <View style={styles.infoLine}>
              <Ionicons name="call-outline" size={16} color={colors.textSub} />
              <Text style={styles.infoText}>{hospital.phone}</Text>
            </View>
          )}
          {!!hospital.placeUrl && (
            <TouchableOpacity style={styles.infoLine} onPress={() => Linking.openURL(hospital.placeUrl)}>
              <Ionicons name="map-outline" size={16} color={colors.primary} />
              <Text style={[styles.infoText, { color: colors.primary }]}>카카오맵에서 보기</Text>
            </TouchableOpacity>
          )}
        </Card>

        {/* Visit booking and reminder */}
        <Card style={styles.bookCard}>
          <View style={styles.bookHeader}>
            <Ionicons name="time-outline" size={18} color={colors.primary} />
            <Text style={styles.bookTitle}>방문 예약 · 알람 설정</Text>
          </View>
          <View style={styles.slotRow}>
            {SLOTS.map((s) => (
              <TouchableOpacity
                key={s.label}
                style={[styles.slot, selected === s.label && styles.slotActive]}
                onPress={() => book(s)}
              >
                <Text style={[styles.slotText, selected === s.label && { color: colors.white }]}>{s.label}</Text>
              </TouchableOpacity>
            ))}
          </View>
        </Card>

        {/* Register / unregister */}
        <TouchableOpacity style={styles.registerBtn} onPress={toggleRegister}>
          <Ionicons
            name={registered ? 'checkmark-circle' : 'add-circle-outline'}
            size={18}
            color={registered ? colors.primary : colors.textSub}
          />
          <Text style={[styles.registerText, registered && { color: colors.primary }]}>
            {registered ? '등록된 병원 · 등록 해제' : '다니는 병원으로 등록'}
          </Text>
        </TouchableOpacity>

        {registered && (
          <TouchableOpacity style={styles.chatBtn} onPress={() => navigation.navigate('채팅')}>
            <Ionicons name="chatbubble-outline" size={18} color={colors.white} />
            <Text style={styles.chatBtnText}>담당 의사와 채팅</Text>
          </TouchableOpacity>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bg },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: spacing.md, paddingVertical: spacing.sm },
  headerTitle: { fontSize: 18, fontWeight: '800', color: colors.text },
  scroll: { padding: spacing.md, paddingBottom: spacing.xl },
  map: {
    height: 120, borderRadius: radius.lg, backgroundColor: colors.primarySoft,
    alignItems: 'center', justifyContent: 'center', gap: 6,
  },
  mapText: { color: colors.textSub, fontSize: 13 },
  infoCard: { marginTop: spacing.md },
  infoTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  dept: { fontSize: 18, fontWeight: '800', color: colors.text },
  ratingRow: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  rating: { fontSize: 15, fontWeight: '700', color: colors.text },
  address: { fontSize: 13, color: colors.textSub, marginTop: 4, marginBottom: 12 },
  infoLine: { flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 8 },
  infoText: { fontSize: 14, color: colors.text },
  bookCard: { marginTop: spacing.md, backgroundColor: colors.surface },
  bookHeader: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 12 },
  bookTitle: { fontSize: 15, fontWeight: '700', color: colors.text },
  slotRow: { flexDirection: 'row', gap: 8 },
  slot: { flex: 1, alignItems: 'center', paddingVertical: 12, borderRadius: radius.md, backgroundColor: colors.white, borderWidth: 1, borderColor: colors.border },
  slotActive: { backgroundColor: colors.primary, borderColor: colors.primary },
  slotText: { fontSize: 13, fontWeight: '600', color: colors.text },
  registerBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, paddingVertical: 16, marginTop: spacing.md, backgroundColor: colors.surface, borderRadius: radius.md },
  registerText: { fontSize: 15, fontWeight: '700', color: colors.textSub },
  chatBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, paddingVertical: 16, marginTop: spacing.sm, backgroundColor: colors.primary, borderRadius: radius.md },
  chatBtnText: { color: colors.white, fontWeight: '700', fontSize: 15 },
});

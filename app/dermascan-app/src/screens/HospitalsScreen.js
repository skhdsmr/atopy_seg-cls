import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import * as Location from 'expo-location';
import { Card, Pill } from '../components/ui';
import KakaoMap from '../components/KakaoMap';
import { getNearbyHospitals } from '../api/client';
import { colors, radius, spacing } from '../theme';

export default function HospitalsScreen({ navigation }) {
  const [hospitals, setHospitals] = useState(null);
  const [coords, setCoords] = useState(null);
  const [error, setError] = useState(null);

  // Acquire location (8s timeout; null on failure -> server uses default coordinates)
  async function getCoords() {
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') return null;
      const pos = await Promise.race([
        Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.Balanced }),
        new Promise((_, rej) => setTimeout(() => rej(new Error('gps timeout')), 8000)),
      ]);
      return { lat: pos.coords.latitude, lng: pos.coords.longitude };
    } catch (e) {
      return null; // still load the list even if location fails
    }
  }

  async function load() {
    setError(null);
    setHospitals(null);
    const c = await getCoords();
    setCoords(c);
    try {
      const list = await getNearbyHospitals(c);
      setHospitals(list);
    } catch (e) {
      setError(e.message || '병원 정보를 불러오지 못했습니다');
    }
  }

  useEffect(() => {
    load();
  }, []);

  const mapCenter = coords || { lat: 37.4979, lng: 127.0276 };

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView contentContainerStyle={styles.scroll}>
        <Text style={styles.h1}>협력 병원</Text>
        <Text style={styles.sub}>현재 위치 기준 가까운 협력 병원</Text>

        {/* Kakao Map */}
        <KakaoMap
          style={styles.map}
          center={mapCenter}
          hospitals={hospitals ?? []}
          onSelect={(id) => {
            const h = (hospitals ?? []).find((x) => x.id === id);
            if (h) navigation.navigate('HospitalDetail', { hospital: h });
          }}
        />

        <View style={styles.listHeader}>
          <Text style={styles.listTitle}>주변 협력 병원</Text>
          <Text style={styles.listCount}>{hospitals ? `${hospitals.length}개 발견` : ''}</Text>
        </View>

        {error ? (
          <View style={styles.errorBox}>
            <Ionicons name="cloud-offline-outline" size={28} color={colors.textMuted} />
            <Text style={styles.errorText}>{error}</Text>
            <Text style={styles.errorHint}>서버가 켜져 있고 같은 망인지 확인하세요</Text>
            <TouchableOpacity style={styles.retryBtn} onPress={load}>
              <Text style={styles.retryText}>다시 시도</Text>
            </TouchableOpacity>
          </View>
        ) : !hospitals ? (
          <View style={{ alignItems: 'center', marginTop: spacing.lg }}>
            <ActivityIndicator color={colors.primary} />
            <Text style={styles.loadingText}>주변 병원 검색 중...</Text>
          </View>
        ) : (
          hospitals.map((h) => (
            <TouchableOpacity key={h.id} activeOpacity={0.8} onPress={() => navigation.navigate('HospitalDetail', { hospital: h })}>
              <Card style={styles.hospCard}>
                <View style={{ flex: 1 }}>
                  <View style={styles.nameRow}>
                    <Text style={styles.hospName}>{h.name}</Text>
                    {h.registered && <Pill label="등록됨" />}
                  </View>
                  <Text style={styles.hospDept}>
                    {h.dept}{h.doctor ? ` · ${h.doctor}` : ''}
                  </Text>
                  <View style={styles.metaRow}>
                    <Ionicons name="location-outline" size={14} color={colors.textSub} />
                    <Text style={styles.meta}>{h.distanceKm != null ? `${h.distanceKm}km` : '거리 정보 없음'}</Text>
                    {h.rating != null && (
                      <>
                        <Ionicons name="star" size={14} color="#F2B705" style={{ marginLeft: 8 }} />
                        <Text style={styles.meta}>{h.rating}</Text>
                      </>
                    )}
                    {!!h.status && (
                      <Text style={[styles.status, { color: h.status === '진료 중' ? colors.mild : colors.textMuted }]}>
                        {'  '}{h.status}
                      </Text>
                    )}
                  </View>
                </View>
                <Ionicons name="chevron-forward" size={20} color={colors.textMuted} />
              </Card>
            </TouchableOpacity>
          ))
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bg },
  scroll: { padding: spacing.md, paddingBottom: spacing.xl },
  h1: { fontSize: 24, fontWeight: '800', color: colors.text },
  sub: { fontSize: 14, color: colors.textSub, marginTop: 4, marginBottom: spacing.md },
  map: { height: 200, borderRadius: radius.lg },
  listHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginTop: spacing.lg, marginBottom: spacing.sm },
  listTitle: { fontSize: 15, fontWeight: '700', color: colors.text },
  listCount: { fontSize: 13, color: colors.textSub },
  hospCard: { flexDirection: 'row', alignItems: 'center', marginBottom: 12 },
  nameRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  hospName: { fontSize: 16, fontWeight: '800', color: colors.text },
  hospDept: { fontSize: 13, color: colors.textSub, marginTop: 4 },
  metaRow: { flexDirection: 'row', alignItems: 'center', marginTop: 8 },
  meta: { fontSize: 13, color: colors.textSub, marginLeft: 3 },
  status: { fontSize: 13, fontWeight: '600' },
  loadingText: { fontSize: 13, color: colors.textSub, marginTop: 8 },
  errorBox: { alignItems: 'center', marginTop: spacing.lg, gap: 6 },
  errorText: { fontSize: 14, fontWeight: '700', color: colors.text, textAlign: 'center' },
  errorHint: { fontSize: 12, color: colors.textMuted, textAlign: 'center' },
  retryBtn: { marginTop: 8, backgroundColor: colors.primary, paddingHorizontal: 20, paddingVertical: 10, borderRadius: radius.pill },
  retryText: { color: colors.white, fontWeight: '700' },
});

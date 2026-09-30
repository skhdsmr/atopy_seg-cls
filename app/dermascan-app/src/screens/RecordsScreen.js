import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Image, Alert } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { Calendar } from 'react-native-calendars';
import * as ImagePicker from 'expo-image-picker';
import { Card, Pill } from '../components/ui';
import { getRecords, addRecord } from '../api/client';
import { colors, radius, spacing, severityColor } from '../theme';

export default function RecordsScreen() {
  const [records, setRecords] = useState({});
  const [selected, setSelected] = useState('2026-06-24');

  useEffect(() => {
    getRecords().then(setRecords);
  }, []);

  // Calendar dots (color by severity)
  const marked = {};
  Object.entries(records).forEach(([date, r]) => {
    marked[date] = { marked: true, dotColor: severityColor(r.severity) };
  });
  marked[selected] = { ...(marked[selected] || {}), selected: true, selectedColor: colors.primary };

  const current = records[selected];

  // Upload today's photo (daily record)
  const uploadPhoto = async () => {
    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) {
      Alert.alert('권한 필요', '사진 보관함 접근 권한을 허용해 주세요.');
      return;
    }
    const res = await ImagePicker.launchImageLibraryAsync({ quality: 0.8 });
    if (res.canceled) return;
    const photo = res.assets[0].uri;
    const next = { ...records, [selected]: { severity: current?.severity ?? '경증', note: current?.note ?? '', photo } };
    setRecords(next);
    await addRecord({ date: selected, ...next[selected] });
  };

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView contentContainerStyle={styles.scroll}>
        <View style={styles.headerRow}>
          <Text style={styles.h1}>증상 기록</Text>
          <TouchableOpacity style={styles.addBtn} onPress={uploadPhoto}>
            <Ionicons name="add" size={22} color={colors.white} />
          </TouchableOpacity>
        </View>

        <Card style={{ padding: 0, overflow: 'hidden' }}>
          <Calendar
            current={selected}
            onDayPress={(d) => setSelected(d.dateString)}
            markedDates={marked}
            theme={{
              todayTextColor: colors.primary,
              arrowColor: colors.primary,
              selectedDayBackgroundColor: colors.primary,
              textMonthFontWeight: '700',
            }}
          />
        </Card>

        {/* Severity legend */}
        <View style={styles.legend}>
          <Legend color={colors.mild} label="경증" />
          <Legend color={colors.moderate} label="중등도" />
          <Legend color={colors.severe} label="중증" />
        </View>

        {/* Record for the selected day */}
        <Card style={styles.recordCard}>
          <View style={styles.recordHead}>
            <Text style={styles.recordDate}>
              {Number(selected.split('-')[1])}월 {Number(selected.split('-')[2])}일 기록
            </Text>
            {current && (
              <Pill
                label={current.severity}
                color={severityColor(current.severity)}
                bg={severityColor(current.severity) + '22'}
              />
            )}
          </View>

          <TouchableOpacity style={styles.photoBox} onPress={uploadPhoto} activeOpacity={0.8}>
            {current?.photo ? (
              <Image source={{ uri: current.photo }} style={styles.photo} />
            ) : (
              <>
                <Ionicons name="camera-outline" size={28} color={colors.textSub} />
                <Text style={styles.photoText}>
                  {current ? '업로드된 사진 보기' : '오늘 사진 업로드'}
                </Text>
              </>
            )}
          </TouchableOpacity>

          {current?.note ? (
            <Text style={styles.note}>{current.note}</Text>
          ) : (
            <Text style={styles.noteEmpty}>이 날의 메모가 없습니다.</Text>
          )}
        </Card>
      </ScrollView>
    </SafeAreaView>
  );
}

function Legend({ color, label }) {
  return (
    <View style={styles.legendItem}>
      <View style={[styles.dot, { backgroundColor: color }]} />
      <Text style={styles.legendText}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bg },
  scroll: { padding: spacing.md, paddingBottom: spacing.xl },
  headerRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: spacing.md },
  h1: { fontSize: 24, fontWeight: '800', color: colors.text },
  addBtn: { width: 40, height: 40, borderRadius: 20, backgroundColor: colors.primary, alignItems: 'center', justifyContent: 'center' },
  legend: { flexDirection: 'row', justifyContent: 'center', gap: 20, marginVertical: spacing.md },
  legendItem: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  dot: { width: 8, height: 8, borderRadius: 4 },
  legendText: { fontSize: 12, color: colors.textSub },
  recordCard: { marginTop: spacing.sm },
  recordHead: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 },
  recordDate: { fontSize: 16, fontWeight: '800', color: colors.text },
  photoBox: {
    height: 120, borderRadius: radius.md, backgroundColor: '#FBEFD9',
    alignItems: 'center', justifyContent: 'center', gap: 6, overflow: 'hidden',
  },
  photo: { width: '100%', height: '100%', resizeMode: 'cover' },
  photoText: { fontSize: 13, color: colors.textSub },
  note: { fontSize: 14, color: colors.text, marginTop: 12 },
  noteEmpty: { fontSize: 13, color: colors.textMuted, marginTop: 12 },
});

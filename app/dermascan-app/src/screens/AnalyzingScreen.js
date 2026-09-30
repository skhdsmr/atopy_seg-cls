import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { segment } from '../ml/segment';
import { classify } from '../ml/classify';
import { colors, radius, spacing } from '../theme';

const STEPS = ['이미지 전처리', '병변 분할 · 중증도 분류', '결과 생성'];

export default function AnalyzingScreen({ navigation, route }) {
  const { imageUri } = route.params;
  const [step, setStep] = useState(0);

  useEffect(() => {
    const timers = [
      setTimeout(() => setStep(1), 500),
      setTimeout(() => setStep(2), 1100),
    ];

    (async () => {
      try {
        // 온디바이스 tflite 분할 추론(서버 호출 없음)
        const { overlayUrl, coverage, detected, timings, memory, imgsz, modelName } = await segment(imageUri);
        // 이어서 분류(중증도·증상) 추론. 분류가 실패해도 분할 결과는 보여준다.
        let cls = null, clsError = null;
        try {
          cls = await classify(imageUri);
        } catch (e) {
          clsError = e?.message || String(e);
        }
        navigation.replace('Result', {
          imageUri, overlayUrl, coverage, detected, timings, memory, imgsz, modelName,
          cls, clsError,
        });
      } catch (e) {
        navigation.replace('Result', { imageUri, error: e.message });
      }
    })();

    return () => timers.forEach(clearTimeout);
  }, []);

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.center}>
        <View style={styles.ring}>
          <ActivityIndicator size="large" color={colors.primary} />
          <Ionicons name="pulse" size={28} color={colors.primary} style={styles.ringIcon} />
        </View>
        <Text style={styles.title}>AI 분석 중</Text>
        <Text style={styles.sub}>병변 패턴을 심층 분석하고 있습니다</Text>

        <View style={styles.steps}>
          {STEPS.map((label, i) => (
            <View key={label} style={styles.stepCard}>
              <Ionicons
                name={i <= step ? 'checkmark-circle' : 'ellipse-outline'}
                size={22}
                color={i <= step ? colors.primary : colors.border}
              />
              <Text style={[styles.stepText, i <= step && { color: colors.text }]}>{label}</Text>
            </View>
          ))}
        </View>

        <View style={styles.progressBar}>
          <View style={[styles.progressFill, { width: `${((step + 1) / STEPS.length) * 100}%` }]} />
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.surface },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', paddingHorizontal: spacing.lg },
  ring: {
    width: 120, height: 120, borderRadius: 60, borderWidth: 4, borderColor: colors.primarySoft,
    alignItems: 'center', justifyContent: 'center', marginBottom: spacing.lg,
  },
  ringIcon: { position: 'absolute' },
  title: { fontSize: 22, fontWeight: '800', color: colors.text },
  sub: { fontSize: 14, color: colors.textSub, marginTop: 6, marginBottom: spacing.lg },
  steps: { width: '100%', gap: 10 },
  stepCard: {
    flexDirection: 'row', alignItems: 'center', gap: 12,
    backgroundColor: colors.card, borderRadius: radius.md, padding: 16,
    borderWidth: 1, borderColor: colors.border,
  },
  stepText: { fontSize: 15, color: colors.textMuted, fontWeight: '600' },
  progressBar: {
    width: '100%', height: 6, borderRadius: 3, backgroundColor: colors.border,
    marginTop: spacing.lg, overflow: 'hidden',
  },
  progressFill: { height: 6, borderRadius: 3, backgroundColor: colors.primary },
});

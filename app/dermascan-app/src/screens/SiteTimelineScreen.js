import React, { useState, useCallback } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, ActivityIndicator, Alert } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { Card, Pill } from '../components/ui';
import { getCaptures, deleteSite } from '../api/client';
import { conditionLabel, sideLabel } from '../data/bodyMap';
import { colors, radius, spacing } from '../theme';

// Registration confidence grades produced on-device (src/ml/register.js).
// 'low' never reaches the server, so it should not appear here.
const GRADE_LABEL = { high: '정합 양호', medium: '정합 보통(참고용)', low: '정합 실패' };
const GRADE_COLOR = { high: colors.mild, medium: colors.moderate, low: colors.severe };

export default function SiteTimelineScreen({ navigation, route }) {
  const { site } = route.params;
  const [captures, setCaptures] = useState(null);

  const load = useCallback(async () => {
    try { setCaptures(await getCaptures(site.id)); } catch { setCaptures([]); }
  }, [site.id]);

  useFocusEffect(useCallback(() => { load(); }, [load]));

  const remove = () => {
    Alert.alert('부위 삭제', '이 부위와 모든 촬영 기록을 삭제할까요?', [
      { text: '취소', style: 'cancel' },
      { text: '삭제', style: 'destructive', onPress: async () => { await deleteSite(site.id); navigation.goBack(); } },
    ]);
  };

  // The anchor is stored as a capture too, but it carries no measurement
  const all = captures || [];
  const hasAnchor = !!site.hasAnchor || all.some((c) => c.kind === 'anchor');
  const sessions = all.filter((c) => c.kind !== 'anchor');

  // Primary metric: lesion area relative to the anchor ROI (fixed denominator,
  // so bars are comparable across sessions). Medium-confidence points are drawn
  // faded — they count, but they are explicitly "참고용".
  const areaPts = sessions.filter((c) => c.areaRatio != null);
  const aMax = Math.max(...areaPts.map((c) => c.areaRatio), 0.0001);
  const latestSeverity = [...sessions].reverse()
    .map((c) => c.severity?.predictions?.find((p) => p.task === 'severity'))
    .find(Boolean);

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} hitSlop={10}>
          <Ionicons name="arrow-back" size={24} color={colors.text} />
        </TouchableOpacity>
        <Text style={styles.headerTitle} numberOfLines={1}>
          {site.bodyPart}{site.side ? ` (${sideLabel(site.side)})` : ''}
        </Text>
        <TouchableOpacity onPress={remove} hitSlop={10}>
          <Ionicons name="trash-outline" size={22} color={colors.textSub} />
        </TouchableOpacity>
      </View>

      <ScrollView contentContainerStyle={styles.scroll}>
        <View style={styles.metaRow}>
          <Pill label={conditionLabel(site.condition)} />
          {!!site.label && <Text style={styles.label}>{site.label}</Text>}
        </View>

        {/* Relative area trend (primary metric) */}
        {areaPts.length >= 2 && (
          <Card style={styles.chartCard}>
            <Text style={styles.chartTitle}>병변 면적 추세 (기준 영역 대비)</Text>
            <View style={styles.chart}>
              {areaPts.map((c, i) => (
                <View
                  key={i}
                  style={[
                    styles.bar,
                    { height: 8 + (c.areaRatio / aMax) * 70 },
                    c.alignGrade === 'medium' && styles.barWeak,
                  ]}
                />
              ))}
            </View>
            <View style={styles.chartFooter}>
              <Text style={styles.chartHint}>
                최근 {(areaPts[areaPts.length - 1].areaRatio * 100).toFixed(1)}%
                {' · '}옅은 막대는 정합 신뢰도 보통(참고용)
              </Text>
              {latestSeverity && (
                <Pill label={`IGA ${latestSeverity.label}`} />
              )}
            </View>
          </Card>
        )}

        {/* Capture list */}
        {!captures ? (
          <ActivityIndicator color={colors.primary} style={{ marginTop: spacing.lg }} />
        ) : !hasAnchor ? (
          <View style={styles.empty}>
            <Ionicons name="scan-outline" size={36} color={colors.textMuted} />
            <Text style={styles.emptyText}>
              기준 사진이 없습니다.{'\n'}부위 전체가 보이는 원거리 사진을 한 번 등록하세요.
            </Text>
          </View>
        ) : sessions.length === 0 ? (
          <View style={styles.empty}>
            <Ionicons name="camera-outline" size={36} color={colors.textMuted} />
            <Text style={styles.emptyText}>
              기준 사진이 등록되었습니다.{'\n'}이제 근거리 촬영만 하면 됩니다.
            </Text>
          </View>
        ) : (
          sessions.slice().reverse().map((c) => (
            <Card key={c.id} style={styles.capCard}>
              <View style={styles.capTop}>
                <Text style={styles.capDate}>{c.date}</Text>
                <Pill
                  label={GRADE_LABEL[c.alignGrade] || '정합 정보 없음'}
                  color={GRADE_COLOR[c.alignGrade] || colors.textMuted}
                  bg={(GRADE_COLOR[c.alignGrade] || colors.textMuted) + '22'}
                />
              </View>
              <View style={styles.capMetrics}>
                <Metric label="면적" value={c.areaRatio != null ? `${(c.areaRatio * 100).toFixed(1)}%` : '-'} />
                <Metric label="홍반" value={c.erythema ?? '-'} />
                <Metric label="일치점" value={c.alignInliers ?? '-'} />
                <Metric label="오차(px)" value={c.alignReproj != null ? c.alignReproj.toFixed(1) : '-'} />
              </View>
              {c.alignReasons?.length > 0 && (
                <Text style={styles.capReason}>{c.alignReasons.join(' · ')}</Text>
              )}
              {!c.qualityOk && c.reasons?.length > 0 && (
                <Text style={styles.capReason}>{c.reasons.join(' · ')}</Text>
              )}
            </Card>
          ))
        )}
      </ScrollView>

      {/* Capture button — sessions are only possible once the anchor exists */}
      <View style={styles.footer}>
        <TouchableOpacity
          style={styles.cta}
          onPress={() => navigation.navigate(hasAnchor ? 'GuidedCapture' : 'AnchorCapture', { site })}
        >
          <Ionicons name={hasAnchor ? 'camera' : 'scan'} size={20} color={colors.white} />
          <Text style={styles.ctaText}>{hasAnchor ? '이 부위 촬영 (근거리)' : '기준 사진 등록하기'}</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const Metric = ({ label, value }) => (
  <View style={styles.metric}>
    <Text style={styles.metricValue}>{value}</Text>
    <Text style={styles.metricLabel}>{label}</Text>
  </View>
);

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bg },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: spacing.md, paddingVertical: spacing.sm, gap: 12 },
  headerTitle: { flex: 1, fontSize: 18, fontWeight: '800', color: colors.text },
  scroll: { padding: spacing.md, paddingBottom: 100 },
  metaRow: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: spacing.md },
  label: { fontSize: 14, color: colors.textSub },
  chartCard: { marginBottom: spacing.md },
  chartTitle: { fontSize: 14, fontWeight: '700', color: colors.text, marginBottom: 12 },
  chart: { flexDirection: 'row', alignItems: 'flex-end', gap: 6, height: 80 },
  bar: { flex: 1, maxWidth: 22, backgroundColor: colors.primary, borderRadius: 4 },
  barWeak: { opacity: 0.4 },
  chartFooter: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 8, marginTop: 8 },
  chartHint: { flex: 1, fontSize: 11, color: colors.textMuted },
  empty: { alignItems: 'center', marginTop: spacing.xl, gap: 8 },
  emptyText: { fontSize: 13, color: colors.textSub, textAlign: 'center', lineHeight: 20 },
  capCard: { marginBottom: 12 },
  capTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 },
  capDate: { fontSize: 14, fontWeight: '700', color: colors.text },
  capMetrics: { flexDirection: 'row', gap: 10 },
  metric: { flex: 1, backgroundColor: colors.surface, borderRadius: radius.sm, paddingVertical: 10, alignItems: 'center' },
  metricValue: { fontSize: 15, fontWeight: '800', color: colors.text },
  metricLabel: { fontSize: 11, color: colors.textSub, marginTop: 2 },
  capReason: { fontSize: 12, color: colors.moderate, marginTop: 8 },
  footer: { position: 'absolute', left: 0, right: 0, bottom: 0, padding: spacing.md, backgroundColor: colors.bg, borderTopWidth: 1, borderTopColor: colors.border },
  cta: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, backgroundColor: colors.primary, borderRadius: radius.md, paddingVertical: 16 },
  ctaText: { color: colors.white, fontSize: 16, fontWeight: '700' },
});

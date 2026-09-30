import React, { useState } from 'react';
import { View, Text, StyleSheet, ScrollView, Image, TouchableOpacity, Modal, Pressable } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { PrimaryButton } from '../components/ui';
import { colors, radius, spacing } from '../theme';

const ms = (v) => (v == null ? '—' : `${v.toFixed(0)} ms`);
const mb = (v) => (v == null ? '—' : `${v.toFixed(0)} MB`);

// 등급(라벨) -> 한글 단계명 + 색. severity(IGA)/증상 공통.
const GRADE = {
  'None':         { ko: '없음', color: colors.textSub },
  'Clear':        { ko: '정상 단계', color: colors.mild },
  'Almost Clear': { ko: '거의 정상', color: colors.mild },
  'Mild':         { ko: '경미한 단계', color: '#E3B341' },
  'Moderate':     { ko: '중등도 단계', color: colors.moderate },
  'Severe':       { ko: '심한 단계', color: colors.severe },
};
const gradeInfo = (label) => GRADE[label] || { ko: label, color: colors.textSub };
const pctOf = (v) => `${Math.round((v || 0) * 100)}%`;

// 등급 배지(라벨 + 한글 단계)
function GradeBadge({ label }) {
  const g = gradeInfo(label);
  return (
    <View style={[styles.badge, { backgroundColor: g.color }]}>
      <Text style={styles.badgeLabel}>{label}</Text>
      <Text style={styles.badgeKo}>{g.ko}</Text>
    </View>
  );
}

// 등급 구간 표시 막대(예측 등급만 색칠). labels 중 predIndex 를 강조.
function GradeSegments({ labels, predIndex }) {
  return (
    <View style={styles.segRow}>
      {labels.map((lab, i) => (
        <View key={lab} style={styles.segCol}>
          <View style={[styles.segBar, { backgroundColor: i === predIndex ? gradeInfo(lab).color : colors.border }]} />
          <Text style={[styles.segLabel, i === predIndex && { color: colors.text, fontWeight: '700' }]} numberOfLines={1}>{lab}</Text>
        </View>
      ))}
    </View>
  );
}

// streamlit prob_chart 를 흉내낸 세로 막대그래프(민트 막대). 예측 등급은 진하게.
function ProbChart({ labels, probs, predIndex }) {
  const top = Math.max(0.0001, ...probs);
  const H = 130;
  return (
    <View style={styles.chartRow}>
      {labels.map((lab, i) => {
        const h = Math.max(3, (probs[i] / top) * H);
        const on = i === predIndex;
        return (
          <View key={lab} style={styles.chartCol}>
            <Text style={[styles.chartPct, on && { color: colors.text, fontWeight: '800' }]}>{Math.round(probs[i] * 100)}%</Text>
            <View style={styles.chartBarArea}>
              <View style={[styles.chartBar, { height: h, backgroundColor: on ? '#14B8A6' : '#5EEAD4' }]} />
            </View>
            <Text style={[styles.chartLabel, on && { color: colors.text, fontWeight: '700' }]} numberOfLines={2}>{lab}</Text>
          </View>
        );
      })}
    </View>
  );
}

export default function ResultScreen({ navigation, route }) {
  const { imageUri, overlayUrl, coverage = 0, detected = false, error,
    timings, memory, imgsz, modelName, cls, clsError } = route.params;
  const pct = Math.round(coverage * 1000) / 10; // 소수 첫째자리 %

  // 클릭한 태스크의 확률분포 모달
  const [chart, setChart] = useState(null); // {titleKo, labels, probs, index}

  // 이미지 확대 모달 {base, overlay, caption}
  const [zoom, setZoom] = useState(null);

  const memAvailable = memory && memory.afterInferMB != null;
  const signed = (v) => (v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(0)} MB`);

  // 분할 + 분류 타이밍 합산: 전처리 / 추론·분할 / 추론·분류 / 후처리 / 합계
  const segT = timings || {};
  const clsT = cls?.timings || {};
  const add = (...xs) => { const v = xs.filter((n) => n != null); return v.length ? v.reduce((a, b) => a + b, 0) : null; };
  const preMs = add(segT.preprocessMs, clsT.preprocessMs);      // 두 모델 각자 리사이즈+정규화
  const segInferMs = segT.inferenceMs ?? null;                  // 분할 model.run
  const clsInferMs = clsT.inferenceMs ?? null;                  // 분류 model.run
  const postMs = add(segT.postprocessMs, clsT.postprocessMs);   // 마스크 오버레이 + 등급 파싱
  const pipelineMs = add(segT.preprocessMs, clsT.preprocessMs, segT.inferenceMs, clsT.inferenceMs, segT.postprocessMs, clsT.postprocessMs);

  // 분류 결과 정리: severity(IGA) + None 이 아닌 증상만
  const severity = cls?.severity || cls?.predictions?.find((p) => p.task === 'severity');
  const symptoms = (cls?.predictions || []).filter((p) => p.task !== 'severity' && p.index > 0);

  const openChart = (p) => setChart({ titleKo: p.titleKo || p.title, labels: p.labels, probs: p.probs, index: p.index });

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView contentContainerStyle={styles.scroll}>
        {/* Header */}
        <View style={styles.headerRow}>
          <View>
            <Text style={styles.h1}>분석 결과</Text>
            <Text style={styles.date}>온디바이스 AI · 병변 분할 + 중증도 분류</Text>
          </View>
          <TouchableOpacity style={styles.retake} onPress={() => navigation.popToTop()}>
            <Text style={styles.retakeText}>재촬영</Text>
          </TouchableOpacity>
        </View>

        {/* 원본 + 병변 예측 영역 (좌우 2장) — 탭하면 전체화면 확대 */}
        <View style={styles.imgRow}>
          <View style={styles.imgCol}>
            <TouchableOpacity
              activeOpacity={0.85}
              onPress={() => setZoom({ base: imageUri, overlay: null, caption: '원본 이미지' })}
            >
              <View style={styles.imgBox}>
                <Image source={{ uri: imageUri }} style={styles.previewImg} resizeMode="stretch" />
                <View style={styles.zoomBadge}>
                  <Ionicons name="expand" size={13} color={colors.white} />
                </View>
              </View>
            </TouchableOpacity>
            <Text style={styles.imgCaption}>원본 이미지</Text>
          </View>
          <View style={styles.imgCol}>
            <TouchableOpacity
              activeOpacity={0.85}
              onPress={() => setZoom({ base: imageUri, overlay: overlayUrl || null, caption: '병변 예측 영역' })}
            >
              <View style={styles.imgBox}>
                <Image source={{ uri: imageUri }} style={styles.previewImg} resizeMode="stretch" />
                {overlayUrl ? (
                  <Image source={{ uri: overlayUrl }} style={[StyleSheet.absoluteFill, { width: undefined, height: undefined }]} resizeMode="stretch" />
                ) : null}
                <View style={styles.zoomBadge}>
                  <Ionicons name="expand" size={13} color={colors.white} />
                </View>
              </View>
            </TouchableOpacity>
            <View style={styles.captionRow}>
              <View style={styles.predDot} />
              <Text style={[styles.imgCaption, { marginTop: 0 }]}>병변 예측 영역</Text>
            </View>
          </View>
        </View>

        {error ? (
          <Text style={styles.error}>분석 실패: {error}</Text>
        ) : (
          <>
            {/* 전체 중증도 (IGA) — 탭하면 확률분포 */}
            {severity && (
              <>
                <Text style={styles.sectionTitle}>전체 중증도 (IGA)</Text>
                <TouchableOpacity activeOpacity={0.8} onPress={() => openChart(severity)} style={styles.igaCard}>
                  <View style={styles.igaTopRow}>
                    <GradeBadge label={severity.label} />
                    <View style={{ alignItems: 'flex-end' }}>
                      <Text style={styles.igaPct}>{pctOf(severity.confidence)}</Text>
                      <Text style={styles.igaPctSub}>예측 확률</Text>
                    </View>
                  </View>
                  <GradeSegments labels={severity.labels} predIndex={severity.index} />
                  <View style={styles.tapHintRow}>
                    <Ionicons name="bar-chart-outline" size={13} color={colors.textMuted} />
                    <Text style={styles.tapHint}>탭하여 등급별 확률 분포 보기</Text>
                  </View>
                </TouchableOpacity>
              </>
            )}

            {/* 병변 분석 — None 이 아닌 증상만 */}
            {cls && (
              <>
                <Text style={styles.sectionTitle}>병변 분석</Text>
                <Text style={styles.subCount}>감지된 병변 {symptoms.length}개</Text>
                {symptoms.length === 0 ? (
                  <View style={styles.emptyCard}>
                    <Text style={styles.emptyText}>감지된 증상이 없습니다 (모든 증상 등급이 None).</Text>
                  </View>
                ) : (
                  symptoms.map((p) => (
                    <TouchableOpacity key={p.task} activeOpacity={0.8} onPress={() => openChart(p)} style={styles.symCard}>
                      <View style={styles.symTopRow}>
                        <View style={styles.symLabelRow}>
                          <View style={[styles.symDot, { backgroundColor: gradeInfo(p.label).color }]} />
                          <Text style={styles.symTitle}>{p.titleKo || p.title}</Text>
                          <GradeBadge label={p.label} />
                        </View>
                        <Text style={styles.symPct}>{pctOf(p.confidence)}</Text>
                      </View>
                      {/* 증상은 None 제외한 등급 구간만 표시 */}
                      <GradeSegments labels={p.labels.slice(1)} predIndex={p.index - 1} />
                    </TouchableOpacity>
                  ))
                )}
                {clsError && (
                  <Text style={styles.clsErr}>중증도·증상 분류를 불러오지 못했습니다: {clsError}</Text>
                )}
              </>
            )}
            {!cls && clsError && (
              <Text style={styles.clsErr}>중증도·증상 분류 실패: {clsError}</Text>
            )}

            {/* 병변 영역(면적) */}
            <Text style={styles.sectionTitle}>병변 영역 분석</Text>
            <View style={styles.metricCard}>
              <View style={styles.metricRow}>
                <View style={styles.metricLabelRow}>
                  <Ionicons name="color-fill" size={18} color={colors.primary} />
                  <Text style={styles.metricLabel}>추정 병변 면적 비율</Text>
                </View>
                <Text style={styles.metricValue}>{pct}%</Text>
              </View>
              <View style={styles.barTrack}>
                <View style={[styles.barFill, { width: `${Math.min(pct, 100)}%` }]} />
              </View>
              <Text style={styles.metricHint}>
                {detected
                  ? '붉게 표시된 영역이 모델이 병변으로 추정한 부위입니다.'
                  : '이미지에서 병변으로 추정되는 영역을 찾지 못했습니다.'}
              </Text>
            </View>
          </>
        )}

        {/* Disclaimer */}
        <View style={styles.notice}>
          <Ionicons name="alert-circle" size={18} color={colors.warning} />
          <Text style={styles.noticeText}>
            본 결과는 AI 참고용이며 의료 전문가의 진단을 대체하지 않습니다.
          </Text>
        </View>

        {/* 온디바이스 성능 계측 */}
        {!error && (
          <>
            <Text style={styles.sectionTitle}>온디바이스 성능</Text>
            <View style={styles.perfModelsRow}>
              {modelName ? <Text style={styles.perfModelTag}>분할 · {modelName}{imgsz ? ` (${imgsz}²)` : ''}</Text> : null}
              {cls?.modelName ? <Text style={styles.perfModelTag}>분류 · {cls.modelName}{cls.imgsz ? ` (${cls.imgsz}²)` : ''}</Text> : null}
            </View>
            <View style={styles.perfCard}>
              <View style={styles.perfRow}><Text style={styles.perfLabel}>전처리(리사이즈+정규화)</Text><Text style={styles.perfVal}>{ms(preMs)}</Text></View>
              <View style={styles.perfRow}><Text style={styles.perfLabel}>추론 · 분할(model.run)</Text><Text style={[styles.perfVal, styles.perfHi]}>{ms(segInferMs)}</Text></View>
              <View style={styles.perfRow}><Text style={styles.perfLabel}>추론 · 분류(model.run)</Text><Text style={[styles.perfVal, styles.perfHi]}>{ms(clsInferMs)}</Text></View>
              <View style={styles.perfRow}><Text style={styles.perfLabel}>후처리(마스크+등급 파싱)</Text><Text style={styles.perfVal}>{ms(postMs)}</Text></View>
              <View style={[styles.perfRow, styles.perfDivider]}><Text style={[styles.perfLabel, { fontWeight: '800' }]}>합계</Text><Text style={[styles.perfVal, { fontWeight: '800' }]}>{ms(pipelineMs)}</Text></View>
              {memAvailable ? (
                <>
                  <View style={[styles.perfRow, styles.perfDivider]}>
                    <Text style={[styles.perfLabel, { fontWeight: '800' }]}>메모리 (PSS)</Text>
                    <Text style={styles.perfValMuted}>기준 {mb(memory.baseMB)}</Text>
                  </View>
                  <View style={styles.perfRow}>
                    <Text style={styles.perfLabel}>모델 로드 증가 {memory.freshLoad ? '' : '(이미 로드됨)'}</Text>
                    <Text style={[styles.perfVal, memory.freshLoad && styles.perfHi]}>{signed(memory.loadDeltaMB)}</Text>
                  </View>
                  <View style={styles.perfRow}>
                    <Text style={styles.perfLabel}>추론(활성화) 증가</Text>
                    <Text style={styles.perfVal}>{signed(memory.inferDeltaMB)}</Text>
                  </View>
                  <View style={styles.perfRow}>
                    <Text style={styles.perfLabel}>총 증가 / 추론 후 총량</Text>
                    <Text style={styles.perfVal}>{signed(memory.totalDeltaMB)} / {mb(memory.afterInferMB)}</Text>
                  </View>
                  {!memory.freshLoad && (
                    <Text style={styles.perfHint}>모델이 이전 진단에서 이미 로드돼 "로드 증가"가 0입니다. 정확한 로드 증가량은 앱을 완전히 종료 후 재실행하여 첫 진단에서 측정하세요.</Text>
                  )}
                </>
              ) : (
                <>
                  <View style={styles.perfRow}><Text style={styles.perfLabel}>메모리</Text><Text style={styles.perfVal}>재빌드 필요</Text></View>
                  <Text style={styles.perfHint}>메모리 수치는 react-native-device-info 네이티브 모듈이 필요합니다 → 한 번 재빌드(eas build) 후 표시됩니다.</Text>
                </>
              )}
            </View>
          </>
        )}

        <PrimaryButton
          title="가까운 협력 병원 보기"
          icon={<Ionicons name="location" size={18} color={colors.white} />}
          style={{ marginTop: spacing.md }}
          onPress={() => navigation.navigate('병원')}
        />
      </ScrollView>

      {/* 확률분포 모달 (streamlit 막대그래프 스타일) */}
      <Modal visible={!!chart} transparent animationType="fade" onRequestClose={() => setChart(null)}>
        <Pressable style={styles.modalOverlay} onPress={() => setChart(null)}>
          <Pressable style={styles.modalCard} onPress={() => {}}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>{chart?.titleKo} · 등급별 확률</Text>
              <TouchableOpacity onPress={() => setChart(null)} hitSlop={10}>
                <Ionicons name="close" size={22} color={colors.textSub} />
              </TouchableOpacity>
            </View>
            {chart && <ProbChart labels={chart.labels} probs={chart.probs} predIndex={chart.index} />}
            <Text style={styles.modalHint}>가장 높은 확률의 등급이 예측 결과입니다.</Text>
          </Pressable>
        </Pressable>
      </Modal>

      {/* 이미지 확대 모달 (전체화면) */}
      <Modal visible={!!zoom} transparent animationType="fade" onRequestClose={() => setZoom(null)}>
        <Pressable style={styles.zoomOverlay} onPress={() => setZoom(null)}>
          <View style={styles.zoomTopBar}>
            <Text style={styles.zoomCaption}>{zoom?.caption}</Text>
            <TouchableOpacity onPress={() => setZoom(null)} hitSlop={12}>
              <Ionicons name="close" size={28} color={colors.white} />
            </TouchableOpacity>
          </View>
          {zoom && (
            <View style={styles.zoomImageWrap}>
              <Image source={{ uri: zoom.base }} style={styles.zoomImage} resizeMode="contain" />
              {zoom.overlay ? (
                <Image source={{ uri: zoom.overlay }} style={styles.zoomImage} resizeMode="contain" />
              ) : null}
            </View>
          )}
          <Text style={styles.zoomHint}>화면을 탭하면 닫힙니다</Text>
        </Pressable>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bg },
  scroll: { padding: spacing.md, paddingBottom: spacing.xl },
  headerRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start' },
  h1: { fontSize: 24, fontWeight: '800', color: colors.text },
  date: { fontSize: 13, color: colors.textSub, marginTop: 4 },
  retake: { backgroundColor: colors.primarySoft, paddingHorizontal: 14, paddingVertical: 8, borderRadius: radius.pill },
  retakeText: { color: colors.primary, fontWeight: '700', fontSize: 13 },

  // 좌우 2장 이미지
  imgRow: { flexDirection: 'row', gap: spacing.sm, marginTop: spacing.md },
  imgCol: { flex: 1 },
  imgBox: { borderRadius: radius.lg, overflow: 'hidden', backgroundColor: colors.surface },
  previewImg: { width: '100%', height: 170 },
  zoomBadge: {
    position: 'absolute', right: 6, bottom: 6,
    width: 24, height: 24, borderRadius: 12,
    backgroundColor: 'rgba(0,0,0,0.5)', alignItems: 'center', justifyContent: 'center',
  },
  predDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: colors.severe },
  captionRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, marginTop: 6 },
  imgCaption: { textAlign: 'center', fontSize: 12, color: colors.textSub, marginTop: 6 },

  sectionTitle: { fontSize: 16, fontWeight: '800', color: colors.text, marginTop: spacing.lg, marginBottom: spacing.sm },
  subCount: { fontSize: 13, color: colors.textSub, marginTop: -4, marginBottom: spacing.sm },

  // 등급 배지
  badge: { flexDirection: 'row', alignItems: 'center', gap: 6, paddingHorizontal: 12, paddingVertical: 7, borderRadius: radius.md },
  badgeLabel: { color: colors.white, fontSize: 14, fontWeight: '800' },
  badgeKo: { color: colors.white, fontSize: 11, fontWeight: '600', opacity: 0.9 },

  // IGA 카드
  igaCard: { borderRadius: radius.lg, padding: spacing.md, borderWidth: 1, borderColor: colors.border, backgroundColor: colors.card },
  igaTopRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  igaPct: { fontSize: 26, fontWeight: '800', color: colors.text },
  igaPctSub: { fontSize: 12, color: colors.textSub, marginTop: 2 },

  // 등급 구간 막대
  segRow: { flexDirection: 'row', gap: 6, marginTop: 16 },
  segCol: { flex: 1, alignItems: 'center' },
  segBar: { height: 6, width: '100%', borderRadius: 3, backgroundColor: colors.border },
  segLabel: { fontSize: 10, color: colors.textMuted, marginTop: 6, textAlign: 'center' },

  tapHintRow: { flexDirection: 'row', alignItems: 'center', gap: 4, marginTop: 12 },
  tapHint: { fontSize: 12, color: colors.textMuted },

  // 증상 카드
  symCard: { borderRadius: radius.lg, padding: spacing.md, borderWidth: 1, borderColor: colors.border, backgroundColor: colors.card, marginBottom: spacing.sm },
  symTopRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  symLabelRow: { flexDirection: 'row', alignItems: 'center', gap: 8, flexShrink: 1 },
  symDot: { width: 9, height: 9, borderRadius: 5 },
  symTitle: { fontSize: 16, fontWeight: '800', color: colors.text },
  symPct: { fontSize: 18, fontWeight: '800', color: colors.text, fontVariant: ['tabular-nums'] },

  emptyCard: { borderRadius: radius.lg, padding: spacing.md, borderWidth: 1, borderColor: colors.border, backgroundColor: colors.card },
  emptyText: { fontSize: 13, color: colors.textSub },
  clsErr: { fontSize: 12, color: colors.severe, marginTop: 6 },

  // 확률 모달
  modalOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.45)', justifyContent: 'center', padding: spacing.lg },
  modalCard: { backgroundColor: colors.card, borderRadius: radius.xl, padding: spacing.lg },
  modalHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: spacing.md },
  modalTitle: { fontSize: 17, fontWeight: '800', color: colors.text, flexShrink: 1 },
  modalHint: { fontSize: 12, color: colors.textMuted, marginTop: spacing.md, textAlign: 'center' },
  chartRow: { flexDirection: 'row', alignItems: 'flex-end', gap: 8, minHeight: 180 },
  chartCol: { flex: 1, alignItems: 'center' },
  chartPct: { fontSize: 11, color: colors.textSub, marginBottom: 4, fontVariant: ['tabular-nums'] },
  chartBarArea: { height: 130, justifyContent: 'flex-end', width: '70%' },
  chartBar: { width: '100%', borderTopLeftRadius: 4, borderTopRightRadius: 4 },
  // 2줄 라벨(Almost Clear)도 같은 높이를 차지하게 고정 -> 막대 baseline 정렬 유지
  chartLabel: { fontSize: 11, lineHeight: 13, height: 26, color: colors.textMuted, marginTop: 6, textAlign: 'center' },

  // 이미지 확대 모달
  zoomOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.92)', justifyContent: 'center', alignItems: 'center' },
  zoomTopBar: {
    position: 'absolute', top: 0, left: 0, right: 0,
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: spacing.lg, paddingTop: spacing.xl, paddingBottom: spacing.md,
  },
  zoomCaption: { color: colors.white, fontSize: 16, fontWeight: '800' },
  zoomImageWrap: { width: '100%', height: '75%', justifyContent: 'center', alignItems: 'center' },
  zoomImage: { ...StyleSheet.absoluteFillObject, width: '100%', height: '100%' },
  zoomHint: { position: 'absolute', bottom: spacing.xl, color: 'rgba(255,255,255,0.6)', fontSize: 13 },

  metricCard: { borderRadius: radius.lg, padding: spacing.md, borderWidth: 1, borderColor: colors.border, backgroundColor: colors.card },
  metricRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  metricLabelRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  metricLabel: { fontSize: 15, fontWeight: '700', color: colors.text },
  metricValue: { fontSize: 20, fontWeight: '800', color: colors.primary },
  barTrack: { height: 8, borderRadius: 4, backgroundColor: colors.border, marginTop: 12, overflow: 'hidden' },
  barFill: { height: 8, borderRadius: 4, backgroundColor: colors.primary },
  metricHint: { fontSize: 13, color: colors.textSub, marginTop: 12 },
  error: { textAlign: 'center', color: colors.severe, marginVertical: spacing.lg },

  notice: {
    flexDirection: 'row', gap: 8, alignItems: 'center', backgroundColor: colors.warningBg,
    borderRadius: radius.md, padding: 12, marginTop: spacing.md,
  },
  noticeText: { flex: 1, fontSize: 13, color: '#8A6D1E' },

  perfModelsRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginBottom: spacing.sm, marginTop: -2 },
  perfModelTag: { fontSize: 11, fontWeight: '700', color: colors.textSub, backgroundColor: colors.surface, borderRadius: radius.sm, paddingHorizontal: 8, paddingVertical: 3 },
  perfCard: { borderRadius: radius.lg, padding: spacing.md, borderWidth: 1, borderColor: colors.border, backgroundColor: colors.card },
  perfRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 5 },
  perfLabel: { fontSize: 13, color: colors.textSub, flex: 1 },
  perfVal: { fontSize: 14, fontWeight: '700', color: colors.text, fontVariant: ['tabular-nums'] },
  perfValMuted: { fontSize: 13, fontWeight: '600', color: colors.textSub, fontVariant: ['tabular-nums'] },
  perfHi: { color: colors.primary },
  perfDivider: { borderTopWidth: 1, borderTopColor: colors.border, marginTop: 4, paddingTop: 8 },
  perfHint: { fontSize: 12, color: colors.textMuted, marginTop: 8, lineHeight: 17 },
});

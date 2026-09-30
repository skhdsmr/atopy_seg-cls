import React, { useState, useRef, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Image, ActivityIndicator, ScrollView } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import { getAnchorCrop, addCapture, checkQuality } from '../api/client';
import { segmentMask } from '../ml/segment';
import { classify } from '../ml/classify';
import { registerToAnchor, toCaptureMetrics } from '../ml/register';
import { sideLabel } from '../data/bodyMap';
import { colors, radius, spacing } from '../theme';

// Live preview check interval
const SAMPLE_MS = 2200;

// Registration confidence -> what we do with the measurement.
// low is never uploaded: a mis-registered frame poisons the area trend permanently,
// and an empty point is cheaper to explain than a wrong one.
const GRADE_META = {
  high: { icon: 'checkmark-circle', color: colors.mild, title: '정합 성공', note: '추세에 정상 반영됩니다' },
  medium: { icon: 'alert-circle', color: colors.moderate, title: '정합 보통', note: '참고용으로 반영됩니다 (가중치 낮음)' },
  low: { icon: 'close-circle', color: colors.severe, title: '정합 실패', note: '기록하지 않았습니다 — 재촬영해주세요' },
};

export default function GuidedCaptureScreen({ navigation, route }) {
  const { site } = route.params;
  const [permission, requestPermission] = useCameraPermissions();
  const camRef = useRef(null);

  const [ghost, setGhost] = useState(null);     // anchor ROI crop, shown as the framing guide
  const [showGhost, setShowGhost] = useState(true);
  const [facing, setFacing] = useState('back'); // front/back camera
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState(null);     // on-device pipeline progress label
  const [result, setResult] = useState(null);   // capture analysis result
  const [hints, setHints] = useState([]);        // live capture guidance (brightness/exposure)
  const [checked, setChecked] = useState(false); // whether the server check has succeeded at least once
  const [anchorErr, setAnchorErr] = useState(null);

  const anchorRef = useRef(null);   // GET /sites/{id}/anchor/crop payload (matching target)
  const sampling = useRef(false);   // prevent concurrent sampling
  const busyRef = useRef(false);    // synchronous guard during shutter capture (faster than state)
  const alive = useRef(true);       // prevent setState after unmount
  const sampleRef = useRef(() => {}); // so the interval always calls the latest sample

  useEffect(() => {
    if (!permission?.granted) requestPermission();
    // Fetch the matching target once per session, not once per shot
    getAnchorCrop(site.id)
      .then((crop) => {
        anchorRef.current = crop;
        setGhost(crop.cropUrl);
      })
      .catch((e) => setAnchorErr(e?.message || String(e)));
  }, []);

  // Live preview check: periodic low-res snapshot -> brightness/exposure (server) + aspect ratio (local) guidance
  // The interval is set once on mount and always calls the latest sample (latest state closure).
  useEffect(() => {
    alive.current = true;
    const id = setInterval(() => sampleRef.current(), SAMPLE_MS);
    return () => { alive.current = false; clearInterval(id); };
  }, []);

  const sample = async () => {
    // Skip the check while capturing, the result sheet is open, or permission is missing
    if (sampling.current || busyRef.current || busy || result || !camRef.current || !permission?.granted) return;
    sampling.current = true;
    try {
      // Not using skipProcessing: so rotation-corrected width/height stay consistent.
      const shot = await camRef.current.takePictureAsync({ quality: 0.3, shutterSound: false });
      const next = [];

      // Brightness/exposure: downscale and check on the server (brightness is resolution-independent).
      // Framing/scale is no longer checked here — registration measures it exactly (relScale)
      // and reports it back through the confidence grade.
      try {
        const ctx = ImageManipulator.manipulate(shot.uri).resize({ width: 256 });
        const img = await ctx.renderAsync();
        const small = await img.saveAsync({ compress: 0.4, format: SaveFormat.JPEG });
        const q = await checkQuality(small.uri);
        if (alive.current) setChecked(true);
        if (q?.reasons?.length) next.push(...q.reasons.map(brightnessHint));
      } catch { /* silently ignore network/check failures (retry next cycle) */ }

      if (alive.current) setHints(next);
    } catch {
      /* ignore snapshot failures */
    } finally {
      sampling.current = false;
    }
  };

  // Capture -> segment -> register -> (severity) -> upload.
  // Everything except the quality gate runs on-device; the server only stores the result.
  const shoot = async () => {
    if (!camRef.current || busy || busyRef.current) return;
    if (!anchorRef.current) {
      setResult({ error: anchorErr || '기준 사진을 불러오는 중입니다. 잠시 후 다시 시도하세요.' });
      return;
    }
    busyRef.current = true; // prevent capturing at the same time as an in-progress preview sample
    setBusy(true);
    setHints([]);

    let pic;
    try {
      pic = await camRef.current.takePictureAsync({ quality: 0.7 });
    } catch (e) {
      setResult({ error: '카메라 촬영 실패: ' + (e?.message || String(e)) });
      setBusy(false);
      busyRef.current = false;
      return;
    }

    try {
      setStage('병변 분할 중…');
      const seg = await segmentMask(pic.uri);

      setStage('기준 사진에 정합 중…');
      const reg = await registerToAnchor({
        anchor: anchorRef.current,
        near: { uri: pic.uri, width: pic.width, height: pic.height },
        seg,
      });

      // Registration failed -> do not record. Uploading it would put a wrong
      // point on the area trend, which is worse than a missing one.
      if (reg.grade === 'low') {
        setResult({ rejected: true, grade: 'low', reasons: reg.reasons, metrics: reg.metrics });
        return;
      }

      // Severity is shown alongside the area trend, never mixed into it
      let severity = null;
      try {
        setStage('중증도 분석 중…');
        const cls = await classify(pic.uri);
        severity = {
          model: cls.modelName,
          predictions: cls.predictions.map((p) => ({
            task: p.task, titleKo: p.titleKo, label: p.label,
            index: p.index, confidence: Number(p.confidence.toFixed(3)),
          })),
        };
      } catch (e) {
        console.warn('[capture] 중증도 분석 생략:', e?.message || e);
      }

      setStage('업로드 중…');
      const res = await addCapture(site.id, pic.uri, { ...toCaptureMetrics(reg), severity });
      setResult({
        ...res,
        grade: reg.grade,
        reasons: reg.reasons,
        areaRatio: reg.areaRatio,
        metrics: reg.metrics,
        severity,
      });
    } catch (e) {
      setResult({ error: '분석 실패: ' + (e?.message || String(e)) });
    } finally {
      setStage(null);
      setBusy(false);
      busyRef.current = false;
    }
  };

  // Update every render so the interval calls the latest closure
  sampleRef.current = sample;

  if (!permission) return <View style={styles.black} />;
  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.center}>
        <Text style={styles.permText}>카메라 권한이 필요합니다</Text>
        <TouchableOpacity style={styles.permBtn} onPress={requestPermission}>
          <Text style={styles.permBtnText}>권한 허용</Text>
        </TouchableOpacity>
      </SafeAreaView>
    );
  }

  return (
    <View style={styles.black}>
      <CameraView ref={camRef} style={StyleSheet.absoluteFill} facing={facing} animateShutter={false} />

      {/* Framing guide = the anchor's ROI neighbourhood, zoomed. Matching it roughly
          is what keeps the scale gap small enough for ORB to lock on. */}
      {ghost && showGhost && (
        <Image source={{ uri: ghost }} style={[StyleSheet.absoluteFill, { opacity: 0.4 }]} resizeMode="contain" />
      )}

      <SafeAreaView style={styles.uiLayer} edges={['top', 'bottom']}>
        {/* Top */}
        <View style={styles.topBar}>
          <TouchableOpacity onPress={() => navigation.goBack()} hitSlop={10}>
            <Ionicons name="close" size={28} color={colors.white} />
          </TouchableOpacity>
          <Text style={styles.topTitle}>
            {site.bodyPart}{site.side ? ` (${sideLabel(site.side)})` : ''}
          </Text>
          {ghost ? (
            <TouchableOpacity onPress={() => setShowGhost((v) => !v)} hitSlop={10}>
              <Ionicons name={showGhost ? 'layers' : 'layers-outline'} size={24} color={colors.white} />
            </TouchableOpacity>
          ) : <View style={{ width: 24 }} />}
        </View>

        <Text style={styles.guideHint}>
          {anchorErr
            ? '기준 사진을 불러오지 못했습니다 — 부위 등록에서 기준 촬영을 먼저 완료하세요'
            : ghost
              ? '반투명 기준 영역과 같은 구도·거리로 병변을 담아 촬영하세요'
              : '기준 사진을 불러오는 중…'}
        </Text>

        {/* Live capture condition guidance (brightness/exposure/aspect ratio) */}
        {!result && !busy && (
          hints.length > 0 ? (
            <View style={styles.hintBanner} pointerEvents="none">
              {hints.map((h, i) => (
                <View key={i} style={styles.hintRow}>
                  <Ionicons name="warning" size={16} color="#FFD54A" />
                  <Text style={styles.hintText}>{h}</Text>
                </View>
              ))}
            </View>
          ) : (
            <View style={[styles.hintBanner, styles.hintBannerOk]} pointerEvents="none">
              <Ionicons name={checked ? 'checkmark-circle' : 'ellipsis-horizontal'} size={16}
                color={checked ? colors.mild : colors.white} />
              <Text style={styles.hintOkText}>{checked ? '촬영 조건 양호' : '촬영 조건 확인 중…'}</Text>
            </View>
          )
        )}

        {/* On-device pipeline progress (segment -> register -> severity -> upload) */}
        {busy && !!stage && (
          <View style={[styles.hintBanner, styles.hintBannerOk]} pointerEvents="none">
            <ActivityIndicator size="small" color={colors.white} />
            <Text style={styles.hintOkText}>{stage}</Text>
          </View>
        )}

        <View style={{ flex: 1 }} />

        {/* Shutter + front/back toggle */}
        <View style={styles.bottomBar}>
          <View style={styles.bottomSide} />
          <TouchableOpacity style={styles.shutter} onPress={shoot} disabled={busy}>
            {busy ? <ActivityIndicator color={colors.primary} />
              : <View style={styles.shutterInner} />}
          </TouchableOpacity>
          <View style={styles.bottomSide}>
            <TouchableOpacity
              style={styles.flipBtn}
              onPress={() => setFacing((f) => (f === 'back' ? 'front' : 'back'))}
              disabled={busy}
              hitSlop={10}
            >
              <Ionicons name="camera-reverse-outline" size={26} color={colors.white} />
            </TouchableOpacity>
          </View>
        </View>
      </SafeAreaView>

      {/* Capture result sheet */}
      {result && (
        <ResultSheet
          result={result}
          onRetake={() => setResult(null)}
          onDone={() => navigation.navigate('SiteTimeline', { site })}
        />
      )}
    </View>
  );
}

function ResultSheet({ result, onRetake, onDone }) {
  const g = GRADE_META[result.grade] || null;
  const m = result.metrics || {};
  const severity = result.severity?.predictions?.find((p) => p.task === 'severity');

  return (
    <View style={styles.sheetWrap}>
      <View style={styles.sheet}>
        {result.error ? (
          <>
            <Ionicons name="alert-circle" size={32} color={colors.severe} />
            <Text style={styles.sheetTitle}>분석 실패</Text>
            <Text style={styles.sheetMsg}>{result.error}</Text>
          </>
        ) : (
          <>
            <Ionicons name={g?.icon || 'checkmark-circle'} size={32} color={g?.color || colors.mild} />
            <Text style={styles.sheetTitle}>{g?.title || '촬영 완료'}</Text>
            <Text style={styles.sheetMsg}>{g?.note}</Text>
            {result.reasons?.length > 0 && (
              <Text style={styles.sheetMsg}>{result.reasons.join(' · ')}</Text>
            )}

            <ScrollView style={styles.metrics} contentContainerStyle={{ gap: 6 }}>
              {result.areaRatio != null && (
                <Metric label="병변 면적 (기준 영역 대비)" value={`${(result.areaRatio * 100).toFixed(1)}%`} />
              )}
              {severity && <Metric label="중증도(IGA)" value={severity.label} />}
              <Metric label="정합 일치점" value={`${m.inliers ?? '-'} / ${m.matches ?? '-'}`} />
              <Metric label="재투영 오차" value={m.reprojErr != null ? `${m.reprojErr.toFixed(2)} px` : '-'} />
              <Metric label="일치점 분포" value={m.coverage != null ? `${Math.round(m.coverage * 100)}%` : '-'} />
              <Metric label="기준 영역 담김" value={m.roiCoverage != null ? `${Math.round(m.roiCoverage * 100)}%` : '-'} />
              {result.erythema != null && <Metric label="홍반 지수(a*)" value={result.erythema} />}
              {result.blur != null && <Metric label="선명도(흐림)" value={result.blur} />}
            </ScrollView>
          </>
        )}
        <View style={styles.sheetBtns}>
          <TouchableOpacity style={[styles.sheetBtn, styles.sheetBtnGhost]} onPress={onRetake}>
            <Text style={styles.sheetBtnGhostText}>재촬영</Text>
          </TouchableOpacity>
          {!result.rejected && (
            <TouchableOpacity style={[styles.sheetBtn, styles.sheetBtnPrimary]} onPress={onDone}>
              <Text style={styles.sheetBtnPrimaryText}>완료</Text>
            </TouchableOpacity>
          )}
        </View>
      </View>
    </View>
  );
}

// Map server brightness/exposure reasons -> actionable guidance text
function brightnessHint(reason) {
  switch (reason) {
    case '너무 어두움': return '화면이 어두워요 — 더 밝은 곳에서 촬영하세요';
    case '너무 밝음': return '화면이 너무 밝아요 — 직사광/조명을 피해주세요';
    case '노출 과다/부족': return '빛 반사·그림자가 심해요 — 각도를 바꿔보세요';
    default: return reason;
  }
}

const Metric = ({ label, value }) => (
  <View style={styles.metricRow}>
    <Text style={styles.metricLabel}>{label}</Text>
    <Text style={styles.metricValue}>{value}</Text>
  </View>
);

const styles = StyleSheet.create({
  black: { flex: 1, backgroundColor: '#000' },
  center: { flex: 1, backgroundColor: colors.bg, alignItems: 'center', justifyContent: 'center', gap: 16 },
  permText: { fontSize: 16, color: colors.text, fontWeight: '600' },
  permBtn: { backgroundColor: colors.primary, paddingHorizontal: 24, paddingVertical: 12, borderRadius: radius.md },
  permBtnText: { color: colors.white, fontWeight: '700' },
  uiLayer: { flex: 1 },
  topBar: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: spacing.md, paddingTop: spacing.sm },
  topTitle: { color: colors.white, fontSize: 16, fontWeight: '800' },
  guideHint: { color: colors.white, textAlign: 'center', fontSize: 13, marginTop: spacing.sm, paddingHorizontal: spacing.lg,
    textShadowColor: 'rgba(0,0,0,0.6)', textShadowRadius: 4 },
  hintBanner: { marginTop: spacing.sm, marginHorizontal: spacing.lg, backgroundColor: 'rgba(0,0,0,0.55)',
    borderRadius: radius.md, paddingVertical: 10, paddingHorizontal: 14, gap: 6 },
  hintBannerOk: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, alignSelf: 'center', paddingVertical: 8 },
  hintRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  hintText: { flex: 1, color: colors.white, fontSize: 13, fontWeight: '600' },
  hintOkText: { color: colors.white, fontSize: 12, fontWeight: '600', opacity: 0.85 },
  bottomBar: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: spacing.xl, paddingBottom: spacing.lg },
  bottomSide: { width: 72, alignItems: 'center', justifyContent: 'center' },
  flipBtn: { width: 52, height: 52, borderRadius: 26, backgroundColor: 'rgba(255,255,255,0.18)', alignItems: 'center', justifyContent: 'center' },
  shutter: { width: 72, height: 72, borderRadius: 36, backgroundColor: 'rgba(255,255,255,0.25)', borderWidth: 4, borderColor: colors.white, alignItems: 'center', justifyContent: 'center' },
  shutterInner: { width: 56, height: 56, borderRadius: 28, backgroundColor: colors.white },
  sheetWrap: { ...StyleSheet.absoluteFillObject, backgroundColor: 'rgba(0,0,0,0.5)', justifyContent: 'flex-end' },
  sheet: { backgroundColor: colors.bg, borderTopLeftRadius: 24, borderTopRightRadius: 24, padding: spacing.lg, alignItems: 'center', gap: 8 },
  sheetTitle: { fontSize: 18, fontWeight: '800', color: colors.text },
  sheetMsg: { fontSize: 13, color: colors.textSub, textAlign: 'center' },
  metrics: { alignSelf: 'stretch', maxHeight: 180, marginTop: 8 },
  metricRow: { flexDirection: 'row', justifyContent: 'space-between', backgroundColor: colors.surface, borderRadius: radius.sm, paddingHorizontal: 14, paddingVertical: 10 },
  metricLabel: { fontSize: 14, color: colors.textSub },
  metricValue: { fontSize: 14, fontWeight: '700', color: colors.text },
  sheetBtns: { flexDirection: 'row', gap: 12, alignSelf: 'stretch', marginTop: spacing.md },
  sheetBtn: { flex: 1, paddingVertical: 14, borderRadius: radius.md, alignItems: 'center' },
  sheetBtnGhost: { backgroundColor: colors.surface },
  sheetBtnGhostText: { color: colors.text, fontWeight: '700' },
  sheetBtnPrimary: { backgroundColor: colors.primary },
  sheetBtnPrimaryText: { color: colors.white, fontWeight: '700' },
});

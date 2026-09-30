import React, { useRef, useState } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity, Image, ActivityIndicator, PanResponder,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { setSiteAnchor } from '../api/client';
import { sideLabel } from '../data/bodyMap';
import { colors, radius, spacing } from '../theme';

// Anchor capture: the ONE wide photo taken at site registration.
// Every later session is a close-up registered against this frame, so this photo
// (and the ROI drawn on it) defines the coordinate system the whole trend lives in.
// A bad anchor cannot be fixed later — hence the explicit review step before upload.

const MIN_ROI = 0.03;   // must match MIN_ROI in server/app.py

export default function AnchorCaptureScreen({ navigation, route }) {
  const { site } = route.params;
  const [permission, requestPermission] = useCameraPermissions();
  const camRef = useRef(null);

  const [photo, setPhoto] = useState(null);   // { uri, width, height }
  const [box, setBox] = useState(null);       // ROI in container px
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  // Container the photo is drawn into. The container keeps the photo's aspect
  // ratio, so touch coords map to image coords with a single divide (no letterbox math).
  const layout = useRef({ width: 0, height: 0 });
  const startPt = useRef(null);

  const pan = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: () => true,
      onPanResponderGrant: (e) => {
        const { locationX, locationY } = e.nativeEvent;
        startPt.current = { x: locationX, y: locationY };
        setBox({ x: locationX, y: locationY, w: 0, h: 0 });
      },
      onPanResponderMove: (e) => {
        const s = startPt.current;
        if (!s) return;
        const { width, height } = layout.current;
        const x2 = clamp(e.nativeEvent.locationX, 0, width);
        const y2 = clamp(e.nativeEvent.locationY, 0, height);
        setBox({
          x: Math.min(s.x, x2), y: Math.min(s.y, y2),
          w: Math.abs(x2 - s.x), h: Math.abs(y2 - s.y),
        });
      },
      onPanResponderRelease: () => { startPt.current = null; },
    })
  ).current;

  const shoot = async () => {
    if (!camRef.current || busy) return;
    setBusy(true);
    setError(null);
    try {
      const pic = await camRef.current.takePictureAsync({ quality: 0.85 });
      setPhoto(pic);
      setBox(null);
    } catch (e) {
      setError('카메라 촬영 실패: ' + (e?.message || String(e)));
    } finally {
      setBusy(false);
    }
  };

  const roi = box && layout.current.width > 0
    ? {
      x: box.x / layout.current.width,
      y: box.y / layout.current.height,
      w: box.w / layout.current.width,
      h: box.h / layout.current.height,
    }
    : null;
  const roiOk = !!roi && roi.w >= MIN_ROI && roi.h >= MIN_ROI;

  const submit = async () => {
    if (!roiOk || busy) return;
    setBusy(true);
    setError(null);
    try {
      const res = await setSiteAnchor(site.id, photo.uri, roi);
      if (res.accepted === false) {
        // Quality gate rejected it — the anchor must be re-shot, not accepted "for now"
        setError(`촬영 품질 미달: ${(res.reasons || []).join(' · ')} — 다시 촬영해주세요`);
        setPhoto(null);
        setBox(null);
        return;
      }
      navigation.replace('SiteTimeline', { site });
    } catch (e) {
      setError(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  };

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

  // ---- Step 2: review the shot and draw the lesion ROI ----
  if (photo) {
    const aspect = photo.width && photo.height ? photo.height / photo.width : 4 / 3;
    return (
      <SafeAreaView style={styles.safe} edges={['top', 'bottom']}>
        <View style={styles.header}>
          <TouchableOpacity onPress={() => { setPhoto(null); setBox(null); }} hitSlop={10}>
            <Ionicons name="arrow-back" size={24} color={colors.text} />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>병변 영역 지정</Text>
          <View style={{ width: 24 }} />
        </View>

        <Text style={styles.hint}>
          추적할 병변을 손가락으로 드래그해 사각형으로 감싸주세요.{'\n'}
          이 영역이 이후 모든 세션의 면적 기준이 됩니다.
        </Text>

        <View
          style={[styles.photoBox, { aspectRatio: 1 / aspect }]}
          onLayout={(e) => { layout.current = e.nativeEvent.layout; }}
          {...pan.panHandlers}
        >
          <Image source={{ uri: photo.uri }} style={StyleSheet.absoluteFill} resizeMode="contain" />
          {box && (
            <View
              pointerEvents="none"
              style={[styles.roiBox, { left: box.x, top: box.y, width: box.w, height: box.h }]}
            />
          )}
        </View>

        {!!error && <Text style={styles.error}>{error}</Text>}
        {box && !roiOk && <Text style={styles.warn}>영역이 너무 작습니다 — 조금 더 크게 지정하세요</Text>}

        <View style={styles.footerRow}>
          <TouchableOpacity
            style={[styles.btn, styles.btnGhost]}
            onPress={() => { setPhoto(null); setBox(null); }}
            disabled={busy}
          >
            <Text style={styles.btnGhostText}>다시 촬영</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.btn, styles.btnPrimary, (!roiOk || busy) && { opacity: 0.5 }]}
            onPress={submit}
            disabled={!roiOk || busy}
          >
            {busy ? <ActivityIndicator color={colors.white} />
              : <Text style={styles.btnPrimaryText}>기준으로 등록</Text>}
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  // ---- Step 1: take the wide shot ----
  return (
    <View style={styles.black}>
      <CameraView ref={camRef} style={StyleSheet.absoluteFill} facing="back" animateShutter={false} />
      <SafeAreaView style={styles.uiLayer} edges={['top', 'bottom']}>
        <View style={styles.topBar}>
          <TouchableOpacity onPress={() => navigation.goBack()} hitSlop={10}>
            <Ionicons name="close" size={28} color={colors.white} />
          </TouchableOpacity>
          <Text style={styles.topTitle}>
            기준 사진 · {site.bodyPart}{site.side ? ` (${sideLabel(site.side)})` : ''}
          </Text>
          <View style={{ width: 28 }} />
        </View>

        <Text style={styles.guideHint}>
          한 걸음 물러나 부위 전체가 보이게 촬영하세요.{'\n'}
          이 사진은 등록 시 한 번만 찍고, 이후 세션은 근거리만 촬영합니다.
        </Text>

        {!!error && <Text style={styles.errorOnCam}>{error}</Text>}

        <View style={{ flex: 1 }} />
        <View style={styles.bottomBar}>
          <TouchableOpacity style={styles.shutter} onPress={shoot} disabled={busy}>
            {busy ? <ActivityIndicator color={colors.primary} /> : <View style={styles.shutterInner} />}
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    </View>
  );
}

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

const styles = StyleSheet.create({
  black: { flex: 1, backgroundColor: '#000' },
  safe: { flex: 1, backgroundColor: colors.bg },
  center: { flex: 1, backgroundColor: colors.bg, alignItems: 'center', justifyContent: 'center', gap: 16 },
  permText: { fontSize: 16, color: colors.text, fontWeight: '600' },
  permBtn: { backgroundColor: colors.primary, paddingHorizontal: 24, paddingVertical: 12, borderRadius: radius.md },
  permBtnText: { color: colors.white, fontWeight: '700' },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: spacing.md, paddingVertical: spacing.sm },
  headerTitle: { fontSize: 18, fontWeight: '800', color: colors.text },
  hint: { fontSize: 13, color: colors.textSub, textAlign: 'center', lineHeight: 20, paddingHorizontal: spacing.lg, marginBottom: spacing.sm },
  photoBox: { marginHorizontal: spacing.md, backgroundColor: '#000', borderRadius: radius.md, overflow: 'hidden' },
  roiBox: { position: 'absolute', borderWidth: 2, borderColor: colors.primary, backgroundColor: 'rgba(59,130,246,0.18)' },
  error: { fontSize: 13, color: colors.severe, textAlign: 'center', marginTop: spacing.sm, paddingHorizontal: spacing.lg },
  warn: { fontSize: 12, color: colors.moderate, textAlign: 'center', marginTop: spacing.sm },
  footerRow: { flexDirection: 'row', gap: 12, padding: spacing.md, marginTop: 'auto' },
  btn: { flex: 1, paddingVertical: 16, borderRadius: radius.md, alignItems: 'center' },
  btnGhost: { backgroundColor: colors.surface },
  btnGhostText: { color: colors.text, fontWeight: '700' },
  btnPrimary: { backgroundColor: colors.primary },
  btnPrimaryText: { color: colors.white, fontWeight: '700' },
  uiLayer: { flex: 1 },
  topBar: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: spacing.md, paddingTop: spacing.sm },
  topTitle: { color: colors.white, fontSize: 15, fontWeight: '800' },
  guideHint: { color: colors.white, textAlign: 'center', fontSize: 13, lineHeight: 20, marginTop: spacing.md, paddingHorizontal: spacing.lg, textShadowColor: 'rgba(0,0,0,0.6)', textShadowRadius: 4 },
  errorOnCam: { color: '#FFD54A', textAlign: 'center', fontSize: 13, marginTop: spacing.sm, paddingHorizontal: spacing.lg },
  bottomBar: { alignItems: 'center', paddingBottom: spacing.lg },
  shutter: { width: 72, height: 72, borderRadius: 36, backgroundColor: 'rgba(255,255,255,0.25)', borderWidth: 4, borderColor: colors.white, alignItems: 'center', justifyContent: 'center' },
  shutterInner: { width: 56, height: 56, borderRadius: 28, backgroundColor: colors.white },
});

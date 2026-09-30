import React, { useState, useRef, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView, Alert, ActivityIndicator, Dimensions } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import { Card } from '../components/ui';
import { colors, radius, spacing } from '../theme';

// 1:1 정사각 프리뷰 한 변 = 화면 너비
const SQUARE = Dimensions.get('window').width;

// 촬영 원본을 중앙 기준 정사각(1:1)으로 크롭. 실패하면 원본 uri 그대로 반환.
async function cropToSquare(pic) {
  if (!pic?.width || !pic?.height) return pic.uri;
  const side = Math.min(pic.width, pic.height);
  const originX = Math.round((pic.width - side) / 2);
  const originY = Math.round((pic.height - side) / 2);
  const ctx = ImageManipulator.manipulate(pic.uri).crop({ originX, originY, width: side, height: side });
  const img = await ctx.renderAsync();
  const saved = await img.saveAsync({ compress: 0.9, format: SaveFormat.JPEG });
  return saved.uri;
}

const GUIDES = [
  { icon: 'sunny', color: '#F2B705', title: '밝은 조명', desc: '자연광이나 밝은 실내에서 촬영하세요' },
  { icon: 'search', color: '#3F7CE8', title: '선명한 초점', desc: '병변 부위에 정확히 초점을 맞추세요' },
  { icon: 'resize', color: '#7A8A84', title: '적정 거리', desc: '10~15cm 거리에서 촬영하세요' },
];

export default function DiagnosisScreen({ navigation }) {
  const [cameraOpen, setCameraOpen] = useState(false);
  const [facing, setFacing] = useState('back'); // front/back camera
  const [busy, setBusy] = useState(false);
  const [permission, requestPermission] = useCameraPermissions();
  const camRef = useRef(null);

  // 주의: 모델 메모리 증가량(로드 델타)을 정확히 재려면 미리 로드하면 안 된다.
  //       (preload 하면 추론 전 기준값에 이미 모델이 포함돼 로드 증가량이 0으로 나옴)
  //       -> 첫 진단에서 로드 전/후를 측정할 수 있도록 preload 하지 않는다.
  // useEffect(() => { preloadModel(); }, []);

  const goAnalyze = (uri) => navigation.navigate('Analyzing', { imageUri: uri });

  const takePhoto = async () => {
    let granted = permission?.granted;
    if (!granted) {
      const res = await requestPermission();
      granted = res?.granted;
    }
    if (!granted) {
      Alert.alert('권한 필요', '카메라 접근 권한을 허용해 주세요.');
      return;
    }
    setFacing('back');
    setCameraOpen(true);
  };

  const shoot = async () => {
    if (!camRef.current || busy) return;
    setBusy(true);
    try {
      const pic = await camRef.current.takePictureAsync({ quality: 0.9 });
      const uri = await cropToSquare(pic);      // 프리뷰와 동일하게 중앙 1:1 크롭
      setCameraOpen(false);
      goAnalyze(uri);
    } catch (e) {
      Alert.alert('촬영 실패', e?.message || String(e));
    } finally {
      setBusy(false);
    }
  };

  const pickImage = async () => {
    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) {
      Alert.alert('권한 필요', '사진 보관함 접근 권한을 허용해 주세요.');
      return;
    }
    // allowsEditing + aspect [1,1] -> 선택 시 1:1 크롭 편집 UI 제공
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.9,
    });
    if (!res.canceled) goAnalyze(res.assets[0].uri);
  };

  if (cameraOpen) {
    return (
      <View style={styles.camRoot}>
        {/* 1:1 정사각 프리뷰 (화면 중앙). 카메라 피드를 정사각으로 클리핑 */}
        <View style={styles.camSquareWrap} pointerEvents="none">
          <View style={[styles.camSquare, { width: SQUARE, height: SQUARE }]}>
            <CameraView ref={camRef} style={StyleSheet.absoluteFill} facing={facing} />
          </View>
        </View>
        <SafeAreaView style={styles.camUi} edges={['top', 'bottom']}>
          <View style={styles.camTopBar}>
            <TouchableOpacity onPress={() => setCameraOpen(false)} hitSlop={10} disabled={busy}>
              <Ionicons name="close" size={28} color={colors.white} />
            </TouchableOpacity>
            <Text style={styles.camTitle}>병변 촬영</Text>
            <View style={{ width: 28 }} />
          </View>

          <Text style={styles.camHint}>1:1 정사각 비율로 촬영됩니다</Text>

          <View style={{ flex: 1 }} />

          <View style={styles.camBottomBar}>
            <View style={styles.camSide} />
            <TouchableOpacity style={styles.shutter} onPress={shoot} disabled={busy}>
              {busy ? <ActivityIndicator color={colors.primary} />
                : <View style={styles.shutterInner} />}
            </TouchableOpacity>
            <View style={styles.camSide}>
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
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView contentContainerStyle={styles.scroll}>
        <Text style={styles.h1}>병변 진단</Text>
        <Text style={styles.sub}>사진을 촬영하거나 업로드하여 AI 진단을 받으세요</Text>

        {/* Upload dropzone */}
        <TouchableOpacity style={styles.dropzone} activeOpacity={0.8} onPress={pickImage}>
          <View style={styles.dropIcon}>
            <Ionicons name="image-outline" size={32} color={colors.textSub} />
          </View>
          <Text style={styles.dropTitle}>사진을 여기에 드래그하거나 탭하세요</Text>
          <Text style={styles.dropSub}>JPG · PNG · HEIF 지원</Text>
        </TouchableOpacity>

        {/* Capture / select */}
        <View style={styles.actionRow}>
          <TouchableOpacity style={[styles.action, styles.actionPrimary]} onPress={takePhoto}>
            <Ionicons name="camera" size={24} color={colors.white} />
            <Text style={styles.actionPrimaryText}>카메라 촬영</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.action, styles.actionSecondary]} onPress={pickImage}>
            <Ionicons name="cloud-upload-outline" size={24} color={colors.primary} />
            <Text style={styles.actionSecondaryText}>갤러리 선택</Text>
          </TouchableOpacity>
        </View>

        {/* Capture guide */}
        <Text style={styles.guideHeader}>촬영 가이드</Text>
        {GUIDES.map((g) => (
          <Card key={g.title} style={styles.guideCard}>
            <View style={[styles.guideIcon, { backgroundColor: g.color + '22' }]}>
              <Ionicons name={g.icon} size={18} color={g.color} />
            </View>
            <View style={{ flex: 1 }}>
              <Text style={styles.guideTitle}>{g.title}</Text>
              <Text style={styles.guideDesc}>{g.desc}</Text>
            </View>
          </Card>
        ))}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  camRoot: { flex: 1, backgroundColor: '#000' },
  camSquareWrap: { ...StyleSheet.absoluteFillObject, alignItems: 'center', justifyContent: 'center' },
  camSquare: { overflow: 'hidden', borderRadius: radius.lg, backgroundColor: '#000' },
  camUi: { flex: 1 },
  camTopBar: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: spacing.md, paddingTop: spacing.sm },
  camTitle: { color: colors.white, fontSize: 16, fontWeight: '800' },
  camHint: { color: colors.white, textAlign: 'center', fontSize: 13, marginTop: spacing.sm, opacity: 0.9,
    textShadowColor: 'rgba(0,0,0,0.6)', textShadowRadius: 4 },
  camBottomBar: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: spacing.xl, paddingBottom: spacing.lg },
  camSide: { width: 72, alignItems: 'center', justifyContent: 'center' },
  flipBtn: { width: 52, height: 52, borderRadius: 26, backgroundColor: 'rgba(255,255,255,0.18)', alignItems: 'center', justifyContent: 'center' },
  shutter: { width: 72, height: 72, borderRadius: 36, backgroundColor: 'rgba(255,255,255,0.25)', borderWidth: 4, borderColor: colors.white, alignItems: 'center', justifyContent: 'center' },
  shutterInner: { width: 56, height: 56, borderRadius: 28, backgroundColor: colors.white },
  safe: { flex: 1, backgroundColor: colors.bg },
  scroll: { padding: spacing.md, paddingBottom: spacing.xl },
  h1: { fontSize: 24, fontWeight: '800', color: colors.text },
  sub: { fontSize: 14, color: colors.textSub, marginTop: 4, marginBottom: spacing.md },
  dropzone: {
    borderWidth: 2, borderColor: colors.border, borderStyle: 'dashed',
    borderRadius: radius.lg, paddingVertical: 36, alignItems: 'center',
    backgroundColor: colors.surface,
  },
  dropIcon: {
    width: 64, height: 64, borderRadius: 16, backgroundColor: '#E7ECEA',
    alignItems: 'center', justifyContent: 'center', marginBottom: 12,
  },
  dropTitle: { fontSize: 15, fontWeight: '700', color: colors.text },
  dropSub: { fontSize: 12, color: colors.textMuted, marginTop: 4 },
  actionRow: { flexDirection: 'row', gap: 12, marginTop: spacing.md },
  action: {
    flex: 1, alignItems: 'center', justifyContent: 'center', gap: 6,
    paddingVertical: 20, borderRadius: radius.md,
  },
  actionPrimary: { backgroundColor: colors.primary },
  actionPrimaryText: { color: colors.white, fontWeight: '700', fontSize: 15 },
  actionSecondary: { backgroundColor: colors.card, borderWidth: 1, borderColor: colors.border },
  actionSecondaryText: { color: colors.primary, fontWeight: '700', fontSize: 15 },
  guideHeader: { fontSize: 14, fontWeight: '700', color: colors.textSub, marginTop: spacing.lg, marginBottom: spacing.sm },
  guideCard: { flexDirection: 'row', alignItems: 'center', gap: 12, marginBottom: 10 },
  guideIcon: { width: 36, height: 36, borderRadius: 18, alignItems: 'center', justifyContent: 'center' },
  guideTitle: { fontSize: 15, fontWeight: '700', color: colors.text },
  guideDesc: { fontSize: 13, color: colors.textSub, marginTop: 2 },
});

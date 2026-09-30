import React, { useState } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, TextInput, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { createSite } from '../api/client';
import { CONDITIONS, BODY_VIEWS, BODY_REGIONS, SIDES } from '../data/bodyMap';
import { colors, radius, spacing } from '../theme';

export default function AddSiteScreen({ navigation }) {
  const [condition, setCondition] = useState('uremic');
  const [view, setView] = useState('front');
  const [bodyPart, setBodyPart] = useState(null);
  const [side, setSide] = useState('C');
  const [label, setLabel] = useState('');
  const [busy, setBusy] = useState(false);

  const condHint = CONDITIONS.find((c) => c.key === condition)?.hint;

  // Registering a site is only half the setup: the anchor photo defines the
  // coordinate frame every later session is measured in, so go straight to it.
  const submit = async () => {
    if (!bodyPart) return;
    setBusy(true);
    try {
      const res = await createSite({ bodyPart, side, label: label.trim(), condition });
      const site = { id: res.id, bodyPart, side, label: label.trim(), condition, hasAnchor: false };
      navigation.replace('AnchorCapture', { site });
    } catch (e) {
      setBusy(false);
    }
  };

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} hitSlop={10}>
          <Ionicons name="arrow-back" size={24} color={colors.text} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>모니터링 부위 추가</Text>
        <View style={{ width: 24 }} />
      </View>

      <ScrollView contentContainerStyle={styles.scroll}>
        {/* Condition */}
        <Text style={styles.section}>질환</Text>
        <View style={styles.row}>
          {CONDITIONS.map((c) => (
            <Chip key={c.key} active={condition === c.key} label={c.label} onPress={() => setCondition(c.key)} />
          ))}
        </View>
        {!!condHint && <Text style={styles.hint}>{condHint}</Text>}

        {/* Front/back */}
        <Text style={styles.section}>부위 (앞/뒤)</Text>
        <View style={styles.segment}>
          {BODY_VIEWS.map((v) => (
            <TouchableOpacity
              key={v.key}
              style={[styles.segBtn, view === v.key && styles.segBtnActive]}
              onPress={() => { setView(v.key); setBodyPart(null); }}
            >
              <Text style={[styles.segText, view === v.key && { color: colors.white }]}>{v.label}</Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* Site grid */}
        <View style={styles.grid}>
          {BODY_REGIONS[view].map((part) => (
            <Chip key={part} active={bodyPart === part} label={part} onPress={() => setBodyPart(part)} />
          ))}
        </View>

        {/* Left/right */}
        <Text style={styles.section}>좌/우</Text>
        <View style={styles.row}>
          {SIDES.map((s) => (
            <Chip key={s.key} active={side === s.key} label={s.label} onPress={() => setSide(s.key)} />
          ))}
        </View>

        {/* Memo/label */}
        <Text style={styles.section}>슬롯 이름 (선택)</Text>
        <TextInput
          style={styles.input}
          value={label}
          onChangeText={setLabel}
          placeholder="예: 좌측 팔뚝 찰상 / 조사부위 1"
          placeholderTextColor={colors.textMuted}
        />
      </ScrollView>

      <View style={styles.footer}>
        <TouchableOpacity
          style={[styles.cta, (!bodyPart || busy) && { opacity: 0.5 }]}
          onPress={submit}
          disabled={!bodyPart || busy}
        >
          {busy ? <ActivityIndicator color={colors.white} />
            : <Text style={styles.ctaText}>{bodyPart ? `${bodyPart} 부위 등록` : '부위를 선택하세요'}</Text>}
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

function Chip({ active, label, onPress }) {
  return (
    <TouchableOpacity style={[styles.chip, active && styles.chipActive]} onPress={onPress} activeOpacity={0.85}>
      <Text style={[styles.chipText, active && { color: colors.white }]}>{label}</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bg },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: spacing.md, paddingVertical: spacing.sm },
  headerTitle: { fontSize: 18, fontWeight: '800', color: colors.text },
  scroll: { padding: spacing.md, paddingBottom: spacing.xl },
  section: { fontSize: 14, fontWeight: '700', color: colors.textSub, marginTop: spacing.lg, marginBottom: spacing.sm },
  row: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  hint: { fontSize: 12, color: colors.textMuted, marginTop: 8 },
  segment: { flexDirection: 'row', backgroundColor: colors.surface, borderRadius: radius.md, padding: 4 },
  segBtn: { flex: 1, alignItems: 'center', paddingVertical: 10, borderRadius: radius.sm },
  segBtnActive: { backgroundColor: colors.primary },
  segText: { fontSize: 14, fontWeight: '700', color: colors.textSub },
  chip: { paddingHorizontal: 14, paddingVertical: 9, borderRadius: radius.pill, backgroundColor: colors.surface, borderWidth: 1, borderColor: colors.border },
  chipActive: { backgroundColor: colors.primary, borderColor: colors.primary },
  chipText: { fontSize: 14, color: colors.text, fontWeight: '600' },
  input: { backgroundColor: colors.surface, borderRadius: radius.md, paddingHorizontal: 14, paddingVertical: 12, fontSize: 15, color: colors.text },
  footer: { padding: spacing.md, borderTopWidth: 1, borderTopColor: colors.border },
  cta: { backgroundColor: colors.primary, borderRadius: radius.md, paddingVertical: 16, alignItems: 'center' },
  ctaText: { color: colors.white, fontSize: 16, fontWeight: '700' },
});

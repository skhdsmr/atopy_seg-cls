import React, { useState, useCallback, useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, ActivityIndicator, Image } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { Calendar } from 'react-native-calendars';
import { Card, Pill } from '../components/ui';
import { getSites, getCaptureDates, getCapturesByDate, captureImageUrl } from '../api/client';
import { conditionLabel, sideLabel } from '../data/bodyMap';
import { colors, radius, spacing } from '../theme';

// Today's date in local time (YYYY-MM-DD)
function todayStr() {
  const d = new Date();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${d.getFullYear()}-${m}-${day}`;
}

export default function MonitoringScreen({ navigation }) {
  const [mode, setMode] = useState('sites'); // 'sites' | 'calendar'
  const [sites, setSites] = useState(null);
  const [error, setError] = useState(null);

  // Calendar state
  const [dates, setDates] = useState({});       // { 'YYYY-MM-DD': count }
  const [selected, setSelected] = useState(todayStr());
  const [dayCaps, setDayCaps] = useState(null); // captures for the selected date

  const load = useCallback(async () => {
    setError(null);
    try {
      setSites(await getSites());
    } catch (e) {
      setError(e.message || '부위 목록을 불러오지 못했습니다');
    }
    try {
      setDates(await getCaptureDates());
    } catch { /* calendar dots failing is not critical */ }
  }, []);

  useFocusEffect(useCallback(() => { load(); }, [load]));

  // Load captures for the selected date (calendar mode only)
  useEffect(() => {
    if (mode !== 'calendar') return;
    let alive = true;
    setDayCaps(null);
    getCapturesByDate(selected)
      .then((d) => { if (alive) setDayCaps(d); })
      .catch(() => { if (alive) setDayCaps([]); });
    return () => { alive = false; };
  }, [mode, selected]);

  // Calendar dots + selected-day highlight
  const marked = {};
  Object.keys(dates).forEach((d) => {
    marked[d] = { marked: true, dotColor: colors.primary };
  });
  marked[selected] = { ...(marked[selected] || {}), selected: true, selectedColor: colors.primary };

  const openTimeline = (cap) => {
    // Build a minimal site object from the by-date result and navigate to the timeline
    navigation.navigate('SiteTimeline', {
      site: {
        id: cap.siteId, bodyPart: cap.bodyPart, side: cap.side,
        label: cap.label, condition: cap.condition,
      },
    });
  };

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView contentContainerStyle={styles.scroll}>
        <View style={styles.headerRow}>
          <View>
            <Text style={styles.h1}>병변 모니터링</Text>
            <Text style={styles.sub}>부위별로 같은 구도·기준으로 추적하세요</Text>
          </View>
          <TouchableOpacity style={styles.addBtn} onPress={() => navigation.navigate('AddSite')}>
            <Ionicons name="add" size={22} color={colors.white} />
          </TouchableOpacity>
        </View>

        {/* View toggle: by site / calendar */}
        <View style={styles.segment}>
          {[
            { key: 'sites', label: '부위별', icon: 'body-outline' },
            { key: 'calendar', label: '캘린더', icon: 'calendar-outline' },
          ].map((t) => (
            <TouchableOpacity
              key={t.key}
              style={[styles.segBtn, mode === t.key && styles.segBtnActive]}
              onPress={() => setMode(t.key)}
            >
              <Ionicons name={t.icon} size={16} color={mode === t.key ? colors.white : colors.textSub} />
              <Text style={[styles.segText, mode === t.key && { color: colors.white }]}>{t.label}</Text>
            </TouchableOpacity>
          ))}
        </View>

        {mode === 'sites' ? (
          <SitesView sites={sites} error={error} onRetry={load} navigation={navigation} />
        ) : (
          <CalendarView
            selected={selected} marked={marked} dayCaps={dayCaps}
            onSelect={(d) => setSelected(d)} onOpen={openTimeline} navigation={navigation}
          />
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

// ------------------------------------------------------------
// By-site view (existing implementation)
// ------------------------------------------------------------
function SitesView({ sites, error, onRetry, navigation }) {
  if (error) {
    return (
      <View style={styles.center}>
        <Text style={styles.errText}>{error}</Text>
        <TouchableOpacity style={styles.retry} onPress={onRetry}><Text style={styles.retryText}>다시 시도</Text></TouchableOpacity>
      </View>
    );
  }
  if (!sites) return <ActivityIndicator color={colors.primary} style={{ marginTop: spacing.xl }} />;
  if (sites.length === 0) {
    return (
      <View style={styles.empty}>
        <Ionicons name="body-outline" size={40} color={colors.textMuted} />
        <Text style={styles.emptyTitle}>등록된 부위가 없습니다</Text>
        <Text style={styles.emptyDesc}>오른쪽 위 + 로 모니터링할 부위를 추가하세요.{'\n'}방사선 조사 부위나 가려운 부위를 등록하면 됩니다.</Text>
      </View>
    );
  }
  return sites.map((s) => (
    <TouchableOpacity key={s.id} activeOpacity={0.85}
      onPress={() => navigation.navigate('SiteTimeline', { site: s })}>
      <Card style={styles.siteCard}>
        <View style={styles.siteIcon}>
          <Ionicons name="locate" size={20} color={colors.primary} />
        </View>
        <View style={{ flex: 1 }}>
          <View style={styles.siteTop}>
            <Text style={styles.siteName}>
              {s.bodyPart}{s.side ? ` (${sideLabel(s.side)})` : ''}
            </Text>
            <Pill label={conditionLabel(s.condition)} />
          </View>
          {!!s.label && <Text style={styles.siteLabel}>{s.label}</Text>}
          <Text style={styles.siteMeta}>
            {s.hasAnchor ? `촬영 ${s.captureCount}회` : '기준 사진 필요'}
            {s.lastAreaRatio != null ? `  ·  최근 면적 ${(s.lastAreaRatio * 100).toFixed(1)}%` : ''}
            {s.lastDate ? `  ·  ${s.lastDate.slice(5)}` : ''}
          </Text>
        </View>
        {/* Sessions require an anchor — send the user to register one first */}
        <TouchableOpacity
          style={styles.shootBtn}
          onPress={() => navigation.navigate(s.hasAnchor ? 'GuidedCapture' : 'AnchorCapture', { site: s })}
        >
          <Ionicons name={s.hasAnchor ? 'camera' : 'scan'} size={18} color={colors.white} />
        </TouchableOpacity>
      </Card>
    </TouchableOpacity>
  ));
}

// ------------------------------------------------------------
// Calendar view (tap a date → all site photos for that day)
// ------------------------------------------------------------
function CalendarView({ selected, marked, dayCaps, onSelect, onOpen, navigation }) {
  const [mo, dy] = [Number(selected.split('-')[1]), Number(selected.split('-')[2])];
  return (
    <>
      <Card style={{ padding: 0, overflow: 'hidden' }}>
        <Calendar
          current={selected}
          onDayPress={(d) => onSelect(d.dateString)}
          markedDates={marked}
          theme={{
            todayTextColor: colors.primary,
            arrowColor: colors.primary,
            selectedDayBackgroundColor: colors.primary,
            textMonthFontWeight: '700',
          }}
        />
      </Card>

      <View style={styles.dayHead}>
        <Text style={styles.dayTitle}>{mo}월 {dy}일</Text>
        {Array.isArray(dayCaps) && dayCaps.length > 0 && (
          <Text style={styles.dayCount}>촬영 {dayCaps.length}장</Text>
        )}
      </View>

      {dayCaps === null ? (
        <ActivityIndicator color={colors.primary} style={{ marginTop: spacing.lg }} />
      ) : dayCaps.length === 0 ? (
        <View style={styles.empty}>
          <Ionicons name="camera-outline" size={36} color={colors.textMuted} />
          <Text style={styles.emptyDesc}>이 날의 촬영이 없습니다.</Text>
        </View>
      ) : (
        <View style={styles.grid}>
          {dayCaps.map((c) => (
            <TouchableOpacity key={c.id} style={styles.tile} activeOpacity={0.85} onPress={() => onOpen(c)}>
              <View style={styles.thumbWrap}>
                {c.hasImage ? (
                  <Image source={{ uri: captureImageUrl(c.id) }} style={styles.thumb} />
                ) : (
                  <View style={[styles.thumb, styles.thumbEmpty]}>
                    <Ionicons name="image-outline" size={22} color={colors.textMuted} />
                  </View>
                )}
                <View style={[styles.qBadge, { backgroundColor: c.qualityOk ? colors.mild : colors.moderate }]}>
                  <Ionicons name={c.qualityOk ? 'checkmark' : 'alert'} size={11} color={colors.white} />
                </View>
              </View>
              <Text style={styles.tileName} numberOfLines={1}>
                {c.bodyPart}{c.side ? ` (${sideLabel(c.side)})` : ''}
              </Text>
              <Text style={styles.tileMeta} numberOfLines={1}>
                {c.date.slice(11)}{c.erythema != null ? `  ·  홍반 ${c.erythema}` : ''}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      )}
    </>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bg },
  scroll: { padding: spacing.md, paddingBottom: spacing.xl },
  headerRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: spacing.md },
  h1: { fontSize: 24, fontWeight: '800', color: colors.text },
  sub: { fontSize: 14, color: colors.textSub, marginTop: 4 },
  addBtn: { width: 40, height: 40, borderRadius: 20, backgroundColor: colors.primary, alignItems: 'center', justifyContent: 'center' },

  segment: { flexDirection: 'row', backgroundColor: colors.surface, borderRadius: radius.md, padding: 4, marginBottom: spacing.md },
  segBtn: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, paddingVertical: 10, borderRadius: radius.sm },
  segBtnActive: { backgroundColor: colors.primary },
  segText: { fontSize: 14, fontWeight: '700', color: colors.textSub },

  center: { alignItems: 'center', marginTop: spacing.xl, gap: 10 },
  errText: { color: colors.text, fontWeight: '600' },
  retry: { backgroundColor: colors.primary, paddingHorizontal: 18, paddingVertical: 9, borderRadius: radius.pill },
  retryText: { color: colors.white, fontWeight: '700' },
  empty: { alignItems: 'center', marginTop: spacing.xl * 1.5, gap: 8 },
  emptyTitle: { fontSize: 16, fontWeight: '700', color: colors.text, marginTop: 6 },
  emptyDesc: { fontSize: 13, color: colors.textSub, textAlign: 'center', lineHeight: 20 },

  siteCard: { flexDirection: 'row', alignItems: 'center', gap: 12, marginBottom: 12 },
  siteIcon: { width: 40, height: 40, borderRadius: 20, backgroundColor: colors.primarySoft, alignItems: 'center', justifyContent: 'center' },
  siteTop: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  siteName: { fontSize: 16, fontWeight: '800', color: colors.text },
  siteLabel: { fontSize: 13, color: colors.textSub, marginTop: 2 },
  siteMeta: { fontSize: 12, color: colors.textMuted, marginTop: 4 },
  shootBtn: { width: 40, height: 40, borderRadius: 20, backgroundColor: colors.primary, alignItems: 'center', justifyContent: 'center' },

  dayHead: { flexDirection: 'row', alignItems: 'baseline', justifyContent: 'space-between', marginTop: spacing.md, marginBottom: spacing.sm },
  dayTitle: { fontSize: 17, fontWeight: '800', color: colors.text },
  dayCount: { fontSize: 13, color: colors.textSub, fontWeight: '600' },
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm },
  tile: { width: '31.5%' },
  thumbWrap: { width: '100%', aspectRatio: 1, borderRadius: radius.md, overflow: 'hidden', backgroundColor: colors.surface },
  thumb: { width: '100%', height: '100%', resizeMode: 'cover' },
  thumbEmpty: { alignItems: 'center', justifyContent: 'center' },
  qBadge: { position: 'absolute', top: 6, right: 6, width: 20, height: 20, borderRadius: 10, alignItems: 'center', justifyContent: 'center' },
  tileName: { fontSize: 12, fontWeight: '700', color: colors.text, marginTop: 4 },
  tileMeta: { fontSize: 11, color: colors.textMuted, marginTop: 1 },
});

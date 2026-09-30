import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { Card } from '../components/ui';
import { getChats, CHAT_FEE_KRW } from '../api/client';
import { colors, radius, spacing } from '../theme';

export default function ChatListScreen({ navigation }) {
  const [chats, setChats] = useState([]);

  useEffect(() => {
    getChats().then(setChats);
  }, []);

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView contentContainerStyle={styles.scroll}>
        <Text style={styles.h1}>채팅</Text>
        <Text style={styles.sub}>담당 의사와 메시지를 교환하세요</Text>

        {/* Billing notice (patient pays) */}
        <View style={styles.notice}>
          <Ionicons name="alert-circle" size={18} color={colors.warning} />
          <Text style={styles.noticeText}>
            채팅 요금은 환자에게 부과됩니다. 메시지 발송 시 건당 {CHAT_FEE_KRW}원이 청구됩니다.
          </Text>
        </View>

        {chats.map((c) => (
          <TouchableOpacity key={c.id} activeOpacity={0.8} onPress={() => navigation.navigate('ChatRoom', { chat: c })}>
            <Card style={styles.chatCard}>
              <View style={styles.avatar}><Text style={styles.avatarText}>{c.avatar}</Text></View>
              <View style={{ flex: 1 }}>
                <View style={styles.chatTop}>
                  <Text style={styles.doctor}>{c.doctor}</Text>
                  <Text style={styles.time}>{c.time}</Text>
                </View>
                <Text style={styles.hospital}>{c.hospital}</Text>
                <Text style={styles.lastMsg} numberOfLines={1}>{c.lastMessage}</Text>
              </View>
              {c.unread > 0 && (
                <View style={styles.badge}><Text style={styles.badgeText}>{c.unread}</Text></View>
              )}
            </Card>
          </TouchableOpacity>
        ))}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bg },
  scroll: { padding: spacing.md, paddingBottom: spacing.xl },
  h1: { fontSize: 24, fontWeight: '800', color: colors.text },
  sub: { fontSize: 14, color: colors.textSub, marginTop: 4, marginBottom: spacing.md },
  notice: { flexDirection: 'row', gap: 8, alignItems: 'center', backgroundColor: colors.warningBg, borderRadius: radius.md, padding: 12, marginBottom: spacing.md },
  noticeText: { flex: 1, fontSize: 13, color: '#8A6D1E' },
  chatCard: { flexDirection: 'row', alignItems: 'center', gap: 12, marginBottom: 12 },
  avatar: { width: 48, height: 48, borderRadius: 24, backgroundColor: colors.primary, alignItems: 'center', justifyContent: 'center' },
  avatarText: { color: colors.white, fontSize: 18, fontWeight: '800' },
  chatTop: { flexDirection: 'row', justifyContent: 'space-between' },
  doctor: { fontSize: 16, fontWeight: '800', color: colors.text },
  time: { fontSize: 12, color: colors.textMuted },
  hospital: { fontSize: 13, color: colors.textSub, marginTop: 2 },
  lastMsg: { fontSize: 14, color: colors.text, marginTop: 4 },
  badge: { minWidth: 22, height: 22, borderRadius: 11, backgroundColor: colors.primary, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 6 },
  badgeText: { color: colors.white, fontSize: 12, fontWeight: '700' },
});

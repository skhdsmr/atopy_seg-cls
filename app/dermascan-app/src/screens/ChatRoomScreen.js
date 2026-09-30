import React, { useState, useRef } from 'react';
import {
  View, Text, StyleSheet, TextInput, TouchableOpacity, ScrollView,
  KeyboardAvoidingView, Platform, Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { sendMessage, CHAT_FEE_KRW } from '../api/client';
import { useAuth } from '../context/AuthContext';
import { colors, radius, spacing } from '../theme';

export default function ChatRoomScreen({ navigation, route }) {
  const { chat } = route.params;
  const { user } = useAuth();
  const role = user?.role ?? 'patient'; // separate doctor/patient views
  const [messages, setMessages] = useState(chat.messages);
  const [text, setText] = useState('');
  const scrollRef = useRef(null);

  const send = async () => {
    const body = text.trim();
    if (!body) return;

    // Patients confirm billing before sending
    if (role === 'patient') {
      Alert.alert(
        '메시지 발송',
        `이 메시지를 보내면 ${CHAT_FEE_KRW}원이 청구됩니다. 계속할까요?`,
        [
          { text: '취소', style: 'cancel' },
          { text: '발송', onPress: () => doSend(body) },
        ]
      );
    } else {
      doSend(body); // free for doctors
    }
  };

  const doSend = async (body) => {
    setText('');
    const res = await sendMessage(chat.id, body, role);
    setMessages((prev) => [...prev, { ...res.message, ts: '방금' }]);
    setTimeout(() => scrollRef.current?.scrollToEnd({ animated: true }), 50);
  };

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} hitSlop={10}>
          <Ionicons name="arrow-back" size={24} color={colors.text} />
        </TouchableOpacity>
        <View style={{ flex: 1 }}>
          <Text style={styles.headerTitle}>{chat.doctor}</Text>
          <Text style={styles.headerSub}>{chat.hospital}</Text>
        </View>
        <View style={styles.roleBadge}>
          <Text style={styles.roleText}>{role === 'doctor' ? '의사' : '환자'}</Text>
        </View>
      </View>

      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={80}
      >
        <ScrollView ref={scrollRef} contentContainerStyle={styles.scroll}>
          {messages.map((m) => {
            const mine = m.senderRole === role;
            return (
              <View key={m.id} style={[styles.bubbleRow, mine ? styles.rowMine : styles.rowTheirs]}>
                <View style={[styles.bubble, mine ? styles.bubbleMine : styles.bubbleTheirs]}>
                  <Text style={[styles.bubbleText, mine && { color: colors.white }]}>{m.text}</Text>
                </View>
                <Text style={styles.ts}>{m.ts}</Text>
              </View>
            );
          })}
        </ScrollView>

        {role === 'patient' && (
          <Text style={styles.feeHint}>메시지 발송 시 건당 {CHAT_FEE_KRW}원 청구</Text>
        )}
        <View style={styles.inputRow}>
          <TextInput
            style={styles.input}
            value={text}
            onChangeText={setText}
            placeholder="메시지를 입력하세요"
            placeholderTextColor={colors.textMuted}
            multiline
          />
          <TouchableOpacity style={styles.sendBtn} onPress={send}>
            <Ionicons name="send" size={18} color={colors.white} />
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bg },
  header: { flexDirection: 'row', alignItems: 'center', gap: 12, paddingHorizontal: spacing.md, paddingVertical: spacing.sm, borderBottomWidth: 1, borderBottomColor: colors.border },
  headerTitle: { fontSize: 16, fontWeight: '800', color: colors.text },
  headerSub: { fontSize: 12, color: colors.textSub },
  roleBadge: { backgroundColor: colors.primarySoft, paddingHorizontal: 10, paddingVertical: 4, borderRadius: radius.pill },
  roleText: { color: colors.primary, fontSize: 12, fontWeight: '700' },
  scroll: { padding: spacing.md, gap: 12 },
  bubbleRow: { maxWidth: '80%' },
  rowMine: { alignSelf: 'flex-end', alignItems: 'flex-end' },
  rowTheirs: { alignSelf: 'flex-start', alignItems: 'flex-start' },
  bubble: { paddingHorizontal: 14, paddingVertical: 10, borderRadius: radius.lg },
  bubbleMine: { backgroundColor: colors.primary, borderTopRightRadius: 4 },
  bubbleTheirs: { backgroundColor: colors.surface, borderTopLeftRadius: 4 },
  bubbleText: { fontSize: 15, color: colors.text },
  ts: { fontSize: 11, color: colors.textMuted, marginTop: 3 },
  feeHint: { textAlign: 'center', fontSize: 12, color: colors.textMuted, paddingVertical: 4 },
  inputRow: { flexDirection: 'row', alignItems: 'flex-end', gap: 8, padding: spacing.sm, borderTopWidth: 1, borderTopColor: colors.border },
  input: { flex: 1, maxHeight: 100, backgroundColor: colors.surface, borderRadius: radius.lg, paddingHorizontal: 14, paddingVertical: 10, fontSize: 15, color: colors.text },
  sendBtn: { width: 44, height: 44, borderRadius: 22, backgroundColor: colors.primary, alignItems: 'center', justifyContent: 'center' },
});

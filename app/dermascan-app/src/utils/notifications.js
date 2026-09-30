// Hospital visit reminder (schedules a notification a few days out)
import * as Notifications from 'expo-notifications';
import { Platform } from 'react-native';

Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: false,
  }),
});

export async function ensurePermission() {
  const { status } = await Notifications.getPermissionsAsync();
  if (status === 'granted') return true;
  const req = await Notifications.requestPermissionsAsync();
  return req.status === 'granted';
}

// Schedule a hospital visit reminder for a specific Date
export async function scheduleVisitReminder(hospitalName, date) {
  if (Platform.OS === 'web') return null;
  const ok = await ensurePermission();
  if (!ok) return null;

  return Notifications.scheduleNotificationAsync({
    content: {
      title: '병원 방문 예약 알림',
      body: `오늘 ${hospitalName} 방문 예정일입니다.`,
      sound: true,
    },
    trigger: date, // Date object -> one-time notification at that time
  });
}

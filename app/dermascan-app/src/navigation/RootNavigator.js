import React from 'react';
import { ActivityIndicator, View } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';

import { useAuth } from '../context/AuthContext';
import { colors } from '../theme';

import LoginScreen from '../screens/LoginScreen';
import RoleSelectScreen from '../screens/RoleSelectScreen';
import DiagnosisScreen from '../screens/DiagnosisScreen';
import AnalyzingScreen from '../screens/AnalyzingScreen';
import ResultScreen from '../screens/ResultScreen';
import HospitalsScreen from '../screens/HospitalsScreen';
import HospitalDetailScreen from '../screens/HospitalDetailScreen';
import MonitoringScreen from '../screens/MonitoringScreen';
import AddSiteScreen from '../screens/AddSiteScreen';
import AnchorCaptureScreen from '../screens/AnchorCaptureScreen';
import GuidedCaptureScreen from '../screens/GuidedCaptureScreen';
import SiteTimelineScreen from '../screens/SiteTimelineScreen';
import ChatListScreen from '../screens/ChatListScreen';
import ChatRoomScreen from '../screens/ChatRoomScreen';
import ProfileScreen from '../screens/ProfileScreen';

const Tab = createBottomTabNavigator();
const Stack = createNativeStackNavigator();

const TAB_ICONS = {
  진단: 'pulse',
  병원: 'location',
  모니터링: 'body',
  채팅: 'chatbubble',
  프로필: 'person',
};

// The diagnosis tab has an upload -> analyzing -> result flow, so it uses an inner stack
function DiagnosisStack() {
  return (
    <Stack.Navigator screenOptions={{ headerShown: false }}>
      <Stack.Screen name="DiagnosisHome" component={DiagnosisScreen} />
      <Stack.Screen name="Analyzing" component={AnalyzingScreen} />
      <Stack.Screen name="Result" component={ResultScreen} />
    </Stack.Navigator>
  );
}

function HospitalsStack() {
  return (
    <Stack.Navigator screenOptions={{ headerShown: false }}>
      <Stack.Screen name="HospitalsHome" component={HospitalsScreen} />
      <Stack.Screen name="HospitalDetail" component={HospitalDetailScreen} />
    </Stack.Navigator>
  );
}

function ChatStack() {
  return (
    <Stack.Navigator screenOptions={{ headerShown: false }}>
      <Stack.Screen name="ChatList" component={ChatListScreen} />
      <Stack.Screen name="ChatRoom" component={ChatRoomScreen} />
    </Stack.Navigator>
  );
}

// Monitoring: site list -> add site -> anchor capture -> timeline / session capture
function MonitoringStack() {
  return (
    <Stack.Navigator screenOptions={{ headerShown: false }}>
      <Stack.Screen name="MonitoringHome" component={MonitoringScreen} />
      <Stack.Screen name="AddSite" component={AddSiteScreen} />
      <Stack.Screen name="AnchorCapture" component={AnchorCaptureScreen} />
      <Stack.Screen name="SiteTimeline" component={SiteTimelineScreen} />
      <Stack.Screen name="GuidedCapture" component={GuidedCaptureScreen} />
    </Stack.Navigator>
  );
}

function MainTabs() {
  const insets = useSafeAreaInsets();   // 시스템 하단바(제스처/네비) 높이만큼 탭바를 위로 올린다
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarActiveTintColor: colors.primary,
        tabBarInactiveTintColor: colors.textMuted,
        tabBarStyle: { height: 64 + insets.bottom, paddingBottom: 8 + insets.bottom, paddingTop: 6 },
        tabBarLabelStyle: { fontSize: 11, fontWeight: '600' },
        tabBarIcon: ({ color, size }) => (
          <Ionicons name={TAB_ICONS[route.name]} size={size} color={color} />
        ),
      })}
    >
      <Tab.Screen name="진단" component={DiagnosisStack} />
      <Tab.Screen name="병원" component={HospitalsStack} />
      <Tab.Screen name="모니터링" component={MonitoringStack} />
      <Tab.Screen name="채팅" component={ChatStack} />
      <Tab.Screen name="프로필" component={ProfileScreen} />
    </Tab.Navigator>
  );
}

export default function RootNavigator() {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
        <ActivityIndicator color={colors.primary} size="large" />
      </View>
    );
  }

  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        {user ? (
          <Stack.Screen name="Main" component={MainTabs} />
        ) : (
          <>
            <Stack.Screen name="Login" component={LoginScreen} />
            <Stack.Screen name="RoleSelect" component={RoleSelectScreen} />
          </>
        )}
      </Stack.Navigator>
    </NavigationContainer>
  );
}

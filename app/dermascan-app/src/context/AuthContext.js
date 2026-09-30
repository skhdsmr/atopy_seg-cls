import React, { createContext, useContext, useState, useEffect } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { setAuthUser } from '../api/client';

const AuthContext = createContext(null);
const STORAGE_USER = '@dermascan/user';        // current login session
const STORAGE_ACCOUNTS = '@dermascan/accounts'; // registered accounts (remembers role per provider)

// Where real OAuth integration would go:
//  - Google : expo-auth-session/providers/google
//  - Kakao  : @react-native-seoul/kakao-login
//  - Naver  : @react-native-seoul/naver-login
// Here we only create a mock user per provider.
const MOCK_USERS = {
  google: { name: '오근완', email: 'ohgeunwan@gmail.com', provider: 'google' },
  naver: { name: '오근완', email: 'ohgeunwan@naver.com', provider: 'naver' },
  kakao: { name: '오근완', email: 'ohgeunwan@kakao.com', provider: 'kakao' },
};

function buildUser(provider, role) {
  const base = MOCK_USERS[provider] ?? MOCK_USERS.google;
  return role === 'doctor'
    ? { ...base, name: '김민준', email: base.email.replace('ohgeunwan', 'dr.kim'),
        role: 'doctor', patients: 12, consults: 48, hospital: '서울 피부과의원' }
    : { ...base, role: 'patient', diagnoses: 24, hospitals: 1, streak: 7 };
}

async function readAccounts() {
  try {
    const raw = await AsyncStorage.getItem(STORAGE_ACCOUNTS);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const raw = await AsyncStorage.getItem(STORAGE_USER);
        if (raw) {
          const u = JSON.parse(raw);
          setUser(u);
          setAuthUser(u.email);
        }
      } catch (e) {
        // ignore
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  async function applyUser(u) {
    setUser(u);
    setAuthUser(u.email);
    await AsyncStorage.setItem(STORAGE_USER, JSON.stringify(u));
  }

  // Attempt social login.
  //  - Existing account -> log in immediately with the stored role and return { isNew:false }
  //  - First-time signup -> don't log in, return { isNew:true } (caller navigates to the role selection screen)
  const startSocialLogin = async (provider) => {
    const accounts = await readAccounts();
    if (accounts[provider]) {
      await applyUser(accounts[provider]);
      return { isNew: false };
    }
    return { isNew: true, provider };
  };

  // Called from the role selection screen -> complete signup + log in
  const completeSignup = async (provider, role) => {
    const newUser = buildUser(provider, role);
    const accounts = await readAccounts();
    accounts[provider] = newUser;
    await AsyncStorage.setItem(STORAGE_ACCOUNTS, JSON.stringify(accounts));
    await applyUser(newUser);
    return newUser;
  };

  // Sign out (keeps the registered account record -> next login auto-assigns the role)
  const signOut = async () => {
    setUser(null);
    setAuthUser(null);
    await AsyncStorage.removeItem(STORAGE_USER);
  };

  // (dev only) Fully reset, including the signup records
  const resetAccounts = async () => {
    await AsyncStorage.multiRemove([STORAGE_USER, STORAGE_ACCOUNTS]);
    setUser(null);
    setAuthUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, loading, startSocialLogin, completeSignup, signOut, resetAccounts }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);

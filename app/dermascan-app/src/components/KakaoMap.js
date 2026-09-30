import React, { useMemo, useState } from 'react';
import { View, Text, StyleSheet, ActivityIndicator } from 'react-native';
import { WebView } from 'react-native-webview';
import { KAKAO_JS_KEY, KAKAO_MAP_BASE_URL } from '../config';
import { colors, radius } from '../theme';

// Renders Kakao Map in a WebView (drag/zoom enabled).
//  props: center {lat,lng}, hospitals [{id,name,lat,lng,distanceKm}], onSelect(id)
export default function KakaoMap({ center, hospitals = [], onSelect, style }) {
  const keyMissing = !KAKAO_JS_KEY || KAKAO_JS_KEY === 'YOUR_KAKAO_JAVASCRIPT_KEY';
  const [ready, setReady] = useState(false);
  const [mapError, setMapError] = useState(null);

  const html = useMemo(() => buildHtml(center, hospitals), [center, hospitals]);

  if (keyMissing) {
    return (
      <View style={[styles.placeholder, style]}>
        <Text style={styles.phTitle}>카카오맵 키 미설정</Text>
        <Text style={styles.phSub}>src/config.js 의 KAKAO_JS_KEY 를 입력하세요</Text>
      </View>
    );
  }

  return (
    <View style={[styles.wrap, style]}>
      <WebView
        originWhitelist={['*']}
        source={{ html, baseUrl: KAKAO_MAP_BASE_URL }}
        javaScriptEnabled
        domStorageEnabled
        mixedContentMode="always"
        onMessage={(e) => {
          let d;
          try { d = JSON.parse(e.nativeEvent.data); } catch { return; }
          if (d.type === 'ready') setReady(true);
          else if (d.type === 'error') setMapError(d.msg);
          else if (d.type === 'select' && onSelect) onSelect(d.id);
        }}
        onError={(e) => setMapError(e.nativeEvent.description || '지도 로드 실패')}
        style={{ backgroundColor: 'transparent' }}
      />

      {/* Loading / error overlay */}
      {!ready && !mapError && (
        <View style={styles.overlay}>
          <ActivityIndicator color={colors.primary} />
          <Text style={styles.overlayText}>지도 불러오는 중...</Text>
        </View>
      )}
      {mapError && (
        <View style={styles.overlay}>
          <Text style={styles.errTitle}>지도를 불러오지 못했습니다</Text>
          <Text style={styles.errMsg} numberOfLines={3}>{mapError}</Text>
          <Text style={styles.errHint}>
            카카오 콘솔 → 앱 → 플랫폼 → Web 사이트 도메인에{'\n'}
            {KAKAO_MAP_BASE_URL} 등록 여부를 확인하세요
          </Text>
        </View>
      )}
    </View>
  );
}

function buildHtml(center, hospitals) {
  const c = center || { lat: 37.4979, lng: 127.0276 };
  const markers = hospitals
    .filter((h) => h.lat && h.lng)
    .map((h) => ({ id: h.id, name: h.name, lat: h.lat, lng: h.lng, d: h.distanceKm }));

  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no" />
  <!-- Suppress the Referer header to bypass Kakao's domain check (standard WebView workaround).
       This lets the map load without registering the JS SDK domain. -->
  <meta name="referrer" content="no-referrer" />
  <style>html,body,#map{margin:0;padding:0;width:100%;height:100%;}</style>
</head>
<body>
  <div id="map"></div>
  <script>
    function post(o){ if(window.ReactNativeWebView) window.ReactNativeWebView.postMessage(JSON.stringify(o)); }
    window.onerror = function(msg){ post({type:'error', msg:String(msg)}); return true; };

    var s = document.createElement('script');
    s.referrerPolicy = 'no-referrer';
    s.src = 'https://dapi.kakao.com/v2/maps/sdk.js?appkey=${KAKAO_JS_KEY}&autoload=false';
    s.onerror = function(){ post({type:'error', msg:'SDK 스크립트 로드 실패 (네트워크/도메인 확인)'}); };
    s.onload = function(){
      try {
        kakao.maps.load(function(){
          var center = new kakao.maps.LatLng(${c.lat}, ${c.lng});
          var map = new kakao.maps.Map(document.getElementById('map'), { center: center, level: 5 });
          new kakao.maps.Marker({ map: map, position: center });

          var hospitals = ${JSON.stringify(markers)};
          var bounds = new kakao.maps.LatLngBounds();
          bounds.extend(center);
          hospitals.forEach(function(h){
            var pos = new kakao.maps.LatLng(h.lat, h.lng);
            bounds.extend(pos);
            var marker = new kakao.maps.Marker({ map: map, position: pos, title: h.name });
            new kakao.maps.CustomOverlay({
              map: map, position: pos, yAnchor: 2.2,
              content: '<div style="background:#fff;border:1px solid #E3E9E6;border-radius:12px;padding:2px 8px;font-size:11px;font-weight:700;color:#2F6B5A;white-space:nowrap;">' + (h.d != null ? h.d + 'km' : h.name) + '</div>'
            });
            kakao.maps.event.addListener(marker, 'click', function(){ post({type:'select', id:h.id}); });
          });
          if (hospitals.length) map.setBounds(bounds);
          post({type:'ready'});
        });
      } catch(e){ post({type:'error', msg:String(e)}); }
    };
    document.head.appendChild(s);
  </script>
</body>
</html>`;
}

const styles = StyleSheet.create({
  wrap: { borderRadius: radius.lg, overflow: 'hidden', backgroundColor: colors.primarySoft },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center', justifyContent: 'center', padding: 16,
    backgroundColor: colors.primarySoft,
  },
  overlayText: { fontSize: 13, color: colors.textSub, marginTop: 8 },
  errTitle: { fontSize: 14, fontWeight: '700', color: colors.text },
  errMsg: { fontSize: 12, color: colors.severe, marginTop: 6, textAlign: 'center' },
  errHint: { fontSize: 12, color: colors.textSub, marginTop: 10, textAlign: 'center', lineHeight: 18 },
  placeholder: {
    borderRadius: radius.lg, backgroundColor: colors.primarySoft,
    alignItems: 'center', justifyContent: 'center', padding: 16,
  },
  phTitle: { fontSize: 14, fontWeight: '700', color: colors.primary },
  phSub: { fontSize: 12, color: colors.textSub, marginTop: 4, textAlign: 'center' },
});

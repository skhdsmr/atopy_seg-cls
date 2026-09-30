#!/usr/bin/env bash
B="http://127.0.0.1:8000"
H="X-User-Id: tester@example.com"
cd /mnt/c/hsbioMVP || exit 1

echo "=== 부위 생성 ==="
SID=$(curl -s -X POST "$B/sites" -H "$H" -H "Content-Type: application/json" \
  -d '{"bodyPart":"전완","side":"L","label":"좌측 팔뚝 찰상","condition":"uremic"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "site id: $SID"

IMGS=( $(ls SkinDisNet_balanced/test/CD/*.jpg | head -2) )

echo ""
echo "=== 1차 촬영 (기준) ==="
curl -s -X POST "$B/sites/$SID/captures" -H "$H" -F "file=@${IMGS[0]}" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(' isReference:', d.get('isReference'), '| qualityOk:', d.get('qualityOk'), '| brightness:', d.get('brightness'), '| blur:', d.get('blur'), '| erythema:', d.get('erythema'), '| reg:', d.get('registration'))"

echo ""
echo "=== 2차 촬영 (정합) ==="
curl -s -X POST "$B/sites/$SID/captures" -H "$H" -F "file=@${IMGS[1]}" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(' isReference:', d.get('isReference'), '| qualityOk:', d.get('qualityOk'), '| erythema:', d.get('erythema'), '| reg:', d.get('registration'))"

echo ""
echo "=== 타임라인 ==="
curl -s "$B/sites/$SID/captures" | python3 -c "import sys,json; d=json.load(sys.stdin); print(' 총', len(d), '건'); [print('  ', c['date'], 'erythema', c['erythema'], 'ok', c['qualityOk']) for c in d]"

echo ""
echo "=== 기준 이미지(고스트) 존재 확인 ==="
curl -s "$B/sites/$SID/reference" | python3 -c "import sys,json; d=json.load(sys.stdin); u=d.get('referenceUrl'); print(' referenceUrl 길이:', len(u) if u else 0)"

echo ""
echo "=== 부위 목록 ==="
curl -s "$B/sites" -H "$H" | python3 -c "import sys,json; d=json.load(sys.stdin); [print('  ', s['bodyPart'], s['side'], '| 촬영', s['captureCount'], '| 최근홍반', s['lastErythema']) for s in d]"

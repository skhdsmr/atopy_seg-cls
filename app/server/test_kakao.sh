#!/usr/bin/env bash
echo "=== /health ==="
curl -s -m 8 http://127.0.0.1:8000/health
echo ""
echo "=== /hospitals (강남역 기준 실제 피부과) ==="
curl -s -m 12 'http://127.0.0.1:8000/hospitals?lat=37.4979&lng=127.0276' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('총', len(d), '개'); [print(f\"  {h['name']:20s} {h.get('distanceKm')}km  {h.get('phone','')}\") for h in d[:8]]"

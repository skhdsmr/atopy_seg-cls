#!/usr/bin/env bash
cd /mnt/c/hsbioMVP || exit 1
echo "=== /health ==="
curl -s -w "\n[HTTP %{http_code}]\n" http://127.0.0.1:8000/health
echo ""
echo "=== /predict (테스트 이미지 1장) ==="
IMG=$(ls SkinDisNet_balanced/test/CD/*.jpg 2>/dev/null | head -1)
echo "이미지: $IMG"
curl -s -w "\n[HTTP %{http_code}]\n" -X POST http://127.0.0.1:8000/predict -F "file=@${IMG}" -o /tmp/pred.json
echo "--- 응답 앞부분 ---"
head -c 400 /tmp/pred.json
echo ""

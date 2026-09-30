# ============================================================
# Server configuration (external API keys) — example template
#   Copy this file to config.py in the same folder and fill in your real key.
#   (config.py with the real key is gitignored and not committed.)
# ============================================================

# Kakao REST API key (different from the Map JavaScript key!)
#   developers.kakao.com → My Application → (app) → App Keys → copy "REST API Key"
#   Setting the KAKAO_REST_KEY environment variable takes precedence.
import os
KAKAO_REST_KEY = os.environ.get('KAKAO_REST_KEY', 'YOUR_KAKAO_REST_API_KEY')

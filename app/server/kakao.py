"""Kakao Local API — looks up nearby dermatology clinics via keyword place search.

Requires a REST API key (different from the Map JS key); without it ENABLED=False
and app.py falls back to seed hospitals.
"""
import json
import urllib.request
import urllib.parse

from config import KAKAO_REST_KEY

ENABLED = bool(KAKAO_REST_KEY) and KAKAO_REST_KEY != 'YOUR_KAKAO_REST_API_KEY'

_KEYWORD_URL = 'https://dapi.kakao.com/v2/local/search/keyword.json'


def search_dermatology(lat, lng, radius=5000, size=15, query='피부과'):
    """Return dermatology clinics within the radius of the current coordinates (lat, lng), ordered by distance."""
    params = urllib.parse.urlencode({
        'query': query,
        'x': lng,            # Kakao uses x=longitude, y=latitude
        'y': lat,
        'radius': radius,    # Meters (max 20000)
        'sort': 'distance',
        'size': size,        # Max 15
    })
    req = urllib.request.Request(
        f'{_KEYWORD_URL}?{params}',
        headers={'Authorization': f'KakaoAK {KAKAO_REST_KEY}'},
    )
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read().decode('utf-8'))

    out = []
    for d in data.get('documents', []):
        dept = (d.get('category_name', '').split('>')[-1].strip() or '피부과')
        dist_m = d.get('distance')
        out.append({
            'id': d['id'],
            'name': d['place_name'],
            'dept': dept,
            'doctor': '',                 # Kakao Local API has no doctor info
            'rating': None,               # No rating
            'status': '',                 # No real-time clinic status
            'address': d.get('road_address_name') or d.get('address_name', ''),
            'phone': d.get('phone', ''),
            'lat': float(d['y']),
            'lng': float(d['x']),
            'placeUrl': d.get('place_url', ''),   # KakaoMap detail link
            'distanceKm': round(int(dist_m) / 1000, 1) if dist_m else None,
            'registered': False,
        })
    return out

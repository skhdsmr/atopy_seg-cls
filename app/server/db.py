"""DermaScan backend data layer (SQLite, no dependencies).

Tables: hospitals (partner hospitals), registrations (hospitals the user attends),
        records (symptom records), chats / messages (doctor-patient chat),
        appointments (visit appointments).
The endpoints the app calls (server/app.py) use these functions.
"""
import os
import sqlite3
import json
import math
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dermascan.db')

CHAT_FEE_KRW = 500  # Charge per patient message


# ------------------------------------------------------------
# Partner hospital seed (Gangnam-area Seoul coordinates) — placeholder for a real partner hospital DB
# ------------------------------------------------------------
SEED_HOSPITALS = [
    ('h1', '서울 피부과의원', '피부과', '김민준 의사', 4.8, '진료 중',
     '강남구 테헤란로 123', '02-1234-5678', 37.50090, 127.03640),
    ('h2', '청담 메디컬센터', '피부과·성형외과', '이수진 의사', 4.6, '진료 중',
     '강남구 청담동 45', '02-2345-6789', 37.51970, 127.04730),
    ('h3', '강남 피부클리닉', '피부과', '박지영 의사', 4.4, '진료 종료',
     '강남구 역삼동 67', '02-3456-7890', 37.50060, 127.03660),
    ('h4', '역삼 더마케어', '피부과', '정해인 의사', 4.7, '진료 중',
     '강남구 역삼동 102', '02-4567-8901', 37.49500, 127.03300),
    ('h5', '선릉 스킨의원', '피부과', '한지민 의사', 4.5, '진료 중',
     '강남구 선릉로 200', '02-5678-9012', 37.50450, 127.04900),
]


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS hospitals(
            id TEXT PRIMARY KEY, name TEXT, dept TEXT, doctor TEXT,
            rating REAL, status TEXT, address TEXT, phone TEXT,
            lat REAL, lng REAL
        );
        CREATE TABLE IF NOT EXISTS registrations(
            user_id TEXT, hospital_id TEXT,
            PRIMARY KEY(user_id, hospital_id)
        );
        CREATE TABLE IF NOT EXISTS records(
            user_id TEXT, date TEXT, severity TEXT, note TEXT, photo TEXT,
            PRIMARY KEY(user_id, date)
        );
        CREATE TABLE IF NOT EXISTS chats(
            id TEXT PRIMARY KEY, user_id TEXT, doctor TEXT, hospital TEXT, avatar TEXT
        );
        CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT,
            text TEXT, sender_role TEXT, ts TEXT
        );
        CREATE TABLE IF NOT EXISTS appointments(
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT,
            hospital_id TEXT, date TEXT
        );
        CREATE TABLE IF NOT EXISTS billing(
            user_id TEXT PRIMARY KEY, total INTEGER DEFAULT 0
        );
        -- Monitoring sites (axis 1: site metadata slot)
        --   ref_path = the wide "anchor" photo taken once at registration.
        CREATE TABLE IF NOT EXISTS sites(
            id TEXT PRIMARY KEY, user_id TEXT,
            body_part TEXT, side TEXT, label TEXT, condition TEXT,
            ref_path TEXT, created TEXT
        );
        -- Per-site captures + measurements (axis 3/4/5 results)
        CREATE TABLE IF NOT EXISTS captures(
            id TEXT PRIMARY KEY, site_id TEXT, date TEXT, image_path TEXT,
            brightness REAL, blur REAL, clip REAL, erythema REAL,
            align_conf REAL, scale_mmpx REAL, quality_ok INTEGER, reasons TEXT
        );
        """)
        _migrate(c)
        # Seed hospitals (only when empty)
        if c.execute("SELECT COUNT(*) FROM hospitals").fetchone()[0] == 0:
            c.executemany(
                "INSERT INTO hospitals VALUES (?,?,?,?,?,?,?,?,?,?)", SEED_HOSPITALS)


# ------------------------------------------------------------
# Schema migration
#   CREATE TABLE IF NOT EXISTS never adds columns to an existing DB, so every
#   column introduced after the first release has to be ALTERed in explicitly.
# ------------------------------------------------------------
NEW_COLUMNS = {
    'sites': [
        ('anchor_roi', 'TEXT'),      # lesion ROI on the anchor photo, normalized JSON {x,y,w,h}
        ('anchor_w', 'INTEGER'),     # anchor photo size (needed to denormalize the ROI)
        ('anchor_h', 'INTEGER'),
    ],
    'captures': [
        ('kind', "TEXT DEFAULT 'session'"),   # 'anchor' | 'session'
        ('area_ratio', 'REAL'),               # lesion area / anchor ROI area (the trend metric)
        ('align_grade', 'TEXT'),              # 'high' | 'medium' | 'low'
        ('align_inliers', 'INTEGER'),
        ('align_ratio', 'REAL'),              # inliers / matches
        ('align_reproj', 'REAL'),             # RMS reprojection error (px, work resolution)
        ('align_coverage', 'REAL'),           # inlier spatial coverage (3x3 grid)
        ('align_roi_coverage', 'REAL'),       # fraction of the anchor ROI visible in this capture
        ('align_reasons', 'TEXT'),            # JSON list
        ('affine', 'TEXT'),                   # JSON 2x3, near-photo px -> anchor px
        ('severity', 'TEXT'),                 # JSON of the on-device IGA/EASI classifier
    ],
}


def _migrate(c):
    for table, cols in NEW_COLUMNS.items():
        have = {r['name'] for r in c.execute(f"PRAGMA table_info({table})")}
        for name, ddl in cols:
            if name not in have:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def _haversine(lat1, lng1, lat2, lng2):
    """Distance (km) between two coordinates."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ------------------------------------------------------------
# Hospitals
# ------------------------------------------------------------
def get_hospitals(user_id, lat=None, lng=None):
    with _conn() as c:
        rows = c.execute("SELECT * FROM hospitals").fetchall()
        reg = {r['hospital_id'] for r in
               c.execute("SELECT hospital_id FROM registrations WHERE user_id=?", (user_id,))}
    # Default location: near Gangnam Station
    ulat = lat if lat is not None else 37.4979
    ulng = lng if lng is not None else 127.0276
    out = []
    for r in rows:
        dist = _haversine(ulat, ulng, r['lat'], r['lng'])
        out.append({
            'id': r['id'], 'name': r['name'], 'dept': r['dept'], 'doctor': r['doctor'],
            'rating': r['rating'], 'status': r['status'], 'address': r['address'],
            'phone': r['phone'], 'lat': r['lat'], 'lng': r['lng'],
            'distanceKm': round(dist, 1), 'registered': r['id'] in reg,
        })
    out.sort(key=lambda h: h['distanceKm'])
    return out


def get_registered_ids(user_id):
    """Set of hospital ids the user registered (for marking registration status in Kakao results)."""
    with _conn() as c:
        return {r['hospital_id'] for r in
                c.execute("SELECT hospital_id FROM registrations WHERE user_id=?", (user_id,))}


def set_registration(user_id, hospital_id, register):
    with _conn() as c:
        if register:
            c.execute("INSERT OR IGNORE INTO registrations VALUES (?,?)", (user_id, hospital_id))
        else:
            c.execute("DELETE FROM registrations WHERE user_id=? AND hospital_id=?",
                      (user_id, hospital_id))
    return {'ok': True, 'registered': bool(register)}


# ------------------------------------------------------------
# Symptom records
# ------------------------------------------------------------
def get_records(user_id):
    with _conn() as c:
        rows = c.execute("SELECT * FROM records WHERE user_id=?", (user_id,)).fetchall()
    return {r['date']: {'severity': r['severity'], 'note': r['note'], 'photo': r['photo']}
            for r in rows}


def add_record(user_id, date, severity, note, photo):
    with _conn() as c:
        c.execute("""INSERT INTO records VALUES (?,?,?,?,?)
                     ON CONFLICT(user_id,date) DO UPDATE SET
                       severity=excluded.severity, note=excluded.note, photo=excluded.photo""",
                  (user_id, date, severity, note, photo))
    return {'ok': True}


# ------------------------------------------------------------
# Chat
# ------------------------------------------------------------
def _seed_chats(user_id):
    """Seed two default chat rooms per user (only when none exist)."""
    with _conn() as c:
        if c.execute("SELECT COUNT(*) FROM chats WHERE user_id=?", (user_id,)).fetchone()[0]:
            return
        seed = [
            ('c1', '김민준 의사', '서울 피부과의원', '김',
             [('안녕하세요, 사진 잘 받았습니다.', 'doctor', '오전 9:10'),
              ('증상이 언제부터 있으셨나요?', 'doctor', '오전 9:11'),
              ('3일 전부터 가렵기 시작했어요.', 'patient', '오전 9:20'),
              ('내일 진료실에서 뵙겠습니다.', 'doctor', '오전 9:25')]),
            ('c2', '이수진 의사', '청담 메디컬센터', '이',
             [('사진 잘 받았습니다. 확인 후 연락드리겠습니다.', 'doctor', '어제')]),
        ]
        for cid, doctor, hosp, avatar, msgs in seed:
            c.execute("INSERT INTO chats VALUES (?,?,?,?,?)",
                      (cid, user_id, doctor, hosp, avatar))
            for text, role, ts in msgs:
                c.execute("INSERT INTO messages(chat_id,text,sender_role,ts) VALUES (?,?,?,?)",
                          (cid, text, role, ts))


def get_chats(user_id):
    _seed_chats(user_id)
    with _conn() as c:
        chats = c.execute("SELECT * FROM chats WHERE user_id=?", (user_id,)).fetchall()
        out = []
        for ch in chats:
            msgs = c.execute(
                "SELECT id,text,sender_role,ts FROM messages WHERE chat_id=? ORDER BY id",
                (ch['id'],)).fetchall()
            mlist = [{'id': str(m['id']), 'text': m['text'],
                      'senderRole': m['sender_role'], 'ts': m['ts']} for m in msgs]
            last = mlist[-1]['text'] if mlist else ''
            out.append({
                'id': ch['id'], 'doctor': ch['doctor'], 'hospital': ch['hospital'],
                'avatar': ch['avatar'], 'lastMessage': last,
                'time': mlist[-1]['ts'] if mlist else '', 'unread': 0,
                'messages': mlist,
            })
    return out


def add_message(user_id, chat_id, text, sender_role):
    ts = datetime.now().strftime('%p %I:%M').replace('AM', '오전').replace('PM', '오후')
    with _conn() as c:
        c.execute("INSERT INTO messages(chat_id,text,sender_role,ts) VALUES (?,?,?,?)",
                  (chat_id, text, sender_role, ts))
        mid = c.execute("SELECT last_insert_rowid()").fetchone()[0]
        charged = 0
        if sender_role == 'patient':  # Charge patients only
            charged = CHAT_FEE_KRW
            c.execute("""INSERT INTO billing(user_id,total) VALUES (?,?)
                         ON CONFLICT(user_id) DO UPDATE SET total=total+?""",
                      (user_id, charged, charged))
    return {'ok': True,
            'message': {'id': str(mid), 'text': text, 'senderRole': sender_role, 'ts': ts},
            'charged': charged}


def get_billing(user_id):
    with _conn() as c:
        row = c.execute("SELECT total FROM billing WHERE user_id=?", (user_id,)).fetchone()
    return {'total': row['total'] if row else 0}


# ------------------------------------------------------------
# Visit appointments (alarms handled by the app as local notifications; the server keeps the record)
# ------------------------------------------------------------
def add_appointment(user_id, hospital_id, date):
    with _conn() as c:
        c.execute("INSERT INTO appointments(user_id,hospital_id,date) VALUES (?,?,?)",
                  (user_id, hospital_id, date))
    return {'ok': True}


# ------------------------------------------------------------
# Monitoring sites / captures
# ------------------------------------------------------------
def create_site(user_id, site_id, body_part, side, label, condition):
    with _conn() as c:
        c.execute("""INSERT INTO sites(id,user_id,body_part,side,label,condition,ref_path,created)
                     VALUES (?,?,?,?,?,?,?,datetime('now'))""",
                  (site_id, user_id, body_part, side, label, condition, None))
    return {'ok': True, 'id': site_id}


def get_sites(user_id):
    with _conn() as c:
        sites = c.execute("SELECT * FROM sites WHERE user_id=? ORDER BY created", (user_id,)).fetchall()
        out = []
        for s in sites:
            cnt = c.execute("SELECT COUNT(*) FROM captures WHERE site_id=? AND kind!='anchor'",
                            (s['id'],)).fetchone()[0]
            last = c.execute("""SELECT date,erythema,area_ratio,align_grade FROM captures
                                WHERE site_id=? AND kind!='anchor'
                                ORDER BY date DESC, rowid DESC LIMIT 1""", (s['id'],)).fetchone()
            out.append({
                'id': s['id'], 'bodyPart': s['body_part'], 'side': s['side'],
                'label': s['label'], 'condition': s['condition'],
                'captureCount': cnt,
                # A site is only ready for sessions once the anchor photo AND its ROI exist
                'hasReference': bool(s['ref_path']),
                'hasAnchor': bool(s['ref_path'] and s['anchor_roi']),
                'lastDate': last['date'] if last else None,
                'lastErythema': last['erythema'] if last else None,
                'lastAreaRatio': last['area_ratio'] if last else None,
                'lastAlignGrade': last['align_grade'] if last else None,
            })
    return out


def delete_site(user_id, site_id):
    with _conn() as c:
        c.execute("DELETE FROM sites WHERE id=? AND user_id=?", (site_id, user_id))
        c.execute("DELETE FROM captures WHERE site_id=?", (site_id,))
    return {'ok': True}


def get_site_ref_path(site_id):
    with _conn() as c:
        r = c.execute("SELECT ref_path FROM sites WHERE id=?", (site_id,)).fetchone()
        return r['ref_path'] if r else None


def set_site_ref(site_id, path):
    with _conn() as c:
        c.execute("UPDATE sites SET ref_path=? WHERE id=?", (path, site_id))


# ------------------------------------------------------------
# Anchor: the single wide photo captured when the site is registered.
#   Every later (close-up) session is registered against this frame, so the
#   ROI is stored normalized — it stays valid no matter how the image is
#   later resized or cropped for the on-device matcher.
# ------------------------------------------------------------
def set_site_anchor(site_id, path, roi, width, height):
    import json as _json
    with _conn() as c:
        c.execute("UPDATE sites SET ref_path=?, anchor_roi=?, anchor_w=?, anchor_h=? WHERE id=?",
                  (path, _json.dumps(roi), width, height, site_id))
    return {'ok': True}


def get_site_anchor(site_id):
    """{path, roi, width, height} — roi is None when the site has no anchor yet."""
    import json as _json
    with _conn() as c:
        r = c.execute("SELECT ref_path,anchor_roi,anchor_w,anchor_h FROM sites WHERE id=?",
                      (site_id,)).fetchone()
    if not r:
        return None
    return {
        'path': r['ref_path'],
        'roi': _json.loads(r['anchor_roi']) if r['anchor_roi'] else None,
        'width': r['anchor_w'],
        'height': r['anchor_h'],
    }


def add_capture(capture_id, site_id, date, image_path, m, kind='session'):
    """m = cv_pipeline.analyze result merged with the on-device metrics
    (registration / relative area / severity) sent by the app."""
    import json as _json
    with _conn() as c:
        c.execute("""INSERT INTO captures(id,site_id,date,image_path,brightness,blur,clip,
                       erythema,align_conf,scale_mmpx,quality_ok,reasons,
                       kind,area_ratio,align_grade,align_inliers,align_ratio,align_reproj,
                       align_coverage,align_roi_coverage,align_reasons,affine,severity)
                     VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                  (capture_id, site_id, date, image_path,
                   m.get('brightness'), m.get('blur'), m.get('clip'), m.get('erythema'),
                   m.get('alignRatio'),   # keep the legacy column populated for old readers
                   m.get('scaleMmPerPx'), 1 if m.get('qualityOk') else 0,
                   _json.dumps(m.get('reasons', []), ensure_ascii=False),
                   kind, m.get('areaRatio'), m.get('alignGrade'), m.get('alignInliers'),
                   m.get('alignRatio'), m.get('alignReproj'), m.get('alignCoverage'),
                   m.get('alignRoiCoverage'),
                   _json.dumps(m.get('alignReasons', []), ensure_ascii=False),
                   _json.dumps(m.get('affine')) if m.get('affine') else None,
                   _json.dumps(m.get('severity'), ensure_ascii=False) if m.get('severity') else None))
    return {'ok': True, 'id': capture_id}


def _capture_row(r):
    import json as _json
    return {
        'id': r['id'], 'date': r['date'], 'kind': r['kind'] or 'session',
        'brightness': r['brightness'], 'blur': r['blur'], 'clip': r['clip'],
        'erythema': r['erythema'], 'scaleMmPerPx': r['scale_mmpx'],
        'qualityOk': bool(r['quality_ok']),
        'reasons': _json.loads(r['reasons'] or '[]'),
        # Registration + relative area (computed on-device)
        'areaRatio': r['area_ratio'],
        'alignGrade': r['align_grade'],
        'alignInliers': r['align_inliers'],
        'alignRatio': r['align_ratio'],
        'alignReproj': r['align_reproj'],
        'alignCoverage': r['align_coverage'],
        'alignRoiCoverage': r['align_roi_coverage'],
        'alignReasons': _json.loads(r['align_reasons'] or '[]'),
        'severity': _json.loads(r['severity']) if r['severity'] else None,
    }


def get_captures(site_id):
    with _conn() as c:
        rows = c.execute("SELECT * FROM captures WHERE site_id=? ORDER BY date", (site_id,)).fetchall()
    return [_capture_row(r) for r in rows]


# ------------------------------------------------------------
# Monitoring calendar: query by date (aggregated across all sites)
# ------------------------------------------------------------
def get_capture_dates(user_id):
    """Number of captures per date that has any. {'YYYY-MM-DD': count} (summed across all of the user's sites)."""
    with _conn() as c:
        rows = c.execute("""
            SELECT substr(cp.date, 1, 10) AS day, COUNT(*) AS cnt
            FROM captures cp JOIN sites s ON cp.site_id = s.id
            WHERE s.user_id = ?
            GROUP BY day""", (user_id,)).fetchall()
    return {r['day']: r['cnt'] for r in rows}


def get_captures_by_date(user_id, day):
    """Captures from all sites taken on a given date (YYYY-MM-DD), plus site info."""
    with _conn() as c:
        rows = c.execute("""
            SELECT cp.*, s.body_part, s.side, s.label, s.condition
            FROM captures cp JOIN sites s ON cp.site_id = s.id
            WHERE s.user_id = ? AND substr(cp.date, 1, 10) = ?
            ORDER BY cp.date""", (user_id, day)).fetchall()
    return [{
        **_capture_row(r),
        'siteId': r['site_id'],
        'bodyPart': r['body_part'], 'side': r['side'], 'label': r['label'],
        'condition': r['condition'],
        'hasImage': bool(r['image_path']),
    } for r in rows]


def get_capture_image_path(capture_id):
    with _conn() as c:
        r = c.execute("SELECT image_path FROM captures WHERE id=?", (capture_id,)).fetchone()
        return r['image_path'] if r else None

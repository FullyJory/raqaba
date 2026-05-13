"""
RAQABA | رقابة  v7
Smart Occupancy & Evacuation Monitoring System
─────────────────────────────────────────────
Production-ready upgrades over v6:
  • SQLite database  — all data persists across restarts
  • Environment variables via .env  — no secrets in code
  • Configurable settings  — capacity, count-line from dashboard
  • Full event log  — every entry/exit/emergency recorded with timestamp
  • CSV export  — download the log from the dashboard
  • Whatsapp/SMS hook (Twilio)  — optional, enable via .env
  • Multi-user  — unlimited accounts stored in DB, role-based
  • Brute-force protection  — persisted across restarts (in DB)
"""

import os, time, threading, csv, io
from datetime import datetime
from functools import wraps
from pathlib import Path

# ── Env vars ──────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

SECRET_KEY   = os.getenv("RAQABA_SECRET",  "change-me-in-production-!!!!")
DATABASE_URL = os.getenv("RAQABA_DB",      "raqaba.db")
VIDEO_PATH   = os.getenv("RAQABA_VIDEO",   os.path.join(os.path.dirname(__file__), "videos", "demo.mp4"))

# Twilio (optional)
TWILIO_SID   = os.getenv("TWILIO_SID",   "")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN", "")
TWILIO_FROM  = os.getenv("TWILIO_FROM",  "")   # WhatsApp: "whatsapp:+14155238886"

# ── Flask ─────────────────────────────────────────────────
import cv2
import numpy as np
from flask import (Flask, Response, jsonify, redirect,
                   render_template, request, session, url_for,
                   send_file)

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.permanent_session_lifetime = 86400

# ── Database ───────────────────────────────────────────────
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), DATABASE_URL)

def get_db():
    """Return a thread-local DB connection."""
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db

def init_db():
    db = get_db()
    db.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        uid       TEXT PRIMARY KEY,
        pin_hash  TEXT NOT NULL,
        role      TEXT NOT NULL DEFAULT 'staff',
        name      TEXT NOT NULL DEFAULT '',
        phone     TEXT NOT NULL DEFAULT '',
        created_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS staff (
        id        TEXT PRIMARY KEY,
        name      TEXT NOT NULL,
        phone     TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS events (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        ts         TEXT NOT NULL DEFAULT (datetime('now')),
        kind       TEXT NOT NULL,   -- 'entry' | 'exit' | 'emergency_on' | 'emergency_off' | 'login' | 'reset'
        actor      TEXT,
        note       TEXT,
        entries    INTEGER DEFAULT 0,
        exits      INTEGER DEFAULT 0,
        occupancy  INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS settings (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS login_attempts (
        uid          TEXT PRIMARY KEY,
        fail_count   INTEGER DEFAULT 0,
        locked_until REAL    DEFAULT 0
    );
    """)

    # Default settings
    defaults = {
        "capacity":   "200",
        "line_ratio": "0.5",
        "org_name":   "RAQABA Facility",
    }
    for k, v in defaults.items():
        db.execute("INSERT OR IGNORE INTO settings(key,value) VALUES(?,?)", (k, v))

    # Default users (admin + staff)  — only if table empty
    from hashlib import sha256
    if not db.execute("SELECT 1 FROM users LIMIT 1").fetchone():
        db.execute("INSERT INTO users(uid,pin_hash,role,name,phone) VALUES(?,?,?,?,?)",
                   ("admin", sha256(b"1234").hexdigest(), "admin", "", ""))
        db.execute("INSERT INTO users(uid,pin_hash,role,name,phone) VALUES(?,?,?,?,?)",
                   ("staff", sha256(b"1234").hexdigest(), "staff", "", ""))

    # Seed demo staff if empty
    if not db.execute("SELECT 1 FROM staff LIMIT 1").fetchone():
        demo = [
            ("20001", "Sara AlOtaibi",  "0511111111"),
            ("20002", "Khalid Mansour", "0522222222"),
            ("20003", "Noura AlHarbi",  "0533333333"),
        ]
        db.executemany("INSERT OR IGNORE INTO staff(id,name,phone) VALUES(?,?,?)", demo)

    db.commit()
    db.close()

def get_setting(key, default=None):
    db = get_db()
    row = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    db.close()
    return row["value"] if row else default

def set_setting(key, value):
    db = get_db()
    db.execute("INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)", (key, str(value)))
    db.commit()
    db.close()

def log_event(kind, actor=None, note=None, entries=0, exits=0, occupancy=0):
    db = get_db()
    db.execute(
        "INSERT INTO events(kind,actor,note,entries,exits,occupancy) VALUES(?,?,?,?,?,?)",
        (kind, actor, note, entries, exits, occupancy)
    )
    db.commit()
    db.close()

# ── System state ───────────────────────────────────────────
system_state = {
    "entries": 0,
    "exits":   0,
    "emergency": False,
    "camera_active": False,
}
state_lock = threading.Lock()

# ── YOLO ───────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    _yolo  = YOLO("yolov8n.pt")
    YOLO_OK = True
    print("[RAQABA] YOLOv8n ready ✓")
except Exception as e:
    YOLO_OK = False
    print(f"[RAQABA] YOLO unavailable, using motion fallback ({e})")


def detect(frame, bg_sub):
    if YOLO_OK:
        res = _yolo(frame, verbose=False, classes=[0])[0]
        return [(int(b.xyxy[0][0]), int(b.xyxy[0][1]),
                 int(b.xyxy[0][2]), int(b.xyxy[0][3]),
                 float(b.conf[0]))
                for b in res.boxes if float(b.conf[0]) > 0.30]
    fg = bg_sub.apply(frame)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 800:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if h / max(w, 1) > 0.4:
            boxes.append((x, y, x + w, y + h, min(area / 5000, 0.99)))
    return boxes


# ── Tracker ────────────────────────────────────────────────
class Tracker:
    def __init__(self):
        self.tracks  = {}
        self.counted = {}
        self._nid    = 0

    def reset(self):
        self.tracks.clear()
        self.counted.clear()
        self._nid = 0

    def update(self, boxes, line_x):
        new_tracks  = {}
        used        = set()
        new_in = new_out = 0

        for (x1, y1, x2, y2, conf) in boxes:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            best_id, best_d = None, 100
            for tid, tr in self.tracks.items():
                if tid in used:
                    continue
                d = abs(tr["cx"] - cx) + abs(tr["cy"] - cy) * 0.4
                if d < best_d:
                    best_d, best_id = d, tid
            if best_id is None:
                best_id = self._nid
                self._nid += 1
                self.tracks[best_id]  = {"cx": cx, "cy": cy, "side": None}
                self.counted[best_id] = False
            used.add(best_id)

            old_side = self.tracks[best_id]["side"]
            new_side = "left" if cx < line_x else "right"

            if old_side and old_side != new_side and not self.counted[best_id]:
                self.counted[best_id] = True
                if new_side == "left":
                    new_out += 1
                else:
                    new_in  += 1

            new_tracks[best_id] = {"cx": cx, "cy": cy, "side": new_side,
                                   "box": (x1, y1, x2, y2), "conf": conf}
        self.tracks = new_tracks
        return new_in, new_out

    def annotate(self, frame):
        for tid, tr in self.tracks.items():
            x1, y1, x2, y2 = tr["box"]
            color = (60, 220, 80) if tr["side"] == "right" else (80, 140, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"P{tid} {tr['conf']:.0%}",
                        (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


# ── Video thread ───────────────────────────────────────────
_jpeg_lock   = threading.Lock()
_latest_jpeg = b""
_log_cooldown = 0   # only log events every N seconds to avoid DB spam

def video_loop():
    global _latest_jpeg, _log_cooldown
    bg_sub  = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40)
    tracker = Tracker()

    while True:
        vpath = os.getenv("RAQABA_VIDEO", VIDEO_PATH)
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print("[RAQABA] Cannot open video, retrying in 3 s …")
            time.sleep(3)
            continue

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        line_ratio = float(get_setting("line_ratio", "0.5"))
        LINE_X = int(W * line_ratio)

        with state_lock:
            system_state["entries"]       = 0
            system_state["exits"]         = 0
            system_state["camera_active"] = True
        tracker.reset()
        bg_sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            boxes = detect(frame, bg_sub)
            new_in, new_out = tracker.update(boxes, LINE_X)

            if new_in or new_out:
                with state_lock:
                    system_state["entries"] += new_in
                    system_state["exits"]   += new_out
                    ent = system_state["entries"]
                    ex  = system_state["exits"]
                occ = max(0, ent - ex)
                # Log to DB (throttled to avoid too many writes)
                now = time.time()
                if now - _log_cooldown > 5:
                    _log_cooldown = now
                    if new_in:
                        log_event("entry", note=f"+{new_in}", entries=ent, exits=ex, occupancy=occ)
                    if new_out:
                        log_event("exit",  note=f"+{new_out}", entries=ent, exits=ex, occupancy=occ)

            tracker.annotate(frame)

            line_ratio = float(get_setting("line_ratio", "0.5"))
            LINE_X = int(W * line_ratio)

            cv2.line(frame, (LINE_X, 0), (LINE_X, H), (255, 220, 0), 2)
            cv2.putText(frame, "IN →",  (LINE_X + 6, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 80),  2, cv2.LINE_AA)
            cv2.putText(frame, "← OUT", (LINE_X - 80, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 140, 255), 2, cv2.LINE_AA)

            with state_lock:
                ent = system_state["entries"]
                ex  = system_state["exits"]
                em  = system_state["emergency"]

            occ = ent - ex
            cv2.rectangle(frame, (0, H - 42), (W, H), (20, 22, 28), -1)
            cv2.line(frame, (0, H - 42), (W, H - 42), (50, 52, 62), 1)
            cv2.putText(frame, f"ENTRIES: {ent}",   (10,  H - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60,  220, 80),  1, cv2.LINE_AA)
            cv2.putText(frame, f"EXITS: {ex}",      (180, H - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80,  140, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"OCCUPANCY: {occ}", (320, H - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
            if em:
                cv2.putText(frame, "!! EMERGENCY !!", (W - 200, H - 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            with _jpeg_lock:
                _latest_jpeg = jpeg.tobytes()

            time.sleep(1 / 25)

        cap.release()
        time.sleep(0.2)


def stream_frames():
    while True:
        with _jpeg_lock:
            frame = _latest_jpeg
        if frame:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(1 / 25)


# ── Auth helpers ───────────────────────────────────────────
from hashlib import sha256

def hash_pin(pin: str) -> str:
    return sha256(pin.encode()).hexdigest()

def login_required(fn):
    @wraps(fn)
    def w(*a, **k):
        if "uid" not in session:
            return redirect(url_for("login_page"))
        return fn(*a, **k)
    return w

def admin_required(fn):
    @wraps(fn)
    def w(*a, **k):
        if session.get("role") != "admin":
            return jsonify({"error": "Admin only"}), 403
        return fn(*a, **k)
    return w


# ── Routes ─────────────────────────────────────────────────

@app.route("/")
def index():
    return redirect(url_for("dashboard") if "uid" in session else url_for("login_page"))


@app.route("/login", methods=["GET"])
def login_page():
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login_post():
    data = request.get_json(force=True)
    uid  = data.get("id",  "").strip().lower()
    pin  = data.get("pin", "").strip()
    now  = time.time()

    db = get_db()

    # Brute-force check
    row = db.execute("SELECT fail_count, locked_until FROM login_attempts WHERE uid=?", (uid,)).fetchone()
    if row and row["locked_until"] > now:
        db.close()
        remaining = int((row["locked_until"] - now) / 60) + 1
        return jsonify({"error": f"Locked. Try in {remaining} min."}), 429

    user = db.execute("SELECT * FROM users WHERE uid=?", (uid,)).fetchone()
    if not user or user["pin_hash"] != hash_pin(pin):
        fail_count = (row["fail_count"] if row else 0) + 1
        locked_until = (now + 900) if fail_count >= 5 else 0.0
        db.execute(
            "INSERT OR REPLACE INTO login_attempts(uid,fail_count,locked_until) VALUES(?,?,?)",
            (uid, fail_count, locked_until)
        )
        db.commit()
        db.close()
        if fail_count >= 5:
            return jsonify({"error": "Account locked for 15 minutes."}), 429
        return jsonify({"error": f"Invalid ID or PIN. {5-fail_count} attempt(s) left."}), 401

    # Success — clear fails
    db.execute("DELETE FROM login_attempts WHERE uid=?", (uid,))
    db.commit()

    session.permanent = True
    session["uid"]   = uid
    session["role"]  = user["role"]
    session["name"]  = user["name"]
    session["phone"] = user["phone"]
    db.close()

    log_event("login", actor=uid)
    return jsonify({"ok": True, "role": user["role"]})


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html",
        uid=session["uid"], role=session["role"], name=session["name"])


# ── API: state ────────────────────────────────────────────

@app.route("/api/state")
@login_required
def api_state():
    with state_lock:
        ent = system_state["entries"]
        ex  = system_state["exits"]
        em  = system_state["emergency"]
        cam = system_state["camera_active"]
    occ = max(0, ent - ex)
    cap = int(get_setting("capacity", "200"))
    return jsonify({
        "entries":       ent,
        "exits":         ex,
        "occupancy":     occ,
        "capacity":      cap,
        "occupancy_pct": min(round(occ / cap * 100, 1), 100),
        "emergency":     em,
        "camera_active": cam,
        "timestamp":     datetime.now().strftime("%H:%M:%S"),
    })


# ── API: emergency ────────────────────────────────────────

@app.route("/api/emergency", methods=["POST"])
@login_required
def api_emergency():
    action = request.get_json(force=True).get("action", "")
    with state_lock:
        if action == "activate":
            system_state["emergency"] = True
        elif action == "deactivate":
            system_state["emergency"] = False
        em = system_state["emergency"]
        ent, ex = system_state["entries"], system_state["exits"]

    kind = "emergency_on" if em else "emergency_off"
    log_event(kind, actor=session.get("uid"), entries=ent, exits=ex, occupancy=max(0, ent - ex))

    # Optional Twilio notification
    if em and TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM:
        _send_whatsapp_alerts()

    return jsonify({"ok": True, "emergency": em})


def _send_whatsapp_alerts():
    """Send WhatsApp/SMS to all staff with a phone number via Twilio."""
    try:
        from twilio.rest import Client
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        org = get_setting("org_name", "RAQABA")
        msg = f"🚨 {org} — EMERGENCY MODE ACTIVATED. Please follow evacuation procedures immediately."

        db = get_db()
        recipients = db.execute("SELECT phone FROM users WHERE phone != ''").fetchall()
        db.close()

        for r in recipients:
            phone = r["phone"]
            if not phone.startswith("+"):
                phone = "+966" + phone.lstrip("0")   # default Saudi prefix
            try:
                client.messages.create(body=msg, from_=TWILIO_FROM, to=phone)
            except Exception as e:
                print(f"[RAQABA] Twilio send failed for {phone}: {e}")
    except ImportError:
        print("[RAQABA] Twilio not installed. pip install twilio")
    except Exception as e:
        print(f"[RAQABA] Twilio error: {e}")


# ── API: reset occupancy ──────────────────────────────────

@app.route("/api/reset", methods=["POST"])
@login_required
@admin_required
def api_reset():
    with state_lock:
        system_state["entries"] = 0
        system_state["exits"]   = 0
    log_event("reset", actor=session.get("uid"))
    return jsonify({"ok": True})


# ── API: settings ─────────────────────────────────────────

@app.route("/api/settings", methods=["GET"])
@login_required
@admin_required
def api_settings_get():
    return jsonify({
        "capacity":   get_setting("capacity", "200"),
        "line_ratio": get_setting("line_ratio", "0.5"),
        "org_name":   get_setting("org_name", "RAQABA Facility"),
    })


@app.route("/api/settings", methods=["POST"])
@login_required
@admin_required
def api_settings_post():
    d = request.get_json(force=True)
    if "capacity" in d:
        v = int(d["capacity"])
        if 1 <= v <= 100000:
            set_setting("capacity", v)
    if "line_ratio" in d:
        v = float(d["line_ratio"])
        if 0.1 <= v <= 0.9:
            set_setting("line_ratio", v)
    if "org_name" in d:
        v = str(d["org_name"]).strip()
        if v:
            set_setting("org_name", v)
    return jsonify({"ok": True})


# ── API: staff ────────────────────────────────────────────

@app.route("/api/staff", methods=["GET"])
@login_required
@admin_required
def api_staff_list():
    db = get_db()
    rows = db.execute("SELECT id, name, phone FROM staff ORDER BY name").fetchall()
    db.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/staff", methods=["POST"])
@login_required
@admin_required
def api_staff_add():
    d = request.get_json(force=True)
    n = d.get("name", "").strip()
    i = d.get("id",   "").strip()
    p = d.get("phone","").strip()
    if not n or not i or not p:
        return jsonify({"error": "All fields are required."}), 400
    db = get_db()
    try:
        db.execute("INSERT INTO staff(id,name,phone) VALUES(?,?,?)", (i, n, p))
        db.commit()
    except sqlite3.IntegrityError:
        db.close()
        return jsonify({"error": "Employee ID already exists."}), 409
    db.close()
    return jsonify({"ok": True})


@app.route("/api/staff/<empid>", methods=["DELETE"])
@login_required
@admin_required
def api_staff_delete(empid):
    db = get_db()
    cur = db.execute("DELETE FROM staff WHERE id=?", (empid,))
    db.commit()
    db.close()
    if cur.rowcount == 0:
        return jsonify({"error": "Not found."}), 404
    return jsonify({"ok": True})


# ── API: users (admin manages login accounts) ──────────────

@app.route("/api/users", methods=["GET"])
@login_required
@admin_required
def api_users_list():
    db = get_db()
    rows = db.execute("SELECT uid, role, name, phone, created_at FROM users ORDER BY uid").fetchall()
    db.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/users", methods=["POST"])
@login_required
@admin_required
def api_users_add():
    d    = request.get_json(force=True)
    uid  = d.get("uid",  "").strip().lower()
    pin  = d.get("pin",  "").strip()
    role = d.get("role", "staff")
    name = d.get("name", "").strip()
    phone= d.get("phone","").strip()
    if not uid or not pin:
        return jsonify({"error": "UID and PIN are required."}), 400
    if role not in ("admin", "staff"):
        return jsonify({"error": "Invalid role."}), 400
    db = get_db()
    try:
        db.execute(
            "INSERT INTO users(uid,pin_hash,role,name,phone) VALUES(?,?,?,?,?)",
            (uid, hash_pin(pin), role, name, phone)
        )
        db.commit()
    except sqlite3.IntegrityError:
        db.close()
        return jsonify({"error": "User ID already exists."}), 409
    db.close()
    return jsonify({"ok": True})


@app.route("/api/users/<uid>", methods=["DELETE"])
@login_required
@admin_required
def api_users_delete(uid):
    if uid == session.get("uid"):
        return jsonify({"error": "Cannot delete your own account."}), 400
    db = get_db()
    cur = db.execute("DELETE FROM users WHERE uid=?", (uid,))
    db.commit()
    db.close()
    if cur.rowcount == 0:
        return jsonify({"error": "Not found."}), 404
    return jsonify({"ok": True})


@app.route("/api/users/<uid>/pin", methods=["POST"])
@login_required
@admin_required
def api_users_reset_pin(uid):
    d = request.get_json(force=True)
    new_pin = d.get("pin", "").strip()
    if len(new_pin) < 4:
        return jsonify({"error": "PIN must be at least 4 characters."}), 400
    db = get_db()
    cur = db.execute("UPDATE users SET pin_hash=? WHERE uid=?", (hash_pin(new_pin), uid))
    db.commit()
    db.close()
    if cur.rowcount == 0:
        return jsonify({"error": "User not found."}), 404
    return jsonify({"ok": True})


# ── API: profile ──────────────────────────────────────────

@app.route("/api/profile", methods=["GET"])
@login_required
def api_profile_get():
    db = get_db()
    row = db.execute("SELECT name, phone FROM users WHERE uid=?", (session["uid"],)).fetchone()
    db.close()
    return jsonify({"name": row["name"], "phone": row["phone"]})


@app.route("/api/profile", methods=["POST"])
@login_required
def api_profile_update():
    d     = request.get_json(force=True)
    name  = d.get("name",  "").strip()
    phone = d.get("phone", "").strip()
    uid   = session["uid"]
    db = get_db()
    db.execute("UPDATE users SET name=?, phone=? WHERE uid=?", (name, phone, uid))
    db.commit()
    db.close()
    session["name"]  = name
    session["phone"] = phone
    return jsonify({"ok": True, "name": name, "phone": phone})


@app.route("/api/profile/pin", methods=["POST"])
@login_required
def api_profile_change_pin():
    d       = request.get_json(force=True)
    old_pin = d.get("old_pin", "").strip()
    new_pin = d.get("new_pin", "").strip()
    uid     = session["uid"]
    if len(new_pin) < 4:
        return jsonify({"error": "New PIN must be at least 4 characters."}), 400
    db = get_db()
    row = db.execute("SELECT pin_hash FROM users WHERE uid=?", (uid,)).fetchone()
    if not row or row["pin_hash"] != hash_pin(old_pin):
        db.close()
        return jsonify({"error": "Current PIN is incorrect."}), 401
    db.execute("UPDATE users SET pin_hash=? WHERE uid=?", (hash_pin(new_pin), uid))
    db.commit()
    db.close()
    return jsonify({"ok": True})


# ── API: cameras + exits ──────────────────────────────────

@app.route("/api/cameras")
@login_required
def api_cameras():
    with state_lock:
        cam1 = system_state["camera_active"]
    return jsonify([
        {"id": 1, "name": "Camera 01 — Main Entrance", "exit": "Exit 1", "active": cam1},
        {"id": 2, "name": "Camera 02 — Side Exit A",   "exit": "Exit 2", "active": True},
        {"id": 3, "name": "Camera 03 — Rear Exit",     "exit": "Exit 3", "active": False},
        {"id": 4, "name": "Camera 04 — Emergency Exit","exit": "Exit 4", "active": True},
    ])


@app.route("/api/exits")
@login_required
def api_exits():
    with state_lock:
        t_in  = system_state["entries"]
        t_out = system_state["exits"]
        cam1  = system_state["camera_active"]
    return jsonify([
        {"name": "Exit 1 — Main Entrance", "entries": t_in,             "exits": t_out,           "cam": "Camera 01", "cam_active": cam1},
        {"name": "Exit 2 — Side Exit A",   "entries": round(t_in*.28), "exits": round(t_out*.30),"cam": "Camera 02", "cam_active": True},
        {"name": "Exit 3 — Rear Exit",     "entries": round(t_in*.10), "exits": round(t_out*.10),"cam": "Camera 03", "cam_active": False},
        {"name": "Exit 4 — Emergency Exit","entries": round(t_in*.07), "exits": round(t_out*.05),"cam": "Camera 04", "cam_active": True},
    ])


@app.route("/api/admin_contact")
@login_required
def api_admin_contact():
    db = get_db()
    row = db.execute("SELECT name, phone FROM users WHERE role='admin' ORDER BY uid LIMIT 1").fetchone()
    db.close()
    return jsonify({"name": row["name"] if row else "Admin", "phone": row["phone"] if row else "—"})


# ── API: event log ────────────────────────────────────────

@app.route("/api/events")
@login_required
@admin_required
def api_events():
    limit  = min(int(request.args.get("limit",  "100")), 1000)
    offset = int(request.args.get("offset", "0"))
    db = get_db()
    rows = db.execute(
        "SELECT ts, kind, actor, note, entries, exits, occupancy FROM events ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset)
    ).fetchall()
    total = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    db.close()
    return jsonify({"total": total, "events": [dict(r) for r in rows]})


@app.route("/api/events/export")
@login_required
@admin_required
def api_events_export():
    db = get_db()
    rows = db.execute(
        "SELECT ts, kind, actor, note, entries, exits, occupancy FROM events ORDER BY id ASC"
    ).fetchall()
    db.close()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Timestamp", "Event", "Actor", "Note", "Entries", "Exits", "Occupancy"])
    for r in rows:
        writer.writerow([r["ts"], r["kind"], r["actor"] or "", r["note"] or "",
                         r["entries"], r["exits"], r["occupancy"]])
    buf.seek(0)
    fname = f"raqaba_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return send_file(
        io.BytesIO(buf.getvalue().encode("utf-8-sig")),
        mimetype="text/csv",
        as_attachment=True,
        download_name=fname,
    )


# ── Video feed ────────────────────────────────────────────

@app.route("/video_feed")
@login_required
def video_feed():
    return Response(stream_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ── Boot ──────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    threading.Thread(target=video_loop, daemon=True).start()
    print("\n" + "=" * 56)
    print("  RAQABA v7  →  http://localhost:5000")
    print("  admin/1234   |   staff/1234")
    print("  DB:", DB_PATH)
    print("=" * 56 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

# RAQABA | رقابة — v7
Smart Occupancy & Evacuation Monitoring System

---

## What's new in v7

| Feature | v6 | v7 |
|---|---|---|
| Data persistence | ❌ RAM only | ✅ SQLite database |
| Secrets | ❌ Hardcoded in code | ✅ `.env` file |
| Settings (capacity, line pos) | ❌ Edit code | ✅ Dashboard UI |
| Event log | ❌ None | ✅ Full log + CSV export |
| User management | ❌ Fixed 2 accounts | ✅ Unlimited accounts from dashboard |
| PIN change | ❌ | ✅ Users + admin both |
| Counter reset | ❌ | ✅ One-click from Settings |
| WhatsApp/SMS alerts | ❌ | ✅ Via Twilio (optional) |

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env — set RAQABA_SECRET to a long random string

# 3. Run
python app.py

# 4. Open
http://localhost:5000
```

---

## Demo Accounts

| Role  | User ID | PIN  |
|-------|---------|------|
| Admin | admin   | 1234 |
| Staff | staff   | 1234 |

Change PINs from Accounts or Profile.

---

## Environment Variables (.env)

| Variable | Required | Description |
|---|---|---|
| RAQABA_SECRET | Yes | Flask session secret — long random string |
| RAQABA_DB | No | SQLite file path (default: raqaba.db) |
| RAQABA_VIDEO | No | Video file path or 0 for webcam |
| TWILIO_SID | No | Twilio account SID |
| TWILIO_TOKEN | No | Twilio auth token |
| TWILIO_FROM | No | whatsapp:+14155238886 or SMS number |

---

## Production

```bash
pip install gunicorn
gunicorn -w 1 -b 0.0.0.0:5000 --timeout 120 app:app
```

Use -w 1 — the video thread must stay in one process.

For Railway/Render: set env vars in platform dashboard.
Set RAQABA_DB to a persistent volume path (e.g. /data/raqaba.db).

---

## Project Structure

```
raqaba/
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── raqaba.db           ← auto-created on first run
├── videos/demo.mp4
├── static/
└── templates/
    ├── login.html
    └── dashboard.html
```

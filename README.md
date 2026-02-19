# ☕ Real-Time Cafe Occupancy Monitoring System

AI-powered person detection and counting using **YOLOv8m** + **DeepSORT** tracking, served via a **Flask** backend with a modern glassmorphism dashboard.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
cd c:\Users\lenovo\Downloads\yolov8_cafe
pip install -r requirements.txt
```

> **Note:** `yolov8m.pt` (~50 MB) is auto-downloaded by Ultralytics on first run. Requires internet access.

### 2. Run the server
```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## 🎥 Video Sources

| Source | How to use |
|--------|-----------|
| **Webcam** | Default (index 0). Click **Webcam** button in UI |
| **Upload video file** | Click **⬆ Upload Video** — supports MP4, AVI, MOV, MKV, WEBM |
| **RTSP stream** | Enter `rtsp://user:pass@ip:port/stream` in Settings → Switch |
| **Custom webcam index** | Enter `1`, `2`, etc. in Settings → Switch |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/video_feed` | MJPEG live stream |
| `GET` | `/api/occupancy` | Current count, %, status |
| `GET` | `/api/history?limit=N` | Last N records (max 100) |
| `POST` | `/api/settings` | `{"max_capacity": 50}` |
| `POST` | `/api/source` | `{"source": "0"}` or RTSP URL |
| `POST` | `/api/upload` | Upload video file (multipart) |
| `GET` | `/api/status` | Health check |

### Example occupancy response
```json
{
  "current_count": 12,
  "max_capacity": 50,
  "occupancy_percent": 24,
  "status": "SAFE"
}
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_MODEL` | `yolov8m.pt` | Model path (use `yolov8l.pt` for higher accuracy) |
| `VIDEO_SOURCE` | `0` | Default video source |
| `CONFIDENCE` | `0.5` | Detection confidence threshold |
| `MAX_CAPACITY` | `50` | Initial max capacity |

Example:
```bash
set VIDEO_SOURCE=cafe_recording.mp4
set YOLO_MODEL=yolov8l.pt
python app.py
```

---

## 📁 Project Structure

```
yolov8_cafe/
├── app.py              # Flask server + API endpoints
├── detector.py         # YOLOv8 + DeepSORT detection engine
├── requirements.txt    # Python dependencies
├── uploads/            # Uploaded video files (auto-created)
├── templates/
│   └── index.html      # Dashboard UI
└── static/
    ├── css/style.css   # Glassmorphism dark theme
    └── js/main.js      # Real-time updates + Chart.js
```

---

## 🎨 Dashboard Features

- 📹 **Live video stream** with bounding boxes and track IDs
- 👥 **KPI cards** — count, capacity, %, status (animated)
- 🎯 **Circular occupancy meter** — color changes at 60% / 85%
- 📈 **Real-time trend chart** — last 20 data points, updates every 5s
- ⬆️ **Video upload** — drag & drop any video file with progress bar
- ⚙️ **Settings panel** — change capacity or switch source live

---

## 📊 Status Thresholds

| Occupancy | Status | Color |
|-----------|--------|-------|
| 0 – 59% | SAFE | 🟢 Green |
| 60 – 84% | WARNING | 🟠 Orange |
| 85 – 100% | FULL | 🔴 Red |

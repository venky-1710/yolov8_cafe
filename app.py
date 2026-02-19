import os
import time
import threading
import json
from datetime import datetime
from pathlib import Path

from flask import (Flask, Response, render_template,
                   request, jsonify, stream_with_context)
from flask_cors import CORS
from werkzeug.utils import secure_filename

from detector import CafeDetector

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER   = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTS    = {"mp4", "avi", "mov", "mkv", "webm", "flv"}
MAX_CONTENT_MB  = 500
app.config["UPLOAD_FOLDER"]    = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

# ─────────────────────────────────────────────
# Global detector (lazy-init on first request)
# ─────────────────────────────────────────────
detector: CafeDetector | None = None
detector_lock = threading.Lock()


def get_detector() -> CafeDetector:
    global detector
    if detector is None:
        with detector_lock:
            if detector is None:
                detector = CafeDetector()
    return detector


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


# ─────────────────────────────────────────────
# MJPEG generator
# ─────────────────────────────────────────────
def generate_frames():
    det = get_detector()
    while True:
        frame_bytes = det.get_jpeg_frame()
        if frame_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes +
                b"\r\n"
            )
        time.sleep(0.033)   # ~30 FPS cap



# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/team")
def team():
    return render_template("team.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG stream endpoint."""
    return Response(
        stream_with_context(generate_frames()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/video_snapshot")
def video_snapshot():
    """Return a single JPEG frame — polled by JS for reliable cross-browser display."""
    frame_bytes = get_detector().get_jpeg_frame()
    if not frame_bytes:
        # Return a 1x1 transparent pixel if no frame yet
        import base64
        pixel = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        frame_bytes = pixel
    resp = Response(frame_bytes, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"]        = "no-cache"
    resp.headers["Expires"]       = "0"
    return resp


@app.route("/api/occupancy")
def api_occupancy():
    """Return current occupancy JSON."""
    data = get_detector().get_occupancy_data()
    return jsonify(data)


@app.route("/api/history")
def api_history():
    """Return last N occupancy records."""
    limit = min(int(request.args.get("limit", 100)), 100)
    history = get_detector().get_history()
    return jsonify(history[-limit:])


@app.route("/api/settings", methods=["POST"])
def api_settings():
    """Update max capacity."""
    body = request.get_json(silent=True) or {}
    cap  = body.get("max_capacity")
    if cap is None or not isinstance(cap, (int, float)) or int(cap) < 1:
        return jsonify({"error": "Invalid max_capacity"}), 400
    get_detector().update_capacity(int(cap))
    return jsonify({"success": True, "max_capacity": int(cap)})


@app.route("/api/source", methods=["POST"])
def api_source():
    """Switch to webcam or RTSP URL."""
    body   = request.get_json(silent=True) or {}
    source = body.get("source", "0")
    try:
        get_detector().switch_source(str(source))
        return jsonify({"success": True, "source": source})
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """
    Upload a video file and switch the detector to use it.
    Accepts multipart/form-data with field name 'video'.
    """
    if "video" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"File type not allowed. Supported: {', '.join(ALLOWED_EXTS)}"
        }), 400

    filename = secure_filename(file.filename)
    save_path = UPLOAD_FOLDER / filename
    file.save(str(save_path))

    try:
        get_detector().switch_source(str(save_path))
        return jsonify({
            "success":  True,
            "filename": filename,
            "message":  f"Now processing uploaded video: {filename}",
        })
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model")
def api_model():
    """Return loaded model information."""
    det = get_detector()
    info = getattr(det, "model_info", {})
    return jsonify(info)


@app.route("/api/status")
def api_status():
    """Health-check endpoint."""
    det = get_detector()
    return jsonify({
        "running":      True,
        "max_capacity": det.max_capacity,
        "timestamp":    datetime.now().isoformat(),
    })


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Cafe Occupancy Monitor  |  http://127.0.0.1:5000")
    print("=" * 55)
    # Pre-init detector so model loads before first request
    get_detector()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

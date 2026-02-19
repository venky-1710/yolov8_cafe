"""
detector.py — 3-thread architecture for zero-lag webcam display

  Thread 1 (grab_loop):    reads frames from webcam as fast as possible,
                            always discards stale buffered frames, keeps only the LATEST.
  Thread 2 (detect_loop):  takes the latest raw frame, runs YOLO, stores annotated result.
  Thread 3 (encode_loop):  JPEG-encodes the latest annotated frame for the web snapshot.

This means the webcam display is NEVER blocked by slow YOLO inference.
"""

import cv2
import numpy as np
import threading
import time
import os
from collections import deque
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
def _resolve_model(env_val: str) -> str:
    candidate = Path("models") / env_val
    if candidate.exists():
        return str(candidate)
    trained = sorted(Path("runs/train").glob("*/weights/best.pt")) if Path("runs/train").exists() else []
    if trained:
        return str(trained[-1])
    return env_val

MODEL_PATH    = _resolve_model(os.environ.get("YOLO_MODEL", "yolov8n.pt"))
VIDEO_SOURCE  = os.environ.get("VIDEO_SOURCE", "0")
CONFIDENCE    = float(os.environ.get("CONFIDENCE", "0.4"))
MAX_CAPACITY  = int(os.environ.get("MAX_CAPACITY", "50"))
FRAME_W       = 640
FRAME_H       = 480
INFER_SIZE    = 320          # smaller ⇒ faster CPU inference
HISTORY_LIMIT = 100
PERSON_CLASS  = 0
JPEG_QUALITY  = 75           # lower ⇒ faster encoding & transfer


def _parse_source(src: str):
    try:
        return int(src)
    except ValueError:
        return src


def _open_capture(src) -> cv2.VideoCapture:
    if isinstance(src, int):
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(src)
    else:
        cap = cv2.VideoCapture(src)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # minimal OS buffer = less stale frames
        cap.set(cv2.CAP_PROP_FPS,          30)
    return cap


def _make_placeholder(msg: str = "Connecting…") -> np.ndarray:
    frame = np.full((FRAME_H, FRAME_W, 3), (20, 15, 30), dtype=np.uint8)
    font  = cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(msg, font, 0.65, 2)
    cv2.putText(frame, msg,
                ((FRAME_W - tw) // 2, FRAME_H // 2),
                font, 0.65, (160, 120, 255), 2, cv2.LINE_AA)
    return frame


class CafeDetector:
    """
    Zero-lag detector using 3 independent threads:
    - grab     : webcam read loop (fastest possible)
    - detect   : YOLO inference loop (slow, runs separately)
    - encode   : JPEG encode loop (feeds web snapshot endpoint)
    """

    def __init__(self, video_source=None, max_capacity=MAX_CAPACITY):
        self.max_capacity = max_capacity
        self._stop        = threading.Event()

        # ── Shared state (each protected by its own lock) ──────────────
        self._raw_lock    = threading.Lock()
        self._raw_frame   = None          # latest frame from camera (unannotated)

        self._ann_lock    = threading.Lock()
        self._ann_frame   = _make_placeholder("Loading model…")  # latest annotated

        self._jpeg_lock   = threading.Lock()
        self._jpeg_bytes  = None          # latest JPEG-encoded frame

        self._data_lock   = threading.Lock()
        self._count       = 0
        self._history     = deque(maxlen=HISTORY_LIMIT)

        # ── Load model ─────────────────────────────────────────────────
        print(f"[Detector] Loading model: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)
        dummy = np.zeros((INFER_SIZE, INFER_SIZE, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        print("[Detector] Model ready.")

        self.model_info = {
            "path": MODEL_PATH, "name": Path(MODEL_PATH).stem,
            "infer_size": INFER_SIZE, "confidence": CONFIDENCE,
        }

        # ── Open webcam ────────────────────────────────────────────────
        src = _parse_source(video_source or VIDEO_SOURCE)
        self._source_str    = str(video_source or VIDEO_SOURCE)
        self._is_video_file = isinstance(src, str) and not str(src).startswith("rtsp")

        self.cap = _open_capture(src)
        if self.cap.isOpened():
            print(f"[Detector] Camera opened: {src}")
        else:
            print(f"[Detector] WARNING: Cannot open source '{src}'")
            with self._ann_lock:
                self._ann_frame = _make_placeholder("Camera not found")

        # ── Start threads ──────────────────────────────────────────────
        self._t_grab   = threading.Thread(target=self._grab_loop,   daemon=True, name="grab")
        self._t_detect = threading.Thread(target=self._detect_loop, daemon=True, name="detect")
        self._t_encode = threading.Thread(target=self._encode_loop, daemon=True, name="encode")
        self._t_grab.start()
        self._t_detect.start()
        self._t_encode.start()
        print("[Detector] All threads started.")

    # ─────────────────────────────────────────────────────────────────
    # Thread 1: GRAB — reads webcam as fast as possible
    # ─────────────────────────────────────────────────────────────────
    def _grab_loop(self):
        fail = 0
        while not self._stop.is_set():
            if not self.cap.isOpened():
                time.sleep(0.5)
                continue

            ret, frame = self.cap.read()

            if not ret or frame is None or frame.size == 0:
                fail += 1
                if self._is_video_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    fail = 0
                elif fail > 30:
                    print("[Grab] Camera stalled — reconnecting…")
                    src = _parse_source(self._source_str)
                    self.cap.release()
                    time.sleep(1)
                    self.cap = _open_capture(src)
                    fail = 0
                    with self._ann_lock:
                        self._ann_frame = _make_placeholder("Reconnecting…")
                continue

            fail = 0
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))

            # Always keep ONLY the latest raw frame (overwrite without waiting)
            with self._raw_lock:
                self._raw_frame = frame

            # No sleep — grab as fast as the camera allows

    # ─────────────────────────────────────────────────────────────────
    # Thread 2: DETECT — runs YOLO on latest raw frame
    # ─────────────────────────────────────────────────────────────────
    def _detect_loop(self):
        last_raw = None
        while not self._stop.is_set():
            # Grab latest raw frame
            with self._raw_lock:
                raw = self._raw_frame

            if raw is None or (last_raw is not None and raw is last_raw):
                # No new frame yet — wait a bit
                time.sleep(0.01)
                continue

            last_raw = raw
            frame    = raw.copy()

            try:
                results = self.model(
                    frame,
                    classes=[PERSON_CLASS],
                    conf=CONFIDENCE,
                    iou=0.4,
                    imgsz=INFER_SIZE,
                    verbose=False,
                    half=False,
                )[0]

                count = 0
                pct   = 0
                for box in results.boxes:
                    count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    pct    = min(100, int(count / max(1, self.max_capacity) * 100))
                    color  = self._status_color(pct)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    conf_v = float(box.conf[0])
                    cv2.putText(frame, f"{conf_v:.2f}",
                                (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            except Exception as e:
                print(f"[Detect] Error: {e}")
                count = 0

            # Draw HUD overlay
            frame = self._draw_hud(frame, count)
            pct   = min(100, int(count / max(1, self.max_capacity) * 100))

            with self._ann_lock:
                self._ann_frame = frame

            with self._data_lock:
                self._count = count
                self._history.append({
                    "timestamp":         datetime.now().isoformat(),
                    "count":             count,
                    "occupancy_percent": pct,
                    "status":            self._status_label(pct),
                })

    # ─────────────────────────────────────────────────────────────────
    # Thread 3: ENCODE — JPEG encodes latest annotated frame
    # ─────────────────────────────────────────────────────────────────
    def _encode_loop(self):
        last_ann = None
        while not self._stop.is_set():
            with self._ann_lock:
                ann = self._ann_frame

            if ann is last_ann:
                time.sleep(0.01)
                continue
            last_ann = ann

            try:
                _, buf = cv2.imencode(
                    ".jpg", ann,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
                with self._jpeg_lock:
                    self._jpeg_bytes = buf.tobytes()
            except Exception as e:
                print(f"[Encode] Error: {e}")

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _status_label(pct: int) -> str:
        if pct < 60:  return "SAFE"
        if pct < 85:  return "WARNING"
        return "FULL"

    @staticmethod
    def _status_color(pct: int):
        if pct < 60:  return (0, 220, 80)
        if pct < 85:  return (0, 165, 255)
        return (0, 50, 220)

    def _draw_hud(self, frame: np.ndarray, count: int) -> np.ndarray:
        h, w   = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 44), (8, 8, 24), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        pct    = min(100, int(count / max(1, self.max_capacity) * 100))
        color  = self._status_color(pct)
        status = self._status_label(pct)
        label  = f"People: {count}  |  {pct}%  |  {status}"

        cv2.putText(frame, label,
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        cv2.putText(frame, datetime.now().strftime("%H:%M:%S"),
                    (w - 90, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
        return frame

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────
    def get_jpeg_frame(self) -> bytes | None:
        with self._jpeg_lock:
            b = self._jpeg_bytes
        if b is None:
            # Encode placeholder synchronously if encode thread hasn't run yet
            with self._ann_lock:
                ann = self._ann_frame
            _, buf = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            return buf.tobytes()
        return b

    def get_occupancy_data(self) -> dict:
        with self._data_lock:
            count = self._count
        pct = min(100, int(count / max(1, self.max_capacity) * 100))
        return {
            "current_count":     count,
            "max_capacity":      self.max_capacity,
            "occupancy_percent": pct,
            "status":            self._status_label(pct),
        }

    def get_history(self) -> list:
        with self._data_lock:
            return list(self._history)

    def update_capacity(self, new_capacity: int):
        with self._data_lock:
            self.max_capacity = max(1, new_capacity)

    def switch_source(self, new_source: str):
        src     = _parse_source(new_source)
        new_cap = _open_capture(src)
        if not new_cap.isOpened():
            raise RuntimeError(f"Cannot open: {new_source}")
        with self._raw_lock:
            self.cap.release()
            self.cap            = new_cap
            self._source_str    = new_source
            self._is_video_file = isinstance(src, str) and not str(src).startswith("rtsp")
            self._raw_frame     = None
        with self._ann_lock:
            self._ann_frame = _make_placeholder("Switching source…")
        with self._data_lock:
            self._count = 0
        print(f"[Detector] Switched to: {new_source}")

    def stop(self):
        self._stop.set()
        self.cap.release()
        print("[Detector] Stopped.")

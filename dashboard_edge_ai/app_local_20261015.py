#!/usr/bin/env python3
"""Dashboard EdgeVision: deteccion de objetos y clasificacion de senas."""

from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import psutil
from flask import Flask, Response, jsonify, render_template
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_OBJECT_MODEL = BASE_DIR.parent / "yolo26n_ncnn_model"
DEFAULT_SIGN_MODEL = (
    BASE_DIR
    / "runs/classify/runs/clasificacion/senas_v1/weights/best_ncnn_model"
)

OBJECT_MODEL_PATH = Path(
    os.environ.get("OBJECT_MODEL_PATH", str(DEFAULT_OBJECT_MODEL))
)
SIGN_MODEL_PATH = Path(os.environ.get("SIGN_MODEL_PATH", str(DEFAULT_SIGN_MODEL)))

CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
OBJECT_CONFIDENCE = float(os.environ.get("CONFIDENCE", "0.40"))
SIGN_CONFIDENCE = float(os.environ.get("SIGN_CONFIDENCE", "0.65"))
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "320"))
SIGN_IMAGE_SIZE = int(os.environ.get("SIGN_IMAGE_SIZE", "224"))
SIGN_INTERVAL = max(1, int(os.environ.get("SIGN_INTERVAL", "2")))
SIGN_STABILITY = max(1, int(os.environ.get("SIGN_STABILITY", "3")))

TRANSLATIONS = {
    "person": "Persona",
    "bottle": "Botella",
    "cup": "Taza",
    "chair": "Silla",
    "laptop": "Computadora portatil",
    "cell phone": "Telefono",
    "keyboard": "Teclado",
    "tv": "Pantalla",
    "dining table": "Mesa",
    "book": "Libro",
    "backpack": "Mochila",
    "mouse": "Raton",
}

SIGN_LABELS = {
    "agua": "AGUA",
    "ayuda": "AYUDA",
    "bano": "BAÑO",
    "emergencia": "EMERGENCIA",
    "no": "NO",
    "si": "SI",
}


class VisionEngine:
    """Captura una camara y ejecuta ambos modelos en un solo hilo."""

    def __init__(self) -> None:
        if not OBJECT_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No se encontro el modelo de objetos: {OBJECT_MODEL_PATH}"
            )
        if not SIGN_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No se encontro el modelo de senas: {SIGN_MODEL_PATH}"
            )

        self.object_model = YOLO(str(OBJECT_MODEL_PATH), task="detect")
        self.sign_model = YOLO(str(SIGN_MODEL_PATH), task="classify")

        self.camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)

        if not self.camera.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la camara /dev/video{CAMERA_INDEX}"
            )

        self.lock = threading.Lock()
        self.latest_jpeg: bytes | None = None
        self.detected: dict[str, int] = {}
        self.sign = "Sin sena"
        self.sign_confidence = 0.0
        self.object_latency_ms = 0.0
        self.sign_latency_ms = 0.0
        self.latency_ms = 0.0
        self.fps = 0.0
        self.frames = 0
        self.running = True
        self.started_at = time.time()
        self.sign_history: deque[str] = deque(maxlen=SIGN_STABILITY)
        self.last_sign = "Sin sena"
        self.last_sign_confidence = 0.0

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _classify_sign(self, frame) -> tuple[str, float, float]:
        started = time.perf_counter()
        results = self.sign_model.predict(
            source=frame,
            imgsz=SIGN_IMAGE_SIZE,
            verbose=False,
        )
        latency = (time.perf_counter() - started) * 1000
        result = results[0]

        if result.probs is None:
            candidate = "Sin sena"
            confidence = 0.0
        else:
            class_id = int(result.probs.top1)
            confidence = float(result.probs.top1conf)
            raw_name = str(result.names[class_id]).lower()
            candidate = SIGN_LABELS.get(raw_name, raw_name.upper())
            if confidence < SIGN_CONFIDENCE:
                candidate = "Sin sena"

        self.sign_history.append(candidate)
        stable_sign, occurrences = Counter(self.sign_history).most_common(1)[0]

        if occurrences >= SIGN_STABILITY:
            self.last_sign = stable_sign
            self.last_sign_confidence = confidence if stable_sign != "Sin sena" else 0.0

        return self.last_sign, self.last_sign_confidence, latency

    def _run(self) -> None:
        previous_time = time.perf_counter()
        smoothed_fps = 0.0
        frame_number = 0

        while self.running:
            ok, frame = self.camera.read()
            if not ok:
                time.sleep(0.1)
                continue

            frame_number += 1
            total_started = time.perf_counter()

            object_started = time.perf_counter()
            object_results = self.object_model.predict(
                source=frame,
                imgsz=IMAGE_SIZE,
                conf=OBJECT_CONFIDENCE,
                verbose=False,
            )
            object_latency_ms = (time.perf_counter() - object_started) * 1000

            object_result = object_results[0]
            translated_names = {
                key: TRANSLATIONS.get(value, value.capitalize())
                for key, value in object_result.names.items()
            }
            object_result.names = translated_names
            annotated = object_result.plot().copy()
            counts: Counter[str] = Counter()

            if object_result.boxes is not None:
                for class_id in object_result.boxes.cls.tolist():
                    label = translated_names[int(class_id)]
                    counts[label] += 1

            sign = self.last_sign
            sign_confidence = self.last_sign_confidence
            sign_latency_ms = self.sign_latency_ms

            if frame_number % SIGN_INTERVAL == 0:
                sign, sign_confidence, sign_latency_ms = self._classify_sign(frame)

            total_latency_ms = (time.perf_counter() - total_started) * 1000

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - previous_time, 0.001)
            previous_time = now
            smoothed_fps = instant_fps if smoothed_fps == 0 else (
                0.85 * smoothed_fps + 0.15 * instant_fps
            )

            cv2.rectangle(annotated, (0, 0), (640, 72), (20, 20, 20), -1)
            cv2.putText(
                annotated,
                f"SENA: {sign} ({sign_confidence * 100:.0f}%)",
                (12, 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"FPS: {smoothed_fps:.1f} | Latencia total: {total_latency_ms:.0f} ms",
                (12, 59),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            encoded, buffer = cv2.imencode(
                ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82]
            )
            if not encoded:
                continue

            with self.lock:
                self.latest_jpeg = buffer.tobytes()
                self.detected = dict(counts)
                self.sign = sign
                self.sign_confidence = sign_confidence
                self.object_latency_ms = object_latency_ms
                self.sign_latency_ms = sign_latency_ms
                self.latency_ms = total_latency_ms
                self.fps = smoothed_fps
                self.frames += 1

    def frame_stream(self):
        while self.running:
            with self.lock:
                frame = self.latest_jpeg
            if frame is None:
                time.sleep(0.05)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.02)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "detected": self.detected.copy(),
                "sign": self.sign,
                "sign_confidence": round(self.sign_confidence, 3),
                "latency_ms": round(self.latency_ms, 1),
                "object_latency_ms": round(self.object_latency_ms, 1),
                "sign_latency_ms": round(self.sign_latency_ms, 1),
                "fps": round(self.fps, 2),
                "frames": self.frames,
            }


def temperature_celsius() -> float | None:
    thermal_file = Path("/sys/class/thermal/thermal_zone0/temp")
    try:
        return round(int(thermal_file.read_text().strip()) / 1000, 1)
    except (OSError, ValueError):
        return None


def local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("1.1.1.1", 80))
        return sock.getsockname()[0]
    except OSError:
        return "Sin red"
    finally:
        sock.close()


def worker_status() -> list[dict]:
    """Lee WORKER_NODES='nombre=ip,nombre=ip' y verifica conectividad local."""
    raw = os.environ.get("WORKER_NODES", "").strip()
    if not raw:
        return [{"name": "edge-worker01", "ip": "Pendiente", "online": False}]

    workers = []
    for item in raw.split(","):
        if "=" not in item:
            continue
        name, ip = (part.strip() for part in item.split("=", 1))
        result = subprocess.run(
            ["ping", "-c", "1", "-W", "1", ip],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        workers.append({"name": name, "ip": ip, "online": result.returncode == 0})
    return workers


app = Flask(__name__)
engine = VisionEngine()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/video_feed")
def video_feed():
    return Response(
        engine.frame_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/status")
def status():
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    vision = engine.snapshot()
    return jsonify(
        {
            "node": {
                "name": socket.gethostname(),
                "ip": local_ip(),
                "online": True,
            },
            "system": {
                "cpu": round(psutil.cpu_percent(interval=None), 1),
                "memory": round(memory.percent, 1),
                "disk": round(disk.percent, 1),
                "temperature": temperature_celsius(),
                "uptime_seconds": int(time.time() - psutil.boot_time()),
            },
            "vision": vision,
            "workers": worker_status(),
            "mode": "objects_and_signs",
        }
    )


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "node": socket.gethostname(),
            "object_model": OBJECT_MODEL_PATH.name,
            "sign_model": SIGN_MODEL_PATH.name,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

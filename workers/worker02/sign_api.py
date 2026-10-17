#!/usr/bin/env python3
import socket
import threading
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request
from ultralytics import YOLO

MODEL_PATH = "/home/soporte/edge-inference/models/best_ncnn_model"
MODEL = YOLO(MODEL_PATH, task="classify")
LOCK = threading.Lock()

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify(
        status="ok",
        node=socket.gethostname(),
        service="sign-classification"
    )

@app.post("/infer")
def infer():
    if "image" not in request.files:
        return jsonify(error="Falta el archivo image"), 400

    data = np.frombuffer(request.files["image"].read(), dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify(error="Imagen invalida"), 400

    started = time.perf_counter()

    with LOCK:
        result = MODEL.predict(
            source=frame,
            imgsz=224,
            verbose=False
        )[0]

    if result.probs is None:
        return jsonify(error="El modelo no devolvio clasificacion"), 500

    class_id = int(result.probs.top1)
    confidence = float(result.probs.top1conf)
    label = str(result.names[class_id]).lower()
    latency = (time.perf_counter() - started) * 1000

    return jsonify(
        node=socket.gethostname(),
        class_id=class_id,
        label=label,
        confidence=round(confidence, 4),
        latency_ms=round(latency, 2)
    )

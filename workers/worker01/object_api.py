#!/usr/bin/env python3
import socket
import threading
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request
from ultralytics import YOLO

MODEL_PATH = "/home/soporte/edge-inference/models/yolo26n_ncnn_model"
MODEL = YOLO(MODEL_PATH, task="detect")
LOCK = threading.Lock()

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify(
        status="ok",
        node=socket.gethostname(),
        service="object-detection"
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
            imgsz=320,
            conf=0.40,
            verbose=False
        )[0]

    detections = []

    if result.boxes is not None:
        for box, class_id, confidence in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.cls.tolist(),
            result.boxes.conf.tolist()
        ):
            class_id = int(class_id)
            detections.append({
                "class_id": class_id,
                "label": str(result.names[class_id]),
                "confidence": round(float(confidence), 4),
                "box": [round(float(value), 2) for value in box]
            })

    latency = (time.perf_counter() - started) * 1000

    return jsonify(
        node=socket.gethostname(),
        detections=detections,
        latency_ms=round(latency, 2)
    )

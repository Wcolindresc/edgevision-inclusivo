#!/usr/bin/env python3
"""Captura muestras uniformes para el clasificador de señas estáticas."""

from __future__ import annotations

import argparse
import re
import time
from datetime import datetime
from pathlib import Path

import cv2


BASE_DIR = Path(__file__).resolve().parent
VALID_SIGNS = ("ayuda", "agua", "bano", "si", "no", "emergencia")
DISPLAY_NAMES = {
    "ayuda": "AYUDA",
    "agua": "AGUA",
    "bano": "BANO",
    "si": "SI",
    "no": "NO",
    "emergencia": "EMERGENCIA",
}


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower())
    if not cleaned:
        raise argparse.ArgumentTypeError("El identificador de persona está vacío")
    return cleaned


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Captura imágenes de señas desde /dev/video0"
    )
    parser.add_argument("--persona", required=True, type=safe_name)
    parser.add_argument("--sena", required=True, choices=VALID_SIGNS)
    parser.add_argument("--cantidad", type=int, default=50)
    parser.add_argument("--camara", type=int, default=0)
    return parser.parse_args()


def draw_interface(
    frame,
    roi_coordinates: tuple[int, int, int, int],
    sign: str,
    person: str,
    captured: int,
    total: int,
    capturing: bool,
):
    x1, y1, x2, y2 = roi_coordinates
    color = (0, 220, 120) if capturing else (255, 190, 40)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 72), (9, 20, 35), -1)
    cv2.putText(
        frame,
        f"SENA: {DISPLAY_NAMES[sign]} | PERSONA: {person}",
        (15, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    state = "CAPTURANDO" if capturing else "ESPACIO: iniciar | Q: salir"
    cv2.putText(
        frame,
        f"{state} | {captured}/{total}",
        (15, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame


def main() -> int:
    args = arguments()
    if args.cantidad < 10 or args.cantidad > 500:
        raise SystemExit("La cantidad debe estar entre 10 y 500")

    output_dir = BASE_DIR / "data" / "senas" / args.sena
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(args.camara, cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    if not camera.isOpened():
        raise SystemExit(
            f"No se pudo abrir /dev/video{args.camara}. Cierre primero el dashboard."
        )

    captured = 0
    capturing = False
    last_capture = 0.0
    session = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Coloque las manos dentro del recuadro.")
    print("Presione ESPACIO para iniciar y Q para salir.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("No se recibió imagen de la cámara.")
                break

            height, width = frame.shape[:2]
            roi_size = min(360, height - 90, width - 40)
            x1 = (width - roi_size) // 2
            y1 = max(80, (height - roi_size) // 2)
            x2, y2 = x1 + roi_size, y1 + roi_size

            now = time.monotonic()
            if capturing and captured < args.cantidad and now - last_capture >= 0.20:
                roi = frame[y1:y2, x1:x2]
                roi = cv2.resize(roi, (320, 320), interpolation=cv2.INTER_AREA)
                filename = output_dir / (
                    f"{args.persona}_{session}_{captured + 1:03d}.jpg"
                )
                cv2.imwrite(str(filename), roi, [cv2.IMWRITE_JPEG_QUALITY, 94])
                captured += 1
                last_capture = now

            display = draw_interface(
                frame.copy(),
                (x1, y1, x2, y2),
                args.sena,
                args.persona,
                captured,
                args.cantidad,
                capturing,
            )
            cv2.imshow("Captura de senas - EdgeVision", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32 and captured < args.cantidad:
                capturing = True

            if captured >= args.cantidad:
                print(f"Captura completa: {captured} imágenes en {output_dir}")
                time.sleep(1)
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

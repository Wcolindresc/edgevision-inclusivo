# EdgeVision Inclusivo — Dashboard inicial

Dashboard local para la Raspberry Pi 4B. Utiliza la webcam, el modelo
`yolo26n_ncnn_model` y el entorno virtual creados durante la preparación del
proyecto.

## Ubicación esperada

Al descomprimir, la estructura debe quedar así:

```text
/home/soporte/edge-ai/
├── venv/
├── yolo26n.pt
├── yolo26n_ncnn_model/
└── dashboard_edge_ai/
```

## Instalación

```bash
cd ~/edge-ai/dashboard_edge_ai
source ~/edge-ai/venv/bin/activate
pip install -r requirements.txt
chmod +x start.sh
```

## Ejecución

Cierre cualquier programa que esté utilizando `/dev/video0` y ejecute:

```bash
cd ~/edge-ai/dashboard_edge_ai
./start.sh
```

Abra en la Raspberry:

```text
http://127.0.0.1:5000
```

Desde otra computadora de la misma red, sustituya la IP por la dirección de la
Raspberry, por ejemplo:

```text
http://192.168.0.13:5000
```

Para detener el servidor, vuelva a la terminal y presione `Ctrl + C`.

## Variables opcionales

```bash
CAMERA_INDEX=0 CONFIDENCE=0.40 IMAGE_SIZE=320 ./start.sh
```

Cuando exista un Worker, se puede mostrar en el panel con:

```bash
WORKER_NODES="edge-worker01=192.168.0.21" ./start.sh
```

## Verificación

- Estado: `http://IP_DE_LA_RASPBERRY:5000/health`
- Métricas: `http://IP_DE_LA_RASPBERRY:5000/api/status`

El modo de reconocimiento de señas aparece como próxima etapa y permanecerá
deshabilitado hasta integrar el modelo entrenado por el equipo.

## Captura del dataset de señas

Detenga primero el dashboard con `Ctrl + C`. Para registrar 50 muestras de la
seña AYUDA para una persona:

```bash
cd ~/edge-ai/dashboard_edge_ai
source ~/edge-ai/venv/bin/activate
python capturar_senas.py --persona wilmer --sena ayuda --cantidad 50
```

Las señas admitidas son `ayuda`, `agua`, `bano`, `si`, `no` y `emergencia`.
Cada integrante debe utilizar un identificador diferente. Coloque las manos
dentro del recuadro, presione la barra espaciadora para iniciar y cambie
ligeramente la posición, distancia y orientación durante la captura.

Las imágenes se guardan en:

```text
dashboard_edge_ai/data/senas/<nombre-de-la-sena>/
```

from ultralytics import YOLO

# Cargar modelo YOLO
model = YOLO('yolov8n.pt')  # o yolov8s.pt, yolov8m.pt, etc.

# Tracking con BoT-SORT
results = model.track(
    source='video.mp4',
    tracker='bytetrack.yaml',
    conf=0.4,              # Confianza mínima de detección
    iou=0.45,               # IoU para NMS
    show=True,              # Mostrar video mientras procesa
    save=True,              # Guardar resultado
    # classes=[0]             # Solo personas (clase 0 en COCO)
)

# Procesar resultados
for result in results:
    boxes = result.boxes
    for box in boxes:
        track_id = int(box.id) if box.id is not None else -1
        conf = float(box.conf)
        cls = int(box.cls)
        xyxy = box.xyxy[0].tolist()
        
        print(f"ID: {track_id}, Clase: {cls}, Conf: {conf:.2f}")
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# Configuraciones estratégicas para smart cooler
configs = [
    # Test 1: Baseline estándar
    {
        'name': 'baseline',
        'conf': 0.3, 
        'iou': 0.45,
        'tracker': 'bytetrack.yaml'
    },
    
    # Test 2: Alta sensibilidad (no perder productos)
    {
        'name': 'high_sensitivity',
        'conf': 0.2,  # Más bajo para detectar todo
        'iou': 0.4,   # Más permisivo
        'tracker': 'bytetrack.yaml'
    },
    
    # Test 3: Alta precisión (evitar falsos positivos)
    {
        'name': 'high_precision',
        'conf': 0.5,  # Solo detecciones muy seguras
        'iou': 0.6,   # Más estricto con duplicados
        'tracker': 'bytetrack.yaml'
    },
    
    # Test 4: BoT-SORT baseline (con ReID)
    {
        'name': 'botsort_baseline',
        'conf': 0.3,
        'iou': 0.45,
        'tracker': 'botsort.yaml'
    },
    
    # Test 5: BoT-SORT sensible
    {
        'name': 'botsort_sensitive',
        'conf': 0.25,
        'iou': 0.4,
        'tracker': 'botsort.yaml'
    },
    
    # Test 6: Para objetos pequeños
    {
        'name': 'small_objects',
        'conf': 0.25,
        'iou': 0.3,  # Muy permisivo (objetos pequeños tienen más movimiento relativo)
        'tracker': 'bytetrack.yaml'
    },
]

for i, config in enumerate(configs, 1):
    print(f"\n{'='*60}")
    print(f"Prueba {i}: {config['name']}")
    print(f"Configuración: conf={config['conf']}, iou={config['iou']}, tracker={config['tracker']}")
    print(f"{'='*60}")
    
    results = model.track(
        source='video.mp4',
        conf=config['conf'],
        iou=config['iou'],
        tracker=config['tracker'],
        save=True,
        project='experiments',
        name=config['name'],
        verbose=False,  # Menos output en consola
        stream=True
    )
    
    print(f"✓ Completado: {config['name']}")
    print(f"  Resultados guardados en: experiments/{config['name']}/")
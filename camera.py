import cv2
import numpy as np
import time

def configure_camera(device_id=2):
    """Configura la cámara con diferentes exposiciones y muestra resultados"""
    
    cap = cv2.VideoCapture(device_id)
    
    if not cap.isOpened():
        print(f"Error: No se pudo abrir /dev/video{device_id}")
        return
    
    # Configuración base
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    # Exposiciones a probar (de más rápido a más lento)
    exposures = [10, 30, 50, 80, 100, 150, 200]
    current_exp_idx = 2  # Empezar con 50
    
    # Modo manual
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual en algunas cámaras
    
    def calculate_blur(image):
        """Calcula el nivel de blur usando Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def calculate_brightness(image):
        """Calcula brillo promedio"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray.mean()
    
    print("Controles:")
    print("  'q' - Salir")
    print("  '+' - Aumentar exposición (más lento, más luz)")
    print("  '-' - Disminuir exposición (más rápido, menos blur)")
    print("  's' - Guardar frame actual")
    print("  'r' - Reset a automático")
    
    while True:
        # Configurar exposición actual
        exposure = exposures[current_exp_idx]
        cap.set(cv2.CAP_PROP_EXPOSURE, -exposure)  # Valor negativo en OpenCV
        
        # Leer frames (descartar algunos para que se aplique la config)
        for _ in range(5):
            cap.read()
        
        ret, frame = cap.read()
        if not ret:
            print("Error capturando frame")
            break
        
        # Calcular métricas
        blur_score = calculate_blur(frame)
        brightness = calculate_brightness(frame)
        
        # Determinar color según calidad
        if blur_score > 500:
            blur_color = (0, 255, 0)  # Verde = Sharp
            blur_text = "SHARP"
        elif blur_score > 200:
            blur_color = (0, 255, 255)  # Amarillo = OK
            blur_text = "OK"
        else:
            blur_color = (0, 0, 255)  # Rojo = Blur
            blur_text = "BLURRY"
        
        # Mostrar información en frame
        cv2.putText(frame, f"Exposure: {exposure}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Blur: {blur_score:.1f} ({blur_text})", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, blur_color, 2)
        cv2.putText(frame, f"Brightness: {brightness:.1f}", 
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Barra de progreso de exposición
        bar_length = 400
        bar_pos = int((current_exp_idx / (len(exposures) - 1)) * bar_length)
        cv2.rectangle(frame, (10, 130), (10 + bar_length, 150), (100, 100, 100), -1)
        cv2.rectangle(frame, (10, 130), (10 + bar_pos, 150), (0, 255, 0), -1)
        cv2.putText(frame, "Rapido", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Lento", (350, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Camera Configuration Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            if current_exp_idx < len(exposures) - 1:
                current_exp_idx += 1
                print(f"Exposición aumentada a: {exposures[current_exp_idx]}")
        elif key == ord('-') or key == ord('_'):
            if current_exp_idx > 0:
                current_exp_idx -= 1
                print(f"Exposición disminuida a: {exposures[current_exp_idx]}")
        elif key == ord('s'):
            filename = f"capture_exp_{exposure}_blur_{blur_score:.0f}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Frame guardado: {filename}")
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Volver a auto
            print("Modo automático restaurado")
            time.sleep(1)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Volver a manual
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Mostrar configuración óptima encontrada
    print("\n" + "="*60)
    print(f"Configuración recomendada:")
    print(f"v4l2-ctl -d /dev/video2 --set-ctrl=auto_exposure=1")
    print(f"v4l2-ctl -d /dev/video2 --set-ctrl=exposure_time_absolute={exposures[current_exp_idx]}")
    print("="*60)

if __name__ == "__main__":
    configure_camera(2)
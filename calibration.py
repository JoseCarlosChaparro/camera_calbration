import cv2
import numpy as np

def quick_distortion_test():
    """Test visual rápido de distorsión fisheye"""
    
    cap = cv2.VideoCapture(2)
    
    # Configurar para mejor calidad
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    print("Capturando frame de prueba...")
    
    # Descartar primeros frames
    for _ in range(10):
        cap.read()
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Error capturando frame")
        return
    
    h, w = frame.shape[:2]
    print(f"✓ Frame capturado: {w}x{h}")
    
    # Crear versión con cuadrícula de prueba
    overlay = frame.copy()
    
    # Cuadrícula cada 100px
    spacing = 100
    for x in range(0, w, spacing):
        cv2.line(overlay, (x, 0), (x, h), (0, 255, 0), 2)
    
    for y in range(0, h, spacing):
        cv2.line(overlay, (0, y), (w, y), (0, 255, 0), 2)
    
    # Círculos concéntricos
    center = (w//2, h//2)
    for radius in [200, 400, 600, 800]:
        if radius < min(w, h)//2:
            cv2.circle(overlay, center, radius, (255, 0, 0), 3)
    
    # Diagonales
    cv2.line(overlay, (0, 0), (w, h), (0, 0, 255), 3)
    cv2.line(overlay, (w, 0), (0, h), (0, 0, 255), 3)
    
    # Marcar zonas
    cv2.putText(overlay, "CENTRO (menos distorsion)", 
                (w//2-200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(overlay, "BORDES (mas distorsion)", 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    # Blend
    result = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
    
    # Guardar
    cv2.imwrite('distortion_test.jpg', result)
    cv2.imwrite('original_frame.jpg', frame)
    
    print("\n" + "="*70)
    print("TEST DE DISTORSIÓN COMPLETADO")
    print("="*70)
    print("\nArchivos guardados:")
    print("  1. distortion_test.jpg  - Frame con cuadrícula de prueba")
    print("  2. original_frame.jpg   - Frame original sin overlay")
    print("\n" + "="*70)
    print("CÓMO INTERPRETAR:")
    print("="*70)
    print("\n🔍 Observa las LÍNEAS VERDES (cuadrícula):")
    print("   ├─ RECTAS en todo el frame → Distorsión mínima/ninguna")
    print("   ├─ Ligeramente curvas en bordes → Distorsión moderada")
    print("   └─ MUY curvas en bordes → Distorsión severa (fisheye)")
    print("\n🔵 Observa los CÍRCULOS AZULES:")
    print("   ├─ Perfectamente circulares → Sin distorsión")
    print("   ├─ Ligeramente ovalados → Distorsión moderada")
    print("   └─ Muy ovalados/deformados → Distorsión severa")
    print("\n🔴 Observa las LÍNEAS ROJAS (diagonales):")
    print("   ├─ Completamente rectas → Sin distorsión")
    print("   └─ Curvas/arqueadas → Hay distorsión")
    print("\n" + "="*70)
    print("RECOMENDACIÓN:")
    print("="*70)
    print("\nPara tu cámara TM Technology (10bb:2b08):")
    print("\n➤ SI las líneas están MUY CURVAS en los bordes:")
    print("  → Definitivamente necesitas calibración (15 min)")
    print("  → Te mejorará detecciones en bordes ~20-40%")
    print("\n➤ SI las líneas están CASI RECTAS:")
    print("  → Opcional, pero recomendado para producción")
    print("  → Mejora marginal pero vale la pena")
    print("\n➤ Para SMART COOLER específicamente:")
    print("  → Productos están en BORDES (zona más distorsionada)")
    print("  → Recomiendo calibrar SÍ O SÍ")
    print("="*70)
    
    # Análisis automático simple
    print("\n📊 ANÁLISIS AUTOMÁTICO:")
    print("-"*70)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Comparar bordes vs centro
    center_region = gray[h//3:2*h//3, w//3:2*w//3]
    edge_region = np.concatenate([
        gray[0:h//10, :].flatten(),
        gray[9*h//10:h, :].flatten(),
        gray[:, 0:w//10].flatten(),
        gray[:, 9*w//10:w].flatten()
    ])
    
    # Detectar líneas con Hough en bordes
    edges = cv2.Canny(gray, 50, 150)
    lines_border = cv2.HoughLinesP(edges[:h//5, :], 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
    
    if lines_border is not None:
        num_lines = len(lines_border)
        print(f"Líneas detectadas en borde superior: {num_lines}")
        if num_lines < 5:
            print("  → Pocas líneas = probable curvatura significativa")
        else:
            print("  → Múltiples líneas = distorsión moderada")
    
    print("-"*70)
    print("\n✅ Abre 'distortion_test.jpg' para inspección visual")

if __name__ == "__main__":
    quick_distortion_test()
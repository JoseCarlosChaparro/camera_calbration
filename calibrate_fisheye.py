# Guardar como calibrate_fisheye.py
import cv2
import numpy as np
import glob
import os

class FisheyeCalibrator:
    def __init__(self, checkerboard_size=(9, 6), device=2):
        self.board_size = checkerboard_size
        self.device = device
        self.images_captured = 0
        self.target_images = 20
        
        # Preparar puntos del objeto
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 
                                     0:checkerboard_size[1]].T.reshape(-1, 2)
        
        self.objpoints = []
        self.imgpoints = []
        
        os.makedirs('calibration_images', exist_ok=True)
    
    def capture_calibration_images(self):
        """Captura imágenes interactivamente"""
        
        cap = cv2.VideoCapture(self.device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        print("\n" + "="*70)
        print("CAPTURA DE IMÁGENES DE CALIBRACIÓN")
        print("="*70)
        print("\nInstrucciones:")
        print("  1. Muestra el tablero de ajedrez a la cámara")
        print("  2. Presiona ESPACIO cuando veas el rectángulo VERDE")
        print("  3. Captura desde diferentes ángulos y distancias")
        print("  4. Necesitas {} imágenes válidas".format(self.target_images))
        print("  5. Presiona 'q' para terminar")
        print("\nTips:")
        print("  - Varía el ángulo (frontal, lateral, arriba, abajo)")
        print("  - Varía la distancia (cerca, lejos)")
        print("  - Cubre todas las áreas del frame (centro, bordes, esquinas)")
        print("  - El tablero debe estar PLANO (no doblado)")
        print("="*70 + "\n")
        
        while self.images_captured < self.target_images:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Buscar esquinas
            ret_corners, corners = cv2.findChessboardCorners(
                gray, self.board_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + 
                cv2.CALIB_CB_FAST_CHECK + 
                cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            # Visualización
            display = frame.copy()
            
            if ret_corners:
                # Refinar esquinas
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                
                # Dibujar esquinas encontradas
                cv2.drawChessboardCorners(display, self.board_size, 
                                         corners_refined, ret_corners)
                
                # Indicador verde
                cv2.putText(display, "PATRON DETECTADO - Presiona ESPACIO", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 3)
                cv2.rectangle(display, (10, 10), 
                             (display.shape[1]-10, display.shape[0]-10), 
                             (0, 255, 0), 5)
            else:
                cv2.putText(display, "Buscando patron...", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 0, 255), 2)
            
            # Contador
            progress = f"{self.images_captured}/{self.target_images}"
            cv2.putText(display, progress, (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
            
            # Barra de progreso
            bar_w = 400
            bar_h = 30
            bar_x, bar_y = 50, 130
            progress_pct = self.images_captured / self.target_images
            
            cv2.rectangle(display, (bar_x, bar_y), 
                         (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
            cv2.rectangle(display, (bar_x, bar_y), 
                         (bar_x + int(bar_w * progress_pct), bar_y + bar_h), 
                         (0, 255, 0), -1)
            
            cv2.imshow('Calibracion Fisheye', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and ret_corners:
                # Guardar imagen
                filename = f'calibration_images/calib_{self.images_captured:02d}.jpg'
                cv2.imwrite(filename, frame)
                
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners_refined)
                self.images_captured += 1
                
                print(f"✓ Imagen {self.images_captured}/{self.target_images} capturada")
                
            elif key == ord('q'):
                print("\nCaptura cancelada por usuario")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if self.images_captured < 10:
            print(f"\n❌ Error: Solo se capturaron {self.images_captured} imágenes")
            print("   Necesitas al menos 10 para calibración confiable")
            return False
        
        print(f"\n✓ Captura completada: {self.images_captured} imágenes")
        return True
    
    def calibrate(self):
        """Ejecuta la calibración fisheye"""
        
        if len(self.objpoints) < 10:
            print("❌ No hay suficientes imágenes para calibrar")
            return None, None
        
        print("\n" + "="*70)
        print("EJECUTANDO CALIBRACIÓN FISHEYE")
        print("="*70)
        print("\nEsto puede tomar 30-60 segundos...")
        
        # Obtener tamaño de imagen
        sample_img = cv2.imread('calibration_images/calib_00.jpg', 0)
        img_shape = sample_img.shape[::-1]
        
        # Matrices de calibración
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        
        # Preparar puntos
        objpoints = np.array(self.objpoints, dtype=np.float64)
        objpoints = np.reshape(objpoints, (len(objpoints), 1, -1, 3))
        imgpoints = np.array(self.imgpoints, dtype=np.float64)
        imgpoints = np.reshape(imgpoints, (len(imgpoints), 1, -1, 2))
        
        # Flags de calibración
        calibration_flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
            cv2.fisheye.CALIB_CHECK_COND +
            cv2.fisheye.CALIB_FIX_SKEW
        )
        
        # Calibrar
        ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints, imgpoints, img_shape,
            K, D, None, None,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
        
        print(f"\n✓ Calibración completada")
        print(f"  Error RMS: {ret:.4f}")
        print(f"\nMatriz de cámara (K):")
        print(K)
        print(f"\nCoeficientes de distorsión (D):")
        print(D.ravel())
        
        # Guardar parámetros
        np.savez('fisheye_calibration.npz', 
                 K=K, D=D, img_shape=img_shape, rms_error=ret)
        print("\n✓ Parámetros guardados en 'fisheye_calibration.npz'")
        
        return K, D
    
    def test_calibration(self, K, D):
        """Prueba la calibración con antes/después"""
        
        print("\n" + "="*70)
        print("PROBANDO CALIBRACIÓN")
        print("="*70)
        
        # Tomar imagen de prueba
        cap = cv2.VideoCapture(self.device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        for _ in range(10):
            cap.read()
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ Error capturando frame de prueba")
            return
        
        h, w = frame.shape[:2]
        
        # Crear mapas de undistortion
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K, (w, h), cv2.CV_16SC2
        )
        
        # Aplicar corrección
        undistorted = cv2.remap(frame, map1, map2, 
                                cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT)
        
        # Comparación lado a lado
        comparison = np.hstack([frame, undistorted])
        
        # Añadir labels
        cv2.putText(comparison, "ORIGINAL (Distorsionado)", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.5, (0, 0, 255), 3)
        cv2.putText(comparison, "CORREGIDO", 
                   (w + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.5, (0, 255, 0), 3)
        
        # Guardar
        cv2.imwrite('calibration_result.jpg', comparison)
        cv2.imwrite('undistorted_test.jpg', undistorted)
        
        print("\n✓ Resultados guardados:")
        print("  - calibration_result.jpg (comparación)")
        print("  - undistorted_test.jpg (solo corregido)")
        print("\n¡Abre los archivos para ver la diferencia!")
        
        # Mostrar
        cv2.imshow('Resultado: Original vs Corregido', comparison)
        print("\nPresiona cualquier tecla para cerrar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    print("\n" + "="*70)
    print("SISTEMA DE CALIBRACIÓN FISHEYE AUTOMATIZADO")
    print("Para cámara TM Technology (10bb:2b08)")
    print("="*70)
    
    calibrator = FisheyeCalibrator(checkerboard_size=(9, 6), device=2)
    
    # Paso 1: Capturar imágenes
    if not calibrator.capture_calibration_images():
        return
    
    # Paso 2: Calibrar
    K, D = calibrator.calibrate()
    
    if K is None:
        return
    
    # Paso 3: Probar
    calibrator.test_calibration(K, D)
    
    print("\n" + "="*70)
    print("¡CALIBRACIÓN COMPLETADA!")
    print("="*70)
    print("\nArchivos generados:")
    print("  ✓ fisheye_calibration.npz - Parámetros de calibración")
    print("  ✓ calibration_result.jpg - Comparación antes/después")
    print("  ✓ calibration_images/ - Imágenes de calibración")
    print("\nAhora puedes usar estos parámetros en tu sistema de tracking")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
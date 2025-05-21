# test_video.py
# coding:UTF-8
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import csv
import time
import cv2
from udp_service import UdpService
import os
from datetime import datetime
import winsound
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Variables
angles_muslo = []
angles_espinilla = []
times = []
video_frames = []
start_time = time.time()
sound_played = False
cycle_data = {'times': [], 'muslo': [], 'espinilla': []}
predictions = []  # Para almacenar las predicciones de los ciclos
window_size = 120  # Tamaño de la ventana igual al entrenamiento

# Archivo CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f'knee_movement_data_{timestamp}.csv'
try:
    csv_file = open(csv_filename, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file, delimiter=';')
    csv_writer.writerow(['Time (s)', 'AngleMuslo (deg)', 'AngleEspinilla (deg)'])
except Exception as e:
    print(f"Error al crear el CSV: {e}")
    csv_file = None
    csv_writer = None

# Configuración de video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()
video_filename = f'knee_movement_video_{timestamp}.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(video_filename, fourcc, 10.0, (640, 480))

# Cargar modelo y escalador
try:
    model = joblib.load('C:/Users/USUARIO/Desktop/INGENIERÍA_BIOMÉDICA_UPM/4-CUARTO/TFG/python/WitWIFI_WT901WIFI/Python/901WIFI_python_sdk/knee_movement_model.pkl')
    scaler = np.load('C:/Users/USUARIO/Desktop/INGENIERÍA_BIOMÉDICA_UPM/4-CUARTO/TFG/python/WitWIFI_WT901WIFI/Python/901WIFI_python_sdk/scaler.npy', allow_pickle=True).item()
except FileNotFoundError:
    print("Error: Modelo o escalador no encontrados. Entrena el modelo primero.")
    exit()

# Variables
server = None
running = True
SENSOR_MUSLO_ID = "WT5500003294"
SENSOR_ESPINILLA_ID = "WT5500003326"
temp_data = {SENSOR_MUSLO_ID: None, SENSOR_ESPINILLA_ID: None}

def updateData(DeviceModel):
    global angles_muslo, angles_espinilla, times, running, temp_data, cycle_data
    if not running:
        return
    angle = DeviceModel.get("AngleX")
    device_id = DeviceModel.deviceName
    current_time = time.time() - start_time
    if angle is not None:
        temp_data[device_id] = angle
        if temp_data[SENSOR_MUSLO_ID] is not None and temp_data[SENSOR_ESPINILLA_ID] is not None:
            angles_muslo.append(temp_data[SENSOR_MUSLO_ID])
            angles_espinilla.append(temp_data[SENSOR_ESPINILLA_ID])
            times.append(current_time)
            cycle_data['times'].append(current_time)
            cycle_data['muslo'].append(temp_data[SENSOR_MUSLO_ID])
            cycle_data['espinilla'].append(temp_data[SENSOR_ESPINILLA_ID])
            temp_data[SENSOR_MUSLO_ID] = None
            temp_data[SENSOR_ESPINILLA_ID] = None
            if csv_writer is not None:
                try:
                    csv_writer.writerow([current_time, angles_muslo[-1], angles_espinilla[-1]])
                    csv_file.flush()
                except Exception as e:
                    print(f"Error al escribir en el CSV: {e}")
            if len(angles_muslo) > 200:
                angles_muslo.pop(0)
                angles_espinilla.pop(0)
                times.pop(0)
            if len(cycle_data['times']) > 200:
                cycle_data['times'].pop(0)
                cycle_data['muslo'].pop(0)
                cycle_data['espinilla'].pop(0)


def detect_cycle():
    global cycle_data, predictions
    if len(cycle_data['times']) < window_size:
        print(f"No hay suficientes datos: {len(cycle_data['times'])} puntos")
        return None
    # Usar una ventana deslizante
    cycle_times = cycle_data['times'][-window_size:]
    cycle_muslo = cycle_data['muslo'][-window_size:]
    cycle_espinilla = cycle_data['espinilla'][-window_size:]
    sample_points = 60
    muslo_resampled = np.interp(np.linspace(0, cycle_times[-1] - cycle_times[0], sample_points), 
                              [t - cycle_times[0] for t in cycle_times], cycle_muslo)
    espinilla_resampled = np.interp(np.linspace(0, cycle_times[-1] - cycle_times[0], sample_points), 
                                  [t - cycle_times[0] for t in cycle_times], cycle_espinilla)

    # Calcular la derivada
    muslo_diff = np.diff(muslo_resampled)
    espinilla_diff = np.diff(espinilla_resampled)

    
    max_muslo_angle = np.max(np.abs(muslo_resampled))

    # Combinar características
    trial_features = np.concatenate([muslo_resampled, espinilla_resampled, muslo_diff, espinilla_diff, [max_muslo_angle]])
    prediction = model.predict(scaler.transform(trial_features.reshape(1, -1)))[0]
    
      
    if max_muslo_angle > 30:  
        prediction = 0  
        
    predictions.append(prediction)
    print(f"Predicción para ventana: {prediction} (max_muslo_angle: {max_muslo_angle:.1f}°)")
    return trial_features

def animate(i, ax_plot, ax_video):
    global sound_played, cap, video_writer
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el fotograma.")
        return
    video_writer.write(frame)
    video_frames.append((time.time() - start_time, frame))
    ax_plot.clear()
    if angles_muslo and angles_espinilla and times:
        ax_plot.plot(times, angles_muslo, label='AngleX (Muslo)', color='green')
        ax_plot.plot(times, angles_espinilla, label='AngleX (Espinilla)', color='blue')
        ax_plot.set_xlabel('Tiempo (s)')
        ax_plot.set_ylabel('Ángulo (grados)')
        ax_plot.set_title('Flexión-Extensión de Rodilla')
        ax_plot.grid(True)
        ax_plot.legend()
        min_angle = min(min(angles_muslo), min(angles_espinilla)) - 10
        max_angle = max(max(angles_muslo), max(angles_espinilla)) + 10
        ax_plot.set_ylim(min_angle, max_angle)
        
        trial_features = detect_cycle()
        if trial_features is not None:
            prediction = predictions[-1]
            print(f"Procesando predicción: {prediction}")
            if prediction == 1:
                ax_plot.set_facecolor('lightgreen')
                if not sound_played:
                    print("¡Ciclo correcto!")
                    winsound.Beep(500, 200)
                    sound_played = True
            else:
                ax_plot.set_facecolor('white')
                sound_played = False
        else:
            ax_plot.set_facecolor('white')
            sound_played = False  # Reiniciar para permitir pitido en el próximo ciclo correcto
    else:
        ax_plot.set_xlabel('Tiempo (s)')
        ax_plot.set_ylabel('Ángulo (grados)')
        ax_plot.set_title('Flexión-Extensión de Rodilla')
        ax_plot.grid(True)
        ax_plot.set_ylim(-180, 180)
    ax_video.clear()
    ax_video.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax_video.axis('off')
    ax_video.set_title('Video en Tiempo Real')

def stop_program(event):
    global running, csv_filename, predictions, fig
    running = False
    print("Deteniendo grabación...")
    print(f"Predicciones registradas: {predictions}")
    if server:
        server.stop()
    if csv_file:
        csv_file.flush()
        csv_file.close()
    if cap.isOpened():
        cap.release()
    if video_writer:
        video_writer.release()
    
    # Etiquetar automáticamente el CSV según la predicción más común
    if predictions:
        label = 1 if sum(predictions) > len(predictions) // 2 else 0
        print(f"Etiqueta seleccionada (predicción más común): {label}")
    else:
        label = 0
        print("No se detectaron ventanas, usando etiqueta por defecto: 0")
    new_csv_filename = csv_filename.replace('.csv', f'_label_{label}.csv')
    os.rename(csv_filename, new_csv_filename)
    print(f"Datos guardados con etiqueta automática en: {new_csv_filename}")
    
    # Cerrar la ventana
    plt.close(fig)

if __name__ == '__main__':
    plt.style.use('fivethirtyeight')
    fig = plt.figure(figsize=(12, 8))
    ax_plot = fig.add_subplot(121)
    ax_plot.set_title('Flexión-Extensión de Rodilla')
    ax_plot.set_xlabel('Tiempo (s)')
    ax_plot.set_ylabel('Ángulo (grados)')
    ax_plot.grid(True)
    ax_video = fig.add_subplot(122)
    ax_video.axis('off')
    ax_video.set_title('Video en Tiempo Real')
    
    # Botón para detener
    ax_button_stop = plt.axes([0.8, 0.05, 0.15, 0.05])
    button = Button(ax_button_stop, 'Detener')
    button.on_clicked(stop_program)
    
    ani = FuncAnimation(fig, animate, fargs=(ax_plot, ax_video), interval=100, cache_frame_data=False)
    server = UdpService(1399, updateData)
    server.start()
    try:
        plt.show()
    except KeyboardInterrupt:
        print("Deteniendo el servidor...")
        if server:
            server.stop()
        if csv_file:
            csv_file.close()
        if cap.isOpened():
            cap.release()
        if video_writer:
            video_writer.release()
# preprocess_data.py
# coding:UTF-8
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Directorio de datos y salida fijo
dataset_dir = "C:/Users/USUARIO/Desktop/INGENIERÍA_BIOMÉDICA_UPM/4-CUARTO/TFG/python/WitWIFI_WT901WIFI/Python/901WIFI_python_sdk/dataset"
output_dir = "C:/Users/USUARIO/Desktop/INGENIERÍA_BIOMÉDICA_UPM/4-CUARTO/TFG/python/WitWIFI_WT901WIFI/Python/901WIFI_python_sdk/"

# Listas para almacenar características y etiquetas
features_list = []
labels_list = []

# Parámetros
sample_points = 60  # Número de puntos resampleados por ciclo

# Procesar cada CSV
for filename in os.listdir(dataset_dir):
    if filename.endswith(".csv") and "label" in filename:
        label = int(filename.split("_label_")[-1].split(".csv")[0])  # 1 o 0
        df = pd.read_csv(os.path.join(dataset_dir, filename), delimiter=";")
        times = df["Time (s)"].values
        muslo_angles = df["AngleMuslo (deg)"].values
        espinilla_angles = df["AngleEspinilla (deg)"].values

        # Resamplear a una longitud fija
        muslo_resampled = np.interp(np.linspace(0, times[-1], sample_points), times, muslo_angles)
        espinilla_resampled = np.interp(np.linspace(0, times[-1], sample_points), times, espinilla_angles)

        # Calcular la derivada (velocidad de cambio)
        muslo_diff = np.diff(muslo_resampled)
        espinilla_diff = np.diff(espinilla_resampled)

        # Nueva característica: ángulo máximo del muslo
        max_muslo_angle = np.max(np.abs(muslo_resampled))  # Módulo para considerar desviaciones grandes

        # Combinar características: ángulos resampleados + derivadas + ángulo máximo
        trial_features = np.concatenate([muslo_resampled, espinilla_resampled, muslo_diff, espinilla_diff, [max_muslo_angle]])
        features_list.append(trial_features)
        labels_list.append(label)

# Convertir a arrays numpy
features = np.array(features_list)
labels = np.array(labels_list)

# Escalar características
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Crear el directorio de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Guardar
np.save(os.path.join(output_dir, "features.npy"), features_scaled)
np.save(os.path.join(output_dir, "labels.npy"), labels)
np.save(os.path.join(output_dir, "scaler.npy"), scaler)

print(f"Procesados {len(features)} archivos. Características guardadas en {output_dir}features.npy, {output_dir}labels.npy y {output_dir}scaler.npy.")
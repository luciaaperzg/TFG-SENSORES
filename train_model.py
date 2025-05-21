# train_model.py
# coding:UTF-8
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Directorio fijo donde están los archivos
data_dir = "C:/Users/USUARIO/Desktop/INGENIERÍA_BIOMÉDICA_UPM/4-CUARTO/TFG/python/WitWIFI_WT901WIFI/Python/901WIFI_python_sdk/"

# Cargar datos preprocesados
features = np.load(data_dir + "features.npy")
labels = np.load(data_dir + "labels.npy")

# Dividir en entrenamiento y prueba con un 30% para prueba
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, shuffle=True)

# Entrenar modelo Random Forest con hiperparámetros 
model = RandomForestClassifier(
    n_estimators=20,           
    max_depth=2,              
    min_samples_split=20,      
    min_samples_leaf=10,       
    max_features="sqrt",       
    random_state=None          
)
model.fit(X_train, y_train)

# Evaluar precisión
accuracy = model.score(X_test, y_test)
print(f"Precisión en el conjunto de prueba: {accuracy:.2f}")

# Guardar modelo en el mismo directorio
joblib.dump(model, data_dir + "knee_movement_model.pkl")
print(f"Modelo guardado como {data_dir}knee_movement_model.pkl")
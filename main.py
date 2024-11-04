# código para extraer fotogramas/características:

import os
import cv2
import random
import torch
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18
import torch.nn.functional as F
import csv
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import pickle

# ----------------------------- TRAINING -----------------------------

# Directorio con los videos de entrenamiento
train_dir = r"C:\Users\alexa\OneDrive\Desktop\Machine Learning\proyecto_clustering\train_subset"

# Directorio donde se van a guardar los fotogramas
output_dir = r"C:\Users\alexa\OneDrive\Desktop\Machine Learning\proyecto_clustering\features"

# Inicializar el modelo preentrenado de acción
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = r3d_18(pretrained=True)
model.eval().to(device)

# Transformaciones necesarias para el modelo
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
])

# Función para extraer características de los fotogramas de un video
def extract_features(video_path, model, transform):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected_frames = random.sample(range(frame_count), min(16, frame_count))

    frames = []
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in selected_frames:
            frame = transform(cv2.resize(frame, (224, 224)))
            frames.append(frame)

    cap.release()

    if len(frames) < 16:
        return None
    frames_tensor = torch.stack(frames[:16], dim=1).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(frames_tensor)

    return features.squeeze().cpu().numpy()

# ----------------------------- TESTING -----------------------------

# Directorio con los videos de prueba
test_dir = r"C:\Users\alexa\OneDrive\Desktop\Machine Learning\proyecto_clustering\test_subset"

# Cargar el CSV con los IDs de los videos que queremos usar para el testing
test_subset_csv = "test_subset_10.csv"
test_ids = pd.read_csv(test_subset_csv)['youtube_id'].tolist()

# Filtrar solo los archivos de video en el directorio de testing que están en test_subset_10
all_test_videos = [f for f in os.listdir(test_dir) if f.endswith('.mp4') and f.split('_')[0] in test_ids]

print(f"Total de videos en testing seleccionados: {len(all_test_videos)}")

# Crear una lista para almacenar los resultados de testing
resultados_testing = []
features_testing = []

# Procesar cada video de testing seleccionado y extraer características
for video in all_test_videos:
    video_path = os.path.join(test_dir, video)
    print(f"Procesando {video}...")

    # Extraer características
    features = extract_features(video_path, model, transform)

    if features is not None:
        youtube_id = video.split('_')[0]
        features_testing.append(features)
        resultados_testing.append([youtube_id])  # Agrega solo el ID del video
        print(f"Características extraídas para {video}")
    else:
        print(f"No se pudo procesar {video} (insuficientes frames)")

# Convertir la lista de características en un array de numpy para aplicar PCA
feature_matrix = np.array(features_testing)

# Aplicar PCA para reducir la dimensionalidad
pca = PCA(n_components=50)
reduced_features = pca.fit_transform(feature_matrix)

# Guardar los vectores de características reducidos en un archivo binario usando pickle
with open("reduced_features_testing.pkl", "wb") as archivo_pickle:
    pickle.dump(reduced_features, archivo_pickle)

print("Reducción de dimensionalidad con PCA completada y guardada en reduced_features_testing.pkl.")

# Guardar resultados de testing en un nuevo CSV
with open("resultados_acciones_testing.csv", mode="w", newline='') as archivo_csv:
    escritor_csv = csv.writer(archivo_csv)
    escritor_csv.writerow(["youtube_id"])
    escritor_csv.writerows(resultados_testing)

print("Procesamiento de videos de testing completado.")

# ----------------------------- MOSTRAR CARACTERÍSTICAS DE UN VIDEO AL AZAR -----------------------------

# Seleccionar un video al azar y mostrar sus características reducidas
random_index = random.randint(0, len(reduced_features) - 1)
selected_video_id = resultados_testing[random_index][0]
selected_features = reduced_features[random_index]

print(f"\nCaracterísticas reducidas para el video con ID '{selected_video_id}':\n{selected_features}")

import os # Librería para interactuar con el sistema operativo (manejo de archivos y directorios)
import pickle # Librería para serializar y deserializar objetos en Python
import face_recognition # Librería para reconocimiento facial basada en deep learning

# 1. Definir la ruta de la carpeta con las imágenes de los estudiantes
dataset_path = 'DataSet'

known_encodings = []
known_names = []

print("=== INICIANDO ENTRENAMIENTO DE IA (Red Neuronal ResNet) ===")

# 2. Recorrer cada archivo dentro de la carpeta DataSet
for file_name in os.listdir(dataset_path):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Ruta completa del archivo e identificación del estudiante por su nombre de archivo
        image_path = os.path.join(dataset_path, file_name)
        student_name = os.path.splitext(file_name)[0]

        try:
            # 3. Cargar la imagen usando face_recognition
            image = face_recognition.load_image_file(image_path)

            # 4. Extraer la marca vectorial (128-d embedding) del rostro
            encodings = face_recognition.face_encodings(image)

            # 5. Validar que se haya detectado un rostro en la imagen
            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(student_name)
                print(f"[SUCCESS] Rostro vectorizado correctamente: {student_name}")
            else:
                print(f"[WARNING] No se detectó ningún rostro en la imagen: {file_name}")

        except Exception as e:
            print(f"[ERROR] No se pudo procesar la imagen {file_name}: {e}")

# 6. Guardar los vectores y nombres en un archivo binario usando Pickle
data = {
    "encodings": known_encodings,
    "names": known_names
}

with open('modelo_ia.pkl', 'wb') as f:
    pickle.dump(data, f)

print("\n¡Entrenamiento completado exitosamente! El modelo fue guardado como 'modelo_ia.pkl'.")
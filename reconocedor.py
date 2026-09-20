import os
import pickle
import face_recognition

dataset_path = 'DataSet'
known_encodings = []
known_names = []

print("Procesando fotos del DataSet con la Red Neuronal (ResNet)...")

for file_name in os.listdir(dataset_path):
    if file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(dataset_path, file_name)
        student_name = os.path.splitext(file_name)[0]
        
        # Cargar la imagen del estudiante
        image = face_recognition.load_image_file(image_path)
        
        # Extraer las características vectoriales (embeddings)
        encodings = face_recognition.face_encodings(image)
        
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(student_name)
            print(f"Rostro vectorizado correctamente: {student_name}")
        else:
            print(f"Advertencia: No se detectó rostro en {file_name}")

# Guardar los vectores en el modelo binario de IA
data = {"encodings": known_encodings, "names": known_names}
with open('modelo_ia.pkl', 'wb') as f:
    pickle.dump(data, f)

print("\n¡Entrenamiento completado! Modelo guardado como 'modelo_ia.pkl'.")
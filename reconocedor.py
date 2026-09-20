import cv2
import os
import numpy as np

# 1. Configuración de rutas y clasificadores
dataset_path = 'DataSet'
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Crear el reconocedor facial LBPH
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

faces_data = []
labels = []
label_to_name = {}
current_id = 0

print("Cargando imágenes y procesando rostros...")

# 2. Leer las imágenes y extraer los rostros
for file_name in os.listdir(dataset_path):
    if file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(dataset_path, file_name)
        
        # Obtener el nombre del estudiante (ej: SANTOS_MARIANA)
        student_name = os.path.splitext(file_name)[0]
        
        # Cargar la imagen en escala de grises
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        # Detectar la cara en la foto
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Recortar el área del rostro
            face_roi = image[y:y+h, x:x+w]
            
            faces_data.append(face_roi)
            labels.append(current_id)
            label_to_name[current_id] = student_name
            current_id += 1
            print(f"Rostro procesado: {student_name}")

# 3. Entrenar el modelo con los rostros procesados
if len(faces_data) > 0:
    print("\nEntrenando el modelo de reconocimiento facial...")
    face_recognizer.train(faces_data, np.array(labels))
    
    # Guardar el modelo entrenado
    face_recognizer.save('modelo_lbph.xml')
    print("¡Entrenamiento completado con éxito! Modelo guardado como 'modelo_lbph.xml'.")
else:
    print("No se encontraron rostros válidos para entrenar.")
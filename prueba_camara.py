import cv2
import pickle
import face_recognition

# 1. Cargar el modelo entrenado con los vectores y nombres
print("Cargando modelo de Inteligencia Artificial...")
with open('modelo_ia.pkl', 'rb') as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

# 2. Inicializar la cámara web (0 es la cámara predeterminada)
cap = cv2.VideoCapture(0)
print("Cámara iniciada. Presiona la tecla 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al acceder a la cámara.")
        break

    # 3. Reducir el tamaño del cuadro a 1/4 para acelerar el procesamiento en tiempo real
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # 4. Convertir la imagen de formato BGR (OpenCV) a RGB (face_recognition)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # 5. Localizar rostros y extraer sus embeddings en el cuadro actual
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # 6. Comparar los rostros detectados contra el modelo guardado
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Calcular similitud (distancia euclidiana) contra todos los vectores conocidos
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Desconocido"

        # Si hay coincidencia, seleccionar la de menor distancia (mayor precisión)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_names[best_match_index]

        # 7. Escalamos las coordenadas x4 debido a la reducción previa del marco
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # 8. Dibujar el recuadro verde y el nombre en la pantalla
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    # 9. Mostrar la ventana de video en tiempo real
    cv2.imshow('Sistema de Reconocimiento Facial (IA 2)', frame)

    # Presionar 'q' para cerrar la ventana
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos de la cámara
cap.release()
cv2.destroyAllWindows()
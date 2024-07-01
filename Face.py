import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace

face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# Untuk menyimpan jumlah wajah yang terdeteksi
num_faces_detected = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5)

    num_faces = len(faces)
    num_faces_detected.append(num_faces)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_crop = frame[y:y + h, x:x + w]

        try:
            analysis = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion']
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception as e:
            print("Error analyzing face:", e)

    cv2.imshow('Face and Emotion', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Visualisasi jumlah data
plt.figure(figsize=(10, 5))
plt.plot(num_faces_detected, label='Jumlah Wajah Terdeteksi')
plt.xlabel('Frame')
plt.ylabel('Jumlah Wajah')
plt.title('Jumlah Wajah Terdeteksi dalam Setiap Frame')
plt.legend()
plt.show()

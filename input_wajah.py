import cv2
import pickle
import numpy as np
import os

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces_data = []
i = 0
name = input("Enter Your Name: ")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        resized_img = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
        
        if len(faces_data) <= 100 and i % 10 == 0:
            faces_data.append(resized_img)
        i += 1

        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or len(faces_data) == 100:
        break

video.release()
cv2.destroyAllWindows()

faces_data = np.asarray(faces_data).reshape(100, -1)

names_path = 'data/names.pkl'
if 'names.pkl' in os.listdir('data/'):
    with open(names_path, 'rb') as f:
        names = pickle.load(f)
else:
    names = []

names += [name] * (100 - len(names))
with open(names_path, 'wb') as f:
    pickle.dump(names, f)

faces_data_path = 'data/faces_data.pkl'
if 'faces_data.pkl' in os.listdir('data/'):
    with open(faces_data_path, 'rb') as f:
        faces = pickle.load(f)
else:
    faces = np.empty((0, faces_data.shape[1]), dtype=np.uint8)

faces = np.vstack((faces, faces_data))
with open(faces_data_path, 'wb') as f:
    pickle.dump(faces, f)

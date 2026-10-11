import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

cap = cv2.VideoCapture(0)
face_Cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

datasetPath = "./data/"
os.makedirs(datasetPath, exist_ok=True)

faceData = []
labels = []
class_id = 0
names = {}

for fx in os.listdir(datasetPath):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        dataItem = np.load(os.path.join(datasetPath, fx))
        faceData.append(dataItem)

        target = class_id * np.ones((dataItem.shape[0],), dtype=int)
        class_id += 1
        labels.append(target)

if len(faceData) == 0:
    print("\n[!] No face datasets (.npy) found in the './data/' folder.")
    print("    -> Run 'py getFaceData.py' to enroll face data, OR")
    print("    -> Use the web app ('py app.py') to enroll faces via webcam!\n")
    exit(0)

faceDataSet = np.concatenate(faceData, axis=0)
faceLabels = np.concatenate(labels, axis=0)
print(f"Loaded dataset: {faceDataSet.shape[0]} samples, {faceDataSet.shape[1]} features, {len(names)} classes.")

# Train Scikit-Learn KNeighborsClassifier model
k_val = min(5, len(faceLabels))
knn_model = KNeighborsClassifier(n_neighbors=k_val, metric="euclidean")
knn_model.fit(faceDataSet, faceLabels)
print(f"[OK] Scikit-Learn KNeighborsClassifier fitted (k={k_val}, metric='euclidean')")

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_Cascade.detectMultiScale(gray, 1.3, 5)

    for face in faces:
        x, y, w, h = face
        offset = 5
        faceSection = frame[max(0, y-offset):min(frame.shape[0], y+offset+h), max(0, x-offset):min(frame.shape[1], x+w+offset)]
        faceSection = cv2.resize(faceSection, (100, 100))

        # Scikit-Learn inference
        face_query = faceSection.flatten().reshape(1, -1)
        pred_label = knn_model.predict(face_query)[0]
        probs = knn_model.predict_proba(face_query)[0]
        conf_pct = probs[pred_label] * 100

        label_text = f"{names[int(pred_label)]} ({conf_pct:.0f}%)"
        cv2.putText(frame, label_text, (x, y-10), font, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Recognizer (Scikit-Learn KNN)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

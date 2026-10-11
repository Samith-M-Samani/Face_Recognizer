import cv2
import numpy as np
import os

cap  = cv2.VideoCapture(0)
face_Cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

skip = 0
faceData = []
datasetPath = "./data/"
os.makedirs(datasetPath, exist_ok=True)

fileName = input("Enter Your Name: ").strip()
if not fileName:
    fileName = "User"

samples_input = input("Enter number of samples to capture [25-200, default 100]: ").strip()
target_samples = int(samples_input) if samples_input.isdigit() else 100
target_samples = max(25, min(200, target_samples))

print(f"\n[*] Capturing {target_samples} face samples for '{fileName}'...")
print("    Tip: Slowly change your facial expressions (neutral, smile, tilt head, turn slightly)")
print("    Press 'q' at any time to finish early.\n")

while True:
    ret, frame = cap.read()
    
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret == False:
        continue

    faces = face_Cascade.detectMultiScale(grayFrame, 1.3, 5)
    if len(faces) == 0:
        continue
    
    k = 1

    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)

    skip += 1

    for face in faces[:1]:
        x, y, w, h = face

        offset = 5
        faceOffset = frame[max(0, y-offset):min(frame.shape[0], y+h+offset), max(0, x-offset):min(frame.shape[1], x+w+offset)]
        faceSelection = cv2.resize(faceOffset, (100, 100))

        if skip % 5 == 0:
            faceData.append(faceSelection)
            pct = int(len(faceData) / target_samples * 100)
            print(f"Captured: {len(faceData)} / {target_samples} ({pct}%)")
            if len(faceData) >= target_samples:
                print(f"[+] Successfully captured all {target_samples} samples!")
                break
        
        cv2.imshow(str(k), faceSelection)
        k += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if len(faceData) >= target_samples:
        break

    cv2.imshow("faces",frame)

    key_pressed = cv2.waitKey(1) & 0xFF

    if key_pressed == ord('q'):
        break

faceData = np.array(faceData)
faceData = faceData.reshape((faceData.shape[0],-1))
print(faceData.shape)
np.save(datasetPath + fileName, faceData)

cap.release()

cv2.destroyAllWindows()

import cv2
import numpy as np

cap  = cv2.VideoCapture(0)
face_Cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

skip = 0
faceData = []
datasetPath = "./faceRecorded/"

fileName = input("Enter Your Name: ")

while True:
    ret,frame = cap.read()
    
    grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    if ret == False:
        continue

    faces = face_Cascade.detectMultiScale(grayFrame,1.3,5)
    if len(faces) == 0:
        continue
    
    k = 1

    faces = sorted(faces, key= lambda x : x[2]*x[3], reverse=True)

    skip +=1

    for face in faces[:1]:
        x,y,w,h = face

        offset = 5
        faceOffset = frame[y-offset: y+h+offset,x-offset:x+w+offset]
        faceSelection = cv2.resize(faceOffset,(100,100))

        if skip % 10 == 0:
            faceData.append(faceSelection)
            print(len(faceData))
        
        cv2.imshow(str(k),faceSelection)
        k+=1

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

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
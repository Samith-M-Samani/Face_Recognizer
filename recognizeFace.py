import cv2
import os
import numpy as np

def distance(v1,v2):
    return np.sqrt(((v1-v2)**2).sum())

def knn(train,test,k=5):
    dist = []
    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test,ix)
        dist.append([d, iy])
    dk = sorted(dist, key=lambda x : x[0])[:k]
    labels = np.array(dk)[:,-1]
    output = np.unique(labels, return_counts=True)

    index = np.argmax(output[1])
    return output[0][index]

cap  = cv2.VideoCapture(0)
face_Cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

datasetPath = "./"

faceData = []
labels = []
class_id = 0
names = {}

for fx in os.listdir(datasetPath):
    if fx.endswith('.npy'):
        names[class_id] = fx[: -4]
        dataItem = np.load(datasetPath + fx)
        faceData.append(dataItem)

        target = class_id * np.ones((dataItem.shape[0],))
        class_id += 1
        labels.append(target)

faceDataSet = np.concatenate(faceData, axis=0)
faceLabels = np.concatenate(labels, axis=0).reshape((-1,1))
print(faceLabels.shape) 
print(faceDataSet.shape)

trainSet = np.concatenate((faceDataSet,faceLabels),axis=1)
print(trainSet.shape)

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret,frame = cap.read()
    if ret == False:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_Cascade.detectMultiScale(gray, 1.3, 5)

    for face in faces:
        x,y,w,h = face

        offset = 5
        faceSection = frame[y-offset:y+offset+h, x-offset:x+w+offset]
        faceSection = cv2.resize(faceSection,(100,100))

        out = knn(trainSet, faceSection.flatten())

        cv2.putText(frame, names[int(out)],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y), (x+w,y+h),(0,255,0),2)
    
    cv2.imshow("Face Recognizer",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()

cv2.destroyAllWindows()

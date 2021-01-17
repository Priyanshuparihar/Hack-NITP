import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = "images"
images = []
className = []
myList = os.listdir(path)
#print(myList)

for i in myList:
	curImage = cv2.imread(f'{path}/{i}')
	images.append(curImage)
	className.append(os.path.splitext(i)[0])
#print(className)

def findEncoding(images):
	endcodeList = []
	for i in images:
		i= cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
		endcode = face_recognition.face_encodings(i)[0]
		endcodeList.append(endcode)
	return endcodeList

def markAttendance(name):
	with open("Attendance.csv",'r+') as f:
		myDataList = f.readlines()
		nameList = []
		for line in myDataList:
			entry = line.split(',')
			nameList.append(entry[0])
		if name not in nameList:
			now = datetime.now()
			dtString = now.strftime('%H:%M:%S')
			f.writelines(f'\n{name},{dtString}')


encodeListKnown = findEncoding(images)
print("Encoding Complete")

cap = cv2.VideoCapture(0)


while True:
	success, img = cap.read()
	imgS = cv2.resize(img,(0,0),None,0.25,0.25)
	imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)


	facesCurFrame = face_recognition.face_locations(imgS)
	encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

	for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
		matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
		faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
		#print(faceDis)
		matchIndex = np.argmin(faceDis)

		if matches[matchIndex]:
			name = className[matchIndex].upper()
			#print(name)
			y1,x2,y2,x1 = faceLoc
			y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
			cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
			cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
			cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
			markAttendance(name)

	cv2.imshow("webcam",img)
	if cv2.waitKey(1) & 0xFF ==ord('q'):
		break



import cv2 
import face_recognition
import numpy as np
import os
from datetime import datetime

path = "Images"
images = []
class_names = []

for cl in os.listdir(path):
    cur = cv2.imread(f"{path}/{cl}")
    images.append(cur)
    class_names.append(os.path.splitext(cl)[0])

print(class_names)

def findEncodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list


def markAttendance(name):
    with open("Attendance.csv", "w+" ) as f:
        datalist = f.readlines()
        namelist = []
        for line in datalist:
            entry = line.split(",")
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            datestr = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name},{datestr}")


encode_know = findEncodings(images)
print("Encoding complete")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    success, img = cap.read()
    img_small = cv2.resize(img, (0,0), None, 0.25, 0.25)
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    face_cur_frame = face_recognition.face_locations(img_small)
    encode_cur_frame = face_recognition.face_encodings(img_small, face_cur_frame)
    
    for encode_face, face_loc in list(zip(encode_cur_frame, face_cur_frame)):
        matches = face_recognition.compare_faces(encode_know, encode_face)
        face_dis = face_recognition.face_distance(encode_know, encode_face)
        # print(face_dis)
        match_index = np.argmin(face_dis)

        if matches[match_index]:
            name = class_names[match_index]
            print(name)
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 0.62, (255,255,255), 2)
            markAttendance(name)


    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
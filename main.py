import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import mysql.connector
from PIL import Image, ImageDraw
import pandas as pd

nameList = []
path = './PhotosWebcam'
images = []
classNames = []

# Read images in the directory PhotosWebcam
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print("Input photos are: ", myList)
print("Trained photos are: ", classNames)

# Create database
mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="111",
        port='3306',
        db='attendance'
)

mycursor = mydb.cursor()
mycursor.execute("CREATE DATABASE mydatabase")
mycursor.execute("SHOW DATABASES")

def addToDatabase(name, time):
    sql = "INSERT INTO customers (name, time) VALUES (%s, %s)"
    val = [
        (name, time)
    ]

    mycursor.executemany(sql, val)

    mydb.commit()

    print(mycursor.rowcount, "was inserted.")

def face_square():
    """
    框选人脸部位
    """
    face_image = face_recognition.load_image_file('../origin_face/face2.jpg')
    face_location = face_recognition.face_locations(face_image, model='cnn')
    print(face_location)
    pil_image = Image.fromarray(face_image)
    pos = face_location[0]
    d = ImageDraw.Draw(pil_image, 'RGBA')
    d.rectangle((pos[3], pos[0], pos[1], pos[2]))
    pil_image.show()
    pil_image.save('result.jpg')

def face_lipstick():
    """
    上口红
    """
    face_image = face_recognition.load_image_file('../origin_face/face2.jpg')
    face_landmarks_list = face_recognition.face_landmarks(face_image)
    print(face_landmarks_list)
    for face_landmarks in face_landmarks_list:
        pil_image = Image.fromarray(face_image)
        d = ImageDraw.Draw(pil_image, 'RGBA')
        d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
        d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
        d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=3)
        d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=3)
        pil_image.show()
        pil_image.save('result.jpg')

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('./Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')
                # addToDatabase(name, dtString)

def DisplayCsv():
    df = pd.read_csv('Attendance.csv')
    print(df.info())

if __name__ == '__main__':

    encodeListKnown = findEncodings(images)
    print('Encoding Complete')

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        # img = captureScreen()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)

        cv2.imshow('Webcam', img)
        cv2.waitKey(1)

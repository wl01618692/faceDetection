import cv2
import face_recognition
import os
import sys

# Input: two images
# Reads an image file for training with SVM, another image for validation

# Training data
if __name__ == '__main__':
    if len(sys.argv) > 1:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
        imgElon = face_recognition.load_image_file(file1)
        imgTest = face_recognition.load_image_file(file2)
    else:
        imgElon = face_recognition.load_image_file('./PhotosBasic/CalvinTraining.jpg')
        imgTest = face_recognition.load_image_file('PhotosWebcam/Calvin.jpg')

    imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
    imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

    # Find face location of training data
    faceLoc = face_recognition.face_locations(imgElon)[0]
    encodeElon = face_recognition.face_encodings(imgElon)[0]

    # Mark face with rectangle
    cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)


    # Find face location of testing data
    faceLocTest = face_recognition.face_locations(imgTest)[0]
    encodeTest = face_recognition.face_encodings(imgTest)[0]

    # Mark face with rectangle
    cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

    # Compare faces, generate result
    results = face_recognition.compare_faces([encodeElon], encodeTest)
    faceDis = face_recognition.face_distance([encodeElon], encodeTest)

    # Output result on screen
    print(results, faceDis)
    cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    # Display
    cv2.imshow('Person 1 for training', imgElon)
    cv2.imshow('Person 2 for validation', imgTest)
    cv2.waitKey(0)


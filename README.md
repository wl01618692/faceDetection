# faceDetection
Face detection system with webcam. There are 2 important files: basic.py and main.py.
# Usage:
```bash
python basic.py
python main.py 
```
# Basic
basic.py reads 2 input images, uses one for training and second one for validation, and determines if the person in the second image is the same as the person in the first. Press any key to terminate the program.
![Alt text](./Doc/DemoImg1.jpg?raw=true)
![Alt text](./Doc/DemoImg2.jpg?raw=true)
![Alt text](./Doc/DemoImg3.png?raw=true)
# Webcam
main.py reads all images in the directory PhotosWebcam for training and reads real-time image on webcam for validation. If a face is recognized, the person 's name and current time will be recorded into Attendance.csv
![Alt text](./Doc/DEMOIMG5.png?raw=true)
![Alt text](./Doc/DemoImg6.jpg?raw=true)
![Alt text](./Doc/DemoImg7.png?raw=true)
![Alt text](./Doc/excel.png?raw=true)


# Face Detection with attendance system
This project supports a face detection system with a webcam. The source code can be found in two files: basic.py and main.py. All the photos used for training and validation can be found at ./PhotoBasics and ./PhotosWebcam. The output record file can be found at ./attendance.csv. This project requires you to install the following package: cv, face_recognition, mysql, pandas, numpy, and PIL.
# Usage:
```bash
python basic.py
python basic.py [image_path1 image_path2]
python main.py
```
# Basic
The first file (basic.py) reads two input file paths through command line arguments., uses the first one for training and second one for validation, and determines if the person in the second image is the same as the person in the first. After comparing, you can press any key to terminate the program. If there are no input file paths, the program will automatically use my image ./PhotosBasic/Calvin.jpg and ./PhotosBasic/Calvin2.jpg by default.
![Alt text](./Doc/Demoimg8.png?raw=true)
![Alt text](./Doc/DemoImg1.jpg?raw=true)
![Alt text](./Doc/DemoImg2.jpg?raw=true)
![Alt text](./Doc/DemoImg3.png?raw=true)
# Webcam
main.py reads all images in the directory PhotosWebcam for training and reads real-time input on webcam for validation. A Mysql database will be initialized at the start of the program. If a face is recognized, the person's name and current time will be recorded into Attendance.csv and the Mysql database. 
![Alt text](./Doc/DEMOIMG5.png?raw=true)
![Alt text](./Doc/DemoImg6.png?raw=true)
![Alt text](./Doc/Demoimg7.png?raw=true)
![Alt text](./Doc/gif.gif?raw=true)
![Alt text](./Doc/excel.png?raw=true)
# Package
Make sure that you install mysql-connector to run the main.py program. To install mysql-connector, you can use
```
python -m pip install mysql-connector
```
To install face_recognition_models, you can use
```
pip install face_recognition_models
```

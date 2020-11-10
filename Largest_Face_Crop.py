# Sertis Face Regconigtion Croping Coding Date 10/11/2020 By Panupong Suksuwan
"""
นำเข้า Lib ที่ใช้งาน 
- Numpy ไว้จัดการ Array จากการภาพเช่น Copy Image Array 
- cv2(OpenCV) Pre-trained Model สำหรับตวรจจับใบน้าโดยจะใช้ส่วนของ haarcascade_frontalcatface.xml 
- mathplotlib สำหรับการ plot กรอบใบหน้าบนภาพ
- อาจจะมีเพิ่มเติมเช่ม os Path File หรือ Flask
"""
from flask import Flask, request
from flask_cors import CORS,cross_origin
import numpy as np
import cv2
import matplotlib.pyplot as plt
face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')

# นำเข้ารูปสำหรับการทำ Face Regconigtion Croping
image = cv2.imread('download.jpg')

# สำรองภาพสีไว้สำหรับผลลัพธ์
raw = np.copy(image)

# แปลงภาพให้เป็นโทนสี RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# สำรองภาพที่เป็นโทนสี RGB
image_copy = np.copy(image)

# แปลงภาพให้เป็นโทนสี RGB เป็น Grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# ใช้ Pre-trained Model ในการ Dectect หน้า
faces = face_cascade.detectMultiScale(raw, 1.25, 6)

# แสดงค่าจำนวนหน้าที่ Pre-trained Model พบ บนหน้า Console
print('Number of faces detected:', len(faces))

# สร้างกรอบบนหน้าที่ Dectect เจอ
face_crop = []
for f in faces:
    x, y, w, h = [ v for v in f ]
    cv2.rectangle(image_copy, (x,y), (x+w, y+h), (255,0,0), 3)

    #  Crop ตามกรอบที่ Plot
    face_crop.append(raw[y:y+h, x:x+w])

for face in face_crop:
    cv2.imshow('face',face)
    cv2.waitKey(0)

# Display the face crops With Gui
fig = plt.figure(figsize = (9,9))
axl = fig.add_subplot(111)
axl.set_xticks([])
axl.set_yticks([])
axl.set_title("Largest Face Immage Cropped")

# จุด Return File
filename = 'Result.jpg'
cv2.imwrite(filename, face)

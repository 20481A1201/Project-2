import cv2,os
import numpy as np
import pandas as pd
import datetime
import time
import urllib.request


recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
recognizer.read("Trainner.yml")
harcascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(harcascadePath);    
df=pd.read_csv("StudentDetails\StudentDetails.csv")
cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX 
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)    
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])

        if(conf < 60):
                aa=df.loc[df['Id'] == Id]['Name'].values
                print(aa)
        else:
            aa='Unknown'
            
        cv2.putText(im,str(aa),(x,y+h), font, 1,(255,255,255),2)            
    cv2.imshow('im',im) 
    cv2.waitKey(1)

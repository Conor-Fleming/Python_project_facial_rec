from twilio.rest import Client
from gpiozero import MotionSensor
import time
import pyimgur
import re
from picamera import PiCamera
import cv2
import numpy as np
import os 
def facial_recognition():
    account_sid = 'AC7b551b2646dfa055d96447b743226f53'
    auth_token = '9a4addbdd1a81952eecc662a1c5fc7ea'

    client = Client(account_sid, auth_token)
    im = pyimgur.Imgur('d0561dc58f00c40')
    image_dir = '/home/pi/project/'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "opencv/data/haarcascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);
    font = cv2.FONT_HERSHEY_SIMPLEX
    #iniciate id counter
    id = 0
    # names related to ids: example ==> Marcelo: id=1,  etc
    names = ['None', 'Conor', 'Mike'] 
    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video widht
    cam.set(4, 480) # set video height
    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    #while True:
    ret, img =cam.read()
    img = cv2.flip(img, -1) # Flip vertically
    cv2.imwrite('project.jpg', img)
    print("photo captured")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    id = "intruder"            
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        # Check if confidence is less than 100 ==> "0" is perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "intruder"
            confidence = "  {0}%".format(round(100 - confidence))
            
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    upload_image = im.upload_image(image_dir + 'project.jpg')
    print("photo uploaded")
    client.messages.create(
        body='Sensor Triggered by ' + id + " on " + time.ctime(),
        from_='+15304085279',
        to='+15309251346',
        media_url = upload_image.link
    )
    print("photo sent")
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

pir = MotionSensor(4)
while True:
    if pir.motion_detected:
        print("Motion Detected")
        facial_recognition()
        break




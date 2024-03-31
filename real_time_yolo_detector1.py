import cv2
import numpy as np
import time
import telepot
import time
import urllib.request
import serial
import pandas as pd
def handle(msg):
  global telegramText
  global chat_id
  global receiveTelegramMessage
  
  chat_id = msg['chat']['id']
  telegramText = msg['text']
  
  print("Message received from " + str(chat_id))
  
  if telegramText == "/start":
    bot.sendMessage(chat_id, "Welcome to ROBOT Bot")
  
  else:
    buz.beep(0.1, 0.1, 1)
    receiveTelegramMessage = True
def capture():
    
    print("Sending photo to " + str(chat_id))
    bot.sendPhoto(chat_id, photo = open('./image.jpg', 'rb'))


bot = telepot.Bot('6818021150:AAEZUNwBEJ8doZ0RqS3Dwku2NNe4G9QxoOg')
chat_id='5042434018'
bot.message_loop(handle)

print("Telegram bot is ready")

bot.sendMessage(chat_id, 'BOT STARTED')
recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
recognizer.read("Trainner.yml")
harcascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(harcascadePath);    
df=pd.read_csv("StudentDetails\StudentDetails.csv")
cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

ser=serial.Serial('com3',baudrate=9600,timeout=0.1)
# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
cap = cv2.VideoCapture(0)


font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0


sts=0
aa=""
def send_sms():
    global aa
    print('Sending SMS to '+ str(aa))

    cmd='AT\r\n'
    ser.write(cmd.encode())
    time.sleep(2)
    rcv = ser.read(10)
    print(rcv)

    cmd='ATE0\r\n'
    ser.write(cmd.encode())
    time.sleep(2)
    rcv = ser.read(10)
    print(rcv)

    
    cmd='AT+CMGF=1\r\n'
    ser.write(cmd.encode())
    time.sleep(2)
    rcv = ser.read(10)
    print(rcv)
    phno="7386434488"  #  default
    if(aa=='yaseen'):
        phno="7386434488"
    if(aa=='yaseen'):
        phno="7386434488"
    print("PH:"+str(phno))
    cmd='AT+CMGS="'+str(phno)+'"\r\n'
    ser.write(cmd.encode())
    rcv = ser.read(20)
    print(rcv)                        
    time.sleep(1)
    cmd="Alert using mobile in campus"
    ser.write(cmd.encode())  # Message

    #ser.write(msg.encode())  # Message
    time.sleep(1)
    cmd = "\x1A"
    ser.write(cmd.encode()) # Enable to send SMS
    
    print('SMS Sent')
    time.sleep(6)


    cmd='AT\r\n'
    ser.write(cmd.encode())
    time.sleep(2)
    rcv = ser.read(20)
    print(rcv)
    cmd='AT+CMGF=1\r\n'
    ser.write(cmd.encode())
    time.sleep(2)
    rcv = ser.read(20)
    print(rcv)                                             
    phno="7386434488"    #hod number                      
    cmd='AT+CMGS="'+str(phno)+'"\r\n'
    ser.write(cmd.encode())
    rcv = ser.read(20)
    print(rcv)                        
    time.sleep(1)
    cmd="Alert using mobile in campus " + str(aa)
    print(cmd)
    ser.write(cmd.encode())  # Message

    #ser.write(msg.encode())  # Message
    time.sleep(1)
    cmd = "\x1A"
    ser.write(cmd.encode()) # Enable to send SMS
    time.sleep(10)
    print('SMS Sent')
    time.sleep(1)

    
    time.sleep(5)
sts=0
while True:
    pflag=0
    cflag=0
    _, frame = cap.read()
    frame_id += 1


    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    aa="face not recognized"
    for(x,y,w,h) in faces:
        pflag=pflag+1
        cv2.rectangle(frame,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])

        if(conf < 60):
                aa=df.loc[df['Id'] == Id]['Name'].values
                print(aa)
        else:
            aa='Unknown'
            
        cv2.putText(frame,str(aa),(x,y+h), font, 1,(255,255,255),2)


    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[3] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 1.8)
                y = int(center_y - h / 1.8)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            if(classes[class_ids[i]]=='person'):
                pflag=pflag+1
            if(classes[class_ids[i]]=='cell phone'):
                cflag=1
            

            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 2, color, 2)





    if(pflag>0 and cflag>0):
        sts=sts+1
        fps_label = "ALERT"
        cv2.putText(
            frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
        )
        cv2.imwrite('image.jpg',frame)
        cv2.waitKey(1)


    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    if(sts>10):
        sts=0
        capture()
        send_sms()

cap.release()
cv2.destroyAllWindows()

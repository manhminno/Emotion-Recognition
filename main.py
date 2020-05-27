import numpy as np
import cv2
import argparse
from PIL import Image

from model import *

def putEmotion(img, emotion, x, y, w, h):
    #Each emotion have a color
    if(emotion == 'Vui ve'):
        cv2.putText(img, emotion, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.rectangle(img, (x, y-5), (x+w, y+h), (0,255,0), 2)
    elif(emotion == 'Gian giu'):
        cv2.putText(img, emotion, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.rectangle(img, (x, y-5), (x+w, y+h), (0,0,255), 2)
    elif(emotion == 'Binh thuong'):
        cv2.putText(img, emotion, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        cv2.rectangle(img, (x, y-5), (x+w, y+h), (255,0,0), 2)
    elif(emotion == 'Buon chan'):
        cv2.putText(img, emotion, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        cv2.rectangle(img, (x, y-5), (x+w, y+h), (0,0,0), 2)
    elif(emotion == 'Wow'):
        cv2.putText(img, emotion, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
        cv2.rectangle(img, (x, y-5), (x+w, y+h), (255,255,0), 2)

def predict(img_detect, model):
    img_detect = cv2.resize(img_detect, (32, 32)) #Resize 32x32
    img = Image.fromarray(img_detect)     
    
    img = transform_val(img)    
    img = img.view(1, 3, 32, 32) #View in tensor
    img = Variable(img)      
    
    model.eval() #Set eval mode

    #To Cuda
    model = model.cuda()
    img = img.cuda()

    output = model(img)
    
    predicted = torch.argmax(output)
    p = label2id[predicted.item()]

    return  predicted

if __name__ == "__main__":
    #Define parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="Img or Video")
    parser.add_argument("--path", help="Link direct")
    opt = parser.parse_args()
    
    #Load model
    model = CNN()
    model = model.cuda()
    model.load_state_dict(torch.load('weights/Emotion-Detection.pth'))

    if(opt.mode == "Image"):
        #Load haarlike feature
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades  + 'haarcascade_frontalface_default.xml')

        #Detect face
        img = cv2.imread(opt.path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
            img2 = img[y+2:y+h-2, x+2:x+w-2]
            emo = predict(img2, model)  #Emotion index 
            emotion = label2id[emo.item()]
            putEmotion(img, emotion, x, y, w, h)

        cv2.imshow('img',img)
        cv2.imwrite("Result.jpg", img)
        k = cv2.waitKey() & 0xff
        if k == ord('q'):
            cv2.destroyAllWindows()

    elif(opt.mode == "Webcam"):
        #Load haarlike feature
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades  + 'haarcascade_frontalface_default.xml')


        list_person = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        emotion = ["Binh thuong", "Binh thuong", "Binh thuong", "Binh thuong", "Binh thuong", "Binh thuong","Binh thuong"]
        
        #Load webcam
        cap = cv2.VideoCapture(0)

        while 1:
            ret, img = cap.read()

            #Detect faces
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            count = 0
            for (x,y,w,h) in faces:
                img2 = img[y+2:y+h-2, x+2:x+w-2]

                #Add emotion for each person
                list_person[count][predict(img2, model)] += 1
                
                #Each 18 frames show emotion
                if(sum(list_person[count]) == 18):
                    #Change emotion
                    emo = list_person[count].index(max(list_person[count]))
                    emotion[count] = label2id[emo]
                    
                    #Refresh emotion after 18 frames
                    list_person[count] = [0, 0, 0, 0, 0]

                #Put emotion and next face
                putEmotion(img, emotion[count], x, y, w, h)
                count += 1

            #Show
            cv2.imshow('Webcam',img)
            k = cv2.waitKey(24) & 0xff #24fps
            if k == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    else:
        print("Add parser: --mode (Image/Webcam) --path (link_Img)")
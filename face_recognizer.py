import csv
import os
from tkinter import *

import cv2
from pandas import np
from pandas.core.dtypes.inference import is_number

window = Tk()
window.title("Face Tracking System")
window.geometry("400x200")



def TakeImages():
    userid = int(idText.get())
    username = (nameText.get())


    cam = cv2.VideoCapture(0)
    harcascade = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascade)
    sampleNum = 0
    while(TRUE):
        ret,img = cam.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            sampleNum+=1
            cv2.imwrite("Training images/"+username+"."+str(userid)+"."+str(sampleNum)+".jpg", gray[y:y+h,x:x+w])
            cv2.imshow("frame",img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        elif sampleNum > 60:
            break
    cam.release()
    cv2.destroyAllWindows()
    res = "Images Saved for ID : "+ userid +" Name : "+ username
    row = [id,username]
    with open('StudentDetails/StudentDetails.csv','a+') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()
    notf.configure(text=res)


def TrainingImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascade = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascade)
    faces,Id = getImagesAndLabels("Trainner.yml")
    recognizer.train(faces,np.array(Id))
    recognizer.save("TrainingImageLabel/Trainner.yml")
    res = "Image Trained"#+",".join(str(f) for f in Id)
    notf.configure(text=res)


def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage,'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces,Ids



idLabel  = Label(window,text = "Enter ID")
idLabel.grid(row=0,column=0)
idText = Entry(window,width="8")
idText.grid(row=0,column=1)
nameLabel = Label(window,text="Enter the name")
nameText = Entry(window,width="15")
nameLabel.grid(row=1,column=0)
nameText.grid(row=1,column=1)


notfLabel  = Label(window,text = "Notifications : ")
notfLabel.grid(row=3,column=0)
notf = Label(window,text="")
notf.grid(row=3,column=1)


takeImgs = Button(window,text="Take Images",command=TakeImages)
takeImgs.grid(row=0,column=9)
trainImgs = Button(window,text="Train Images",command=TrainingImages)
trainImgs.grid(row=1,column=9)
trackImgs = Button(window,text="Track Images",command="")
trackImgs.grid(row=2,column=9)


detectLabel  = Label(window,text = "Detecting person : ")
detectLabel.grid(row=5,column=0)
detect = Label(window,text="")
detect.grid(row=5,column=1)

window.mainloop()

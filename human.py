from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np
import cv2
import time
import keras
import mediapipe as mp

from keras.utils.np_utils import to_categorical
from keras.layers import Input
from keras.models import Model
from keras.layers import MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
import keras

from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import pickle
import matplotlib.pyplot as plt
import os
from keras.models import model_from_json

global filename
global classifier

main = tkinter.Tk()
main.title("Gait Classification")
main.geometry("1200x1200")

global filename
proto_File = "Models/pose_deploy_linevec.prototxt"
weights_File = "Models/pose_iter_440000.caffemodel"
n_Points = 18
POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]
in_Width = 368
in_Height = 368
threshold = 0.1
global net
POSE_NAMES = ["Head", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
              "RAnkle", "LHip", "LKnee", "LAnkle", "Chest", "Background"]

def uploadVideo():
    global filename
    global net
    filename = filedialog.askopenfilename(initialdir="inputVideos")
    net = cv2.dnn.readNetFromCaffe(proto_File, weights_File)
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
#global accuracy


def gaitClassification():
    global filename
    global net
    cap = cv2.VideoCapture(filename)
    has_Frame, frame = cap.read()
    video_writer = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

    while cv2.waitKey(1) < 0:
        t = time.time()
        has_Frame, frame = cap.read()
        frameCopy = np.copy(frame)
        if not has_Frame:
            cv2.waitKey()
            break
        frame_Width = frame.shape[1]
        frame_Height = frame.shape[0]
        inp_Blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (in_Width, in_Height), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inp_Blob)
        output = net.forward()
        H = output.shape[2]
        W = output.shape[3]
        points = []
        for i in range(n_Points):
            probMap = output[0, i, :, :]
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            x = (frame_Width * point[0]*1.5) / W
            y = (frame_Height * point[1]*1.1) / H 
            if prob > threshold : 
                cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                points.append((int(x), int(y)))
            else :
                points.append(None)
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]
            print(str(pair[0])+" "+str(pair[1])+" "+str(partA)+" "+str(partB))
            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
                cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                if partA >= 8 and partA < len(POSE_NAMES)-1:
                    #cv2.putText(frame, POSE_NAMES[partA], (50, 100),  cv2.FONT_HERSHEY_SIMPLEX,0.8, (0, 255, 255), 2)
                    text.insert(END,POSE_NAMES[partA]+"\n")
                    text.update_idletasks()
        cv2.putText(frame, "time taken = {:.2f} sec --Gait is classified".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.imshow('Output-Skeleton', frame)
        video_writer.write(frame)                
    video_writer.release()

def MediaPipe():
    cap = cv2.VideoCapture(filename)
    cap.set(50,1500)
    #cap.set(4,720)
    #cap.set(10,150)
   # out = cv2.VideoWriter('output1.avi', fourcc, 20.0, (640, 480))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output1.avi',fourcc,20.0,(640,480))

    pTime = 0
    mpPose = mp.solutions.pose
    mpDraw = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    pose = mpPose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5

    )
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
            

        frame = cv2.resize(frame, (700, 1000))
        
        frame.flags.writeable = False
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        # print(results.pose_landmarks)

        frame.flags.writeable = True
        main = results.pose_landmarks
        if results.pose_landmarks :
            # print(results.pose_landmarks)
            for landmark in main.landmark:
                landmark.x+= 0.4
                
                # landmark.y+=0.5
            mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS,  connection_drawing_spec= mpDraw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2))
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, "Frames per second: {:.2f} sec --using Mediapipe".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        #cv2.putText(frame, "time taken = {:.2f} sec --Gait is classified".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 255, 255), 2, lineType=cv2.LINE_AA)

        # frame.translateXY(0, 0)
        # cv2.flip(frame, 1)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()





def Livedemo():
        cap = cv2.VideoCapture(0)
        cap.set(50,1500)
        #cap.set(4,720)
        #cap.set(10,150)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output2.avi', fourcc, 20.0, (640, 480))
        pTime = 0
        mpPose = mp.solutions.pose
        mpDraw = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        pose = mpPose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5

        )
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
                

            frame = cv2.resize(frame, (700, 1000))
            
            frame.flags.writeable = False
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            # print(results.pose_landmarks)

            frame.flags.writeable = True
            main = results.pose_landmarks
            if results.pose_landmarks :
                # print(results.pose_landmarks)
                for landmark in main.landmark:
                    landmark.x+= 0.4
                    
                    # landmark.y+=0.5
                mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS,  connection_drawing_spec= mpDraw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2))
            
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            # cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # frame.translateXY(0, 0)
            # cv2.flip(frame, 1)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()


def close():
    main.destroy()

    

font = ('times', 14, 'bold')
title = Label(main, text='Human Action Imitation using Gait Classifier')
title.config(bg='DarkGoldenrod1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Video File", command=uploadVideo)
uploadButton.place(x=100,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=400,y=100)

processButton = Button(main, text="Start Gait Phase Classification", command=gaitClassification)
processButton.place(x=100,y=175)
processButton.config(font=font1)

processButton = Button(main, text="Using Mediapipe", command=MediaPipe)
processButton.place(x=100,y=250)
processButton.config(font=font1)

processButton = Button(main, text="Live demo", command=Livedemo)
processButton.place(x=100,y=325)
processButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=100,y=400)
exitButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=110)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400,y=150)
text.config(font=font1)


main.config(bg='LightSteelBlue1')
main.mainloop()

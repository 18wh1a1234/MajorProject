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
global accuracy
def generateModel():
  global classifier
  global accuracy
  text.delete('1.0', END)
  text.insert(END,"Genetating model...")
  if os.path.exists('model/model.json'):
      with open('model/model.json', "r") as json_file:
          loaded_model_json = json_file.read()
          classifier = model_from_json(loaded_model_json)
      classifier.load_weights("model/model_weights.h5")
      classifier._make_predict_function()   
      print(classifier.summary())
      f = open('model/history.pckl', 'rb')
      data = pickle.load(f)
      f.close()
      acc = data['accuracy']
      accuracy = acc[9] * 100
      text.insert(END,"CNN Training Model Accuracy = "+str(accuracy)+"\n")
  else:
      classifier = Sequential()
      classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 1), activation = 'relu'))
      classifier.add(MaxPooling2D(pool_size = (2, 2)))
      classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
      classifier.add(MaxPooling2D(pool_size = (2, 2)))
      classifier.add(Flatten())
      classifier.add(Dense(output_dim = 256, activation = 'relu'))
      classifier.add(Dense(output_dim = 1, activation = 'softmax'))
      print(classifier.summary())
      classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
      hist = classifier.fit(X_train, Y_train, batch_size=16, epochs=10, shuffle=True, verbose=2)
      classifier.save_weights('model/model_weights.h5')            
      model_json = classifier.to_json()
      with open("model/model.json", "w") as json_file:
          json_file.write(model_json)
      f = open('model/history.pckl', 'wb')
      pickle.dump(hist.history, f)
      f.close()
      f = open('model/history.pckl', 'rb')
      data = pickle.load(f)
      f.close()
      acc = data['accuracy']
      accuracy = acc[9] * 100
      text.insert(END,"CNN Training Model Accuracy = "+str(accuracy)+"\n")

  
def VGG16():
    global VGG16_accuracy,VGG16_accuracy,VGG16_loss_val,vgg_acc,VGG16_loss
    model = Sequential()
    X_train = np.load('model/model1/features.txt.npy')
    Y_train = np.load('model/model1/labels.txt.npy')
    input_shape = X_train[0].shape
    model.add(Dense(256, activation='relu',input_shape = input_shape))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['acc'])  #optimizers.RMSprop(lr=le-4)
    history1 = model.fit(X_train, Y_train,epochs=100,batch_size=256)
    vgg_acc = history1.history['acc']
    VGG16_loss = history1.history['loss']
    VGG16_loss_val=VGG16_loss[29]*100
    VGG16_accuracy = vgg_acc[29]*100
    text.insert(END,"VGG16 Training Model Accuracy = "+str(VGG16_accuracy)+"\n")

def Gaitclassification():
    cap = cv2.VideoCapture(filename)
    cap.set(50,1500)
    #cap.set(4,720)
    #cap.set(10,150)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
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
    cv2.destroyAllWindowsS()





def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()

    accuracy = data['accuracy']
    loss = data['loss']

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', color = 'red')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.title('CNN Accuracy & Loss')
    plt.show()


global VGG16_accuracy,accuracy
def compare():
    global accuracy
    labels =['CNN accuracy','VGG16 accuracy']
    y_values= [accuracy,VGG16_accuracy]
    y_pos=np.arange(len(labels))
    plt.xlabel("Algorithm")
    plt.ylabel("Accuracy")
    plt.title("Comparision Graph")
    plt.bar(y_pos, y_values,color='yellow')
    plt.xticks(y_pos,labels)
    plt.show()




"""def Livedemo():
 #   cap = cv2.VideoCapture(0)
 #  cap.set(50,1500)
    #cap.set(4,720)
    #cap.set(10,150)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
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
    cv2.destroyAllWindowsS()"""


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
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=400,y=100)

cnnButton = Button(main, text="Generate CNN  Model", command=generateModel)
cnnButton.place(x=50,y=150)
cnnButton.config(font=font1)

vggButton = Button(main, text="VGG16 Algorithm", command=VGG16)
vggButton.place(x=50,y=200)
vggButton.config(font=font1) 

processButton = Button(main, text="Start Gait Phase Classification", command=Gaitclassification)
processButton.place(x=50,y=250)
processButton.config(font=font1)

graphButton = Button(main, text="Accuracy & Loss Graph", command=graph)
graphButton.place(x=50,y=300)
graphButton.config(font=font1)

comparegraph = Button(main, text="Compare Graph", command=compare)
comparegraph.place(x=50,y=350)
comparegraph.config(font=font1)

#comparegraph = Button(main, text="Live demo", command=Livedemo)
#comparegraph.place(x=50,y=400)
#comparegraph.config(font=font1)


exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=50,y=400)
exitButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=110)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400,y=150)
text.config(font=font1)


main.config(bg='LightSteelBlue1')
main.mainloop()

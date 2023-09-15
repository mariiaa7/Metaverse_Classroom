import cv2
import sys
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
#import imutils
import matplotlib.pyplot as plt
import matplotlib.image as plt_image
from keras.models import model_from_json
from keras.preprocessing import image
import socket
from fastapi import FastAPI

class MySequentialModel:

  def __init__(self):
    self.model = Sequential()
    self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Dropout(0.25))
    self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Dropout(0.25))
    self.model.add(Flatten())
    self.model.add(Dense(1024, activation='relu'))
    self.model.add(Dropout(0.5))
    self.model.add(Dense(7, activation='softmax'))
    cv2.ocl.setUseOpenCL(False)
    self.model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

  def loadWeights(self):
    # load json and create model
    emotion_model_json_file = open('Assets\Scripts\emotion_model.json', 'r')
    json_file = emotion_model_json_file.read()
    emotion_model_json_file.close()
    emotion_model = model_from_json(json_file)
    # load weights into new model
    self.model.load_weights("Assets\Scripts\emotion_model.h5")
    print("Loaded model from disk")


myModel = MySequentialModel()
myModel.loadWeights()

def execute():
  casc = sys.argv[0]
  faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
  emotion_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
  cap = cv2.VideoCapture(0)
  while True:
      # Find haar cascade to draw bounding box around face
      val, frame = cap.read()
      frame = cv2.resize(frame, (1280, 720))
      if not val:
          break
      detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      # detect faces available on camera
      faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
      # take each face available on the camera and Preprocess it
      for (x, y, w, h) in faces:
          cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
          gray_frame = gray[y:y + h, x:x + w]
          cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray_frame, (48, 48)), -1), 0)
          # predict the emotions
          prediction = myModel.model.predict(cropped_img)
          max_index = int(np.argmax(prediction))
          yield max_index

          cv2.putText(frame, emotion_dict[max_index], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
      cv2.imshow('Emotion Detection', frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  cap.release()
  cv2.destroyAllWindows()

generator = execute()
app = FastAPI()

@app.get("/")
def read_root():
  try:
    return next(generator)
  except:
    return -1 #esecuzione terminata
#! /usr/bin/env python3

# import copy
# import cv2
# import numpy as np
# from keras.models import load_model
# import time
# from keras.preprocessing.image import ImageDataGenerator, load_img
# General Settings


from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import threading
from matplotlib.pyplot import imshow
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras import backend as K



classes = {0: 'closed',
                 1: 'left',
                 2: 'open',
                 3: 'right'}


def load_model():
    try:
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("weights.hdf5")
        print("Model successfully loaded from disk.")
        
        #compile again
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        return model
    except:
        print("""Model not found. Please train the CNN by running the script 
cnn_train.py. Note that the training and test samples should be properly 
set up in the dataset directory.""")
        return None

model = load_model()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    #print(frame)
    if not ret:
        break
    frame=cv2.flip(frame,1)
    cv2.rectangle(frame,(300,200),(500,400),(0,255,0),1)
    cv2.putText(frame,"Place your hand in the green box.", (50,50), cv2.FONT_HERSHEY_PLAIN , 1, 255)
    cv2.putText(frame,"Press esc to exit.", (50,100), cv2.FONT_HERSHEY_PLAIN , 1, 255)
    
    cv2.imshow("preview", frame)
    frame=frame[200:400,300:500]
    frame = cv2.resize(frame, (200,200))
    frame = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY)
    frame=frame.reshape((1,)+frame.shape)
    frame=frame.reshape(frame.shape+(1,))
    test_datagen = ImageDataGenerator(rescale=1./255)
    m=test_datagen.flow(frame,batch_size=1)
    y_pred=model.predict_generator(m,1)
    histarray2={'CLOSED': y_pred[0][0], 'LEFT': y_pred[0][1], 'OPEN': y_pred[0][2], 'RIGHT': y_pred[0][3]}
    # update(histarray2)
    print(classes[list(y_pred[0]).index(y_pred[0].max())])
    # print(y_pred[0][0])
    ret, frame = cap.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
    cv2.destroyWindow("preview")
    vc=None

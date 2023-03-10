# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 23:36:41 2018

@author: Abhimanyu
"""
# Import OpenCV2 for image processing
# Import os for file path
import cv2, os

# Import numpy for matrix calculation
import numpy as np

# Import Python Image Library (PIL)
from PIL import Image


def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.FisherFaceRecognizer_create()

# Using prebuilt frontal face training model, for face detection
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# Create method to get the images and label data
def getImagesAndLabels(path):

    # Get all file path
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
    
    # Initialize empty face sample
    faceSamples=[]
    
    # Initialize empty id
    ids = []

    # Loop all the file path
    for imagePath in imagePaths:
        #print(imagePath)
        # Get the image and convert it to grayscale
        PIL_img = Image.open(imagePath).convert('L')
        #PIL_img = cv2.resize(PIL_img, (350, 350))

        # PIL image to numpy array
        img_numpy = np.array(PIL_img,'uint8')

        # Get the image id
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        #print(id)
        # Get the face from the training images
        faces = detector.detectMultiScale(img_numpy)
        

        # Loop for each face, append to their respective ID
        for (x,y,w,h) in faces:
            
            img=img_numpy[y:y+h,x:x+w]
            img=cv2.resize(img,(350,350))

            # Add the image to face samples
            faceSamples.append(img)

            # Add the ID to IDs
            ids.append(id)

    # Pass the face array and IDs array
    return faceSamples,ids

# Get the faces and IDs
faces,ids = getImagesAndLabels('dataset2')

#recognizer.read('trainer/trainer.yml')


# Train the model using the faces and IDs
recognizer.train(faces, np.array(ids))

# Save the model into trainer.yml
assure_path_exists('trainer/')
recognizer.save('trainer/trainer2.xml')

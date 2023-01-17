# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 19:25:53 2018

@author: Abhimanyu
"""


# Import OpenCV2 for image processing
import cv2
import os

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Start capturing video 
vid_cam = cv2.VideoCapture(0)

# Detect object in video stream using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, one face id
face_id = 21

# Initialize sample face image
count = 1

assure_path_exists("dataset2/")

# Start looping
while(True):

    # Capture video frame
    _, image_frame = vid_cam.read()
    

    # Convert frame to grayscale
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # Detect frames of different sizes, list of faces rectangles
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

    # Loops for each faces
    for (x,y,w,h) in faces:

        # Crop the image frame into rectangle
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
        gray = gray[y:y+h, x:x+w]
        out = cv2.resize(gray, (350, 350))


        # Save the captured image into the datasets folder
        for i in range(1,11):
            cv2.imwrite("dataset2/User." + str(face_id) + '.' + str(count*100+i) + ".jpg", out)
        
                
        # Increment sample face image
        count += 1
        
        # Display the video frame, with bounded rectangle on the person's face
        cv2.imshow('frame', image_frame)

    # To stop taking video, press 'q' for at least 100ms
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # If image taken reach 100, stop taking video
    elif count>100:
        break

# Stop video
vid_cam.release()

# Close all started windows
cv2.destroyAllWindows()

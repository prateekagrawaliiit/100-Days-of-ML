# -*- coding: utf-8 -*-
# @Author: prateek
# @Date:   2020-08-20 23:30:41
# @Last Modified by:   prateek
# @Last Modified time: 2020-08-20 23:35:40


#DEPENDENCIES

import face_recognition
import os 
import cv2

# CONSTANTS

KNOWN_FACES_DIR = 'known_faces'
UNKNOWN_FACES_DIR = 'unknown_faces'
# Tolerance : lower the tolerance lesser is the chance of false positives Default 0.6
TOLERANCE = 0.6
FRAME_THICKNESS = 2
FONT_THICKNESS = 1
MODEL = 'cnn'

print('Starting to Load the Known Faces')
	
known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
	for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
		image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
		encoding = face_recognition.face_encodings(image)[0]
		known_faces.append(encoding)
		known_names.append(name)

print('Processing the unknown images')

for filename in os.listdir(UNKNOWN_FACES_DIR):
	print(filename)
	image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
	loc = face_recognition.face_locations(image,model=MODEL)
	encoding = face_recognition.face_encodings(image,loc)
	
	image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
	
	for face_encoding,face_location in zip(encoding,loc):
		results = face_recognition.compare_faces(known_faces,face_encoding,tolerance=TOLERANCE)
		match = None
		if True in results:
			match = known_names[results.index(True)]
			print(f"Match FOUND : {match}")
			
			
			top_left = (face_location[3],face_location[0])
			bottom_right = (face_location[1],face_location[2])
			
			color = [255,0,0]
			
			cv2.rectangle(image,top_left,bottom_right,color,FRAME_THICKNESS)
			
			top_left = (face_location[3],face_location[2])
			bottom_right = (face_location[1],face_location[2]+22)
			
			cv2.rectangle(image,top_left,bottom_right,color,cv2.FILLED)
			cv2.putText(image,match,(face_location[3]+10,face_location[2]+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),FONT_THICKNESS)
			
	cv2.imshow(filename,image)
	cv2.waitKey(10000)

			
			
	
	

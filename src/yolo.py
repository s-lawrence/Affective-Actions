#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 13:56:12 2021

@author: s7lawren
"""
import cv2
import numpy as np
import os

class Yolo():
    def __init__(self, yoloPath, confidence_bound=0.25, threshold=0.3):
        self.labels           = self.get_yolo_labels(yoloPath)
        self.colors           = self.get_yolo_colors(self.labels) 
        self.net              = self.load_yolo_net(yoloPath)
        self.output_layers    = self.get_output_layers(self.net)
        self.confidence_bound = confidence_bound
        self.threshold        = threshold
    
    def get_yolo_labels(self, yolo):
        labelsPath = os.path.sep.join([yolo, "data/coco.names"])
        labels     = open(labelsPath).read().strip().split("\n")
        return labels

    def get_yolo_colors(self, labels):
        np.random.seed(42)
        return np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
    
    def load_yolo_net(self, yolo):
        weightsPath = os.path.sep.join([yolo, "yolov3.weights"])
        configPath  = os.path.sep.join([yolo, "cfg/yolov3.cfg"])
        
        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        return net
    
    def get_output_layers(self, net):
        # determine only the *output* layer names that we need from YOLO
        layers        = net.getLayerNames()
        output_layers = [layers[i - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers
    
    def get_blob_from_image(self, image):
        return cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    
    def get_object_data(self, layerOutputs, image_width, image_height):
        boxes = []
        confidences = []
        classIDs = []
        # loop over each of the layer outputs
        for output in layerOutputs:
        	# loop over each of the detections
        	for detection in output:
        		# extract the class ID and confidence (i.e., probability) of
        		# the current object detection
        		scores = detection[5:]
        		classID = np.argmax(scores)
        		confidence = scores[classID]
        
        		# filter out weak predictions by ensuring the detected
        		# probability is greater than the minimum probability
        		if confidence > self.confidence_bound:
        			# scale the bounding box coordinates back relative to the
        			# size of the image, keeping in mind that YOLO actually
        			# returns the center (x, y)-coordinates of the bounding
        			# box followed by the boxes' width and height
        			box = detection[0:4] * np.array([image_width, image_height, image_width, image_height])
        			(centerX, centerY, width, height) = box.astype("int")
        
        			# use the center (x, y)-coordinates to derive the top and
        			# and left corner of the bounding box
        			x = int(centerX - (width / 2))
        			y = int(centerY - (height / 2))
        
        			# update our list of bounding box coordinates, confidences,
        			# and class IDs
        			boxes.append([x, y, int(width), int(height)])
        			confidences.append(float(confidence))
        			classIDs.append(classID)
        return boxes, confidences, classIDs
    
    def parse_object_data(self, boxes, confidences, classIDs):
        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_bound, self.threshold)
        
        # ensure at least one detection exists
        if len(idxs) > 0:
            boxes_list, confidences_list, classID_list = [], [], []
        	# loop over the indexes we are keeping
            for i in idxs.flatten():
                boxes_list.append(boxes[i])
                confidences_list.append(confidences[i])
                classID_list.append(classIDs[i])
            
            return boxes_list, confidences_list, classID_list
        
    def get_yolo_image(self, image, boxes, confidences, classIDs):
       	for i, box in enumerate(boxes):
       		# extract the bounding box coordinates
       		(x, y) = (box[0], box[1])
       		(w, h) = (box[2], box[3])
       
       		# draw a bounding box rectangle and label on the image
       		color = [int(c) for c in self.colors[classIDs[i]]]
       		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
       		text = "{}: {:.4f}".format(self.labels[classIDs[i]], confidences[i])
       		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
       			0.5, color, 2)
        return image
            
            
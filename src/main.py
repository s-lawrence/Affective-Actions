#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 11:09:58 2021

@author: s7lawren
"""
import matplotlib.pylab as plt
import numpy as np
import time
from fuzzy_system import FuzzySystem
from yolo import Yolo
from iCub import ICub
from utils import args

if __name__ == "__main__":
    
    yolo = Yolo(args.yolo)
    
    icub = ICub()
    icub.intialize_yarp()
    icub.connect_image_ports(args.right_view_input_port, args.right_view_output_port,
                            args.left_view_input_port, args.left_view_output_port)
    icub.open_stress_port(args.emotion_port)
    icub.open_object_port(args.object_port)
    
    clutter, uncertainty = 0, 0
    try:
        while True:
            right_image = icub.get_right_image()
            left_image  = icub.get_left_image()
            H,W,D = right_image.shape
            
            images = [left_image, right_image]
            
            object_centers = []
            
            for i, image in enumerate(images):
                # construct a blob from the input image and then perform a forward
                blob         = yolo.get_blob_from_image(image)
                yolo.net.setInput(blob)
                layerOutputs = yolo.net.forward(yolo.output_layers)
                
                # pass of the YOLO object detector, giving us our bounding boxes and
                # associated probabilities
                boxes, confidences, classIDs, centers = yolo.get_object_data(layerOutputs, image_width=W, image_height=H)
                boxes, confidences, classIDs, centers = yolo.parse_object_data(boxes, confidences, classIDs, centers)
                object_centers.append(centers)
                
                # display the image that has been read
                image = yolo.get_yolo_image(image, boxes, confidences, classIDs)
                if i == 1: 
                    icub.publish_object_image(image)
                # plt.imshow(image)
                # plt.show()

            clutter = len(classIDs)
            uncertainty = 1 - np.round(np.mean(confidences), 2)
            
            fuzzy = FuzzySystem()
            fuzzy.interpret(clutter, uncertainty)
            # fuzzy.plot_membersip_functions()
            # fuzzy.plot_interpretation()
            # plt.show()
            
            icub.publish_affect_level(fuzzy.emotion)
            icub.publish_objects(object_centers)
            # publish clutter, centers
            print(clutter)
            print(fuzzy.emotion)
            
            
    except KeyboardInterrupt:
        pass
    # close port connections
    icub.cleanup()
    
    



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 11:09:58 2021

@author: s7lawren
"""
import matplotlib.pylab
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
    icub.connect_image_port(args.input_port, args.output_port)
    
    image = icub.get_image()
    H,W,D = image.shape
    
    # construct a blob from the input image and then perform a forward
    blob         = yolo.get_blob_from_image(image)
    yolo.net.setInput(blob)
    start        = time.time()
    layerOutputs = yolo.net.forward(yolo.output_layers)
    end          = time.time()
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    boxes, confidences, classIDs = yolo.get_object_data(layerOutputs, image_width=W, image_height=H)
    boxes, confidences, classIDs = yolo.parse_object_data(boxes, confidences, classIDs)
    
    clutter = len(classIDs)
    uncertainty = 1 - np.round(np.mean(confidences), 2)
    
    image = yolo.get_yolo_image(image, boxes, confidences, classIDs)
    
    icub.cleanup()
    
    # display the image that has been read
    matplotlib.pylab.imshow(image)
    
    fuzzy = FuzzySystem()
    # fuzzy.plot_membersip_functions()
    fuzzy.interpret(clutter, uncertainty)
    fuzzy.plot_interpretation()
    print(fuzzy.emotion)


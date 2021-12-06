#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 19:30:17 2021

@author: s7lawren
"""
import numpy as np
import yarp

class ICub():
    
    def intialize_yarp(self):
        yarp.Network.init()

    def connect_image_port(self, input_port, output_port):
        # Create a port and connect it to the iCub simulator virtual camera
        self.image_port = yarp.Port()
        self.image_port.open(output_port)
        yarp.Network.connect(input_port, output_port)
        
    def get_image(self):
        # Create numpy array to receive the image and the YARP image wrapped around it
        img_array = np.zeros((240, 320, 3), dtype=np.uint8)
        yarp_image = yarp.ImageRgb()
        yarp_image.resize(320, 240)
        yarp_image.setExternal(img_array, img_array.shape[1], img_array.shape[0])
         
        # Read the data from the port into the image
        self.image_port.read(yarp_image)
        return img_array
    
    def cleanup(self):
        self.image_port.close()
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

    def connect_image_ports(self, right_view_input_port, right_view_output_port, 
                            left_view_input_port, left_view_output_port):
        # Create a port and connect it to the iCub simulator virtual camera
        self.right_image_port = yarp.Port()
        self.right_image_port.open(right_view_output_port)
        yarp.Network.connect(right_view_input_port, right_view_output_port)
        
        self.left_image_port = yarp.Port()
        self.left_image_port.open(left_view_output_port)
        yarp.Network.connect(left_view_input_port, left_view_output_port)
        
        self.image_output_port = yarp.Port()
        self.image_output_port.open('/icub/yolo/view')
        yarp.Network.connect('/icub/yolo/view', '/view01')
    
    def open_stress_port(self, port):
        self.stress_port = yarp.Port()
        self.stress_port.open(port)
    
    def open_object_port(self, port):
        self.object_port = yarp.Port()
        self.object_port.open(port)
    
    def get_right_image(self):
        # Create numpy array to receive the image and the YARP image wrapped around it
        img_array = np.zeros((244, 320, 3), dtype=np.uint8)
        yarp_image = yarp.ImageRgb()
        yarp_image.resize(320, 244)
        yarp_image.setExternal(img_array, img_array.shape[1], img_array.shape[0])
         
        # Read the data from the port into the image
        self.right_image_port.read(yarp_image)
        return img_array
    
    def get_left_image(self):
        # Create numpy array to receive the image and the YARP image wrapped around it
        img_array = np.zeros((244, 320, 3), dtype=np.uint8)
        yarp_image = yarp.ImageRgb()
        yarp_image.resize(320, 244)
        yarp_image.setExternal(img_array, img_array.shape[1], img_array.shape[0])
         
        # Read the data from the port into the image
        self.left_image_port.read(yarp_image)
        return img_array
    
    
    def publish_affect_level(self, affect):
        yarp_bottle = yarp.Bottle()
        yarp_bottle.addDouble(affect)
        self.stress_port.write(yarp_bottle)
        
    def publish_object_image(self, image):
        yarp_image = yarp.ImageRgb()
        yarp_image.setExternal(np.ascontiguousarray(image), image.shape[1], image.shape[0])
        self.image_output_port.write(yarp_image)
        
    def publish_objects(self, objects):
        yarp_bottle = yarp.Bottle()
        
        left_objects, right_objects = objects[0], objects[1]
        visible_objects = min(len(left_objects), len(right_objects))
        
        yarp_bottle.addInt(visible_objects)
        
        for i in range(visible_objects):
            x, y = left_objects[i]
            yarp_bottle.addDouble(float(x))
            yarp_bottle.addDouble(float(y))
            
            x, y = right_objects[i]
            yarp_bottle.addDouble(float(x))
            yarp_bottle.addDouble(float(y))
            
        self.object_port.write(yarp_bottle)
    
    def cleanup(self):
        self.right_image_port.close()
        self.left_image_port.close()
        self.image_output_port.close()
        self.stress_port.close()
        self.object_port.close()
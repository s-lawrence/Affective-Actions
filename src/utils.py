#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 12:39:38 2021

@author: s7lawren
"""
import argparse
import cv2
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--yolo", default="../../darknet", help="Path to root of YOLO")
parser.add_argument("--input_port", default="/icubSim/cam/right", help="Input camera port to use for iCub")
parser.add_argument("--output_port", default="/python-image-port", help="Output port to use for image from iCub")
args = parser.parse_args()


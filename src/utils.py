#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 12:39:38 2021

@author: s7lawren
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--yolo", default="../darknet", help="Path to root of YOLO")
parser.add_argument("--right_view_input_port", default="/icub/cam/right", help="Input camera port to use right eye for iCub")
parser.add_argument("--left_view_input_port", default="/icub/cam/left", help="Input camera port to use left eye for iCub")
parser.add_argument("--right_view_output_port", default="/python-right-view", help="Output port to use right eye from iCub")
parser.add_argument("--left_view_output_port", default="/python-left-view", help="Output port to use left eye from iCub")
parser.add_argument("--emotion_port", default="/icub/affect", help="Output port iCub affect")
parser.add_argument("--object_port", default="/icub/view/objects", help="Output port iCub affect")

args = parser.parse_args()


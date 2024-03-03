import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# Import necessary modules
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import pycocotools.mask as mask_utils

import cv2

import torch
from PIL import Image

import os
import shutil
import numpy as np

import time

import pickle

# Link to understand: https://pallawi-ds.medium.com/detectron2-evaluation-cocoevaluator-9f1ab0236d4c

# Load a configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))  # Load your configuration file
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

# Load pretrained weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")

# Create a predictor
predictor = DefaultPredictor(cfg)

# https://github.com/ppwwyyxx/cocoapi/blob/master/PythonAPI/pycocotools/mask.py
# All this code is extracted from the pycocotools library and it's to decode the info from the KITTI dataset

'''
https://github.com/ppwwyyxx/cocoapi/blob/master/PythonAPI/pycocotools/_mask.pyx
After a lot of looking for how to convert the lre string, we found that piece of code.
It's a cython code that is used to convert the lre string to a binary mask.
objs.append({
    'size': [Rs._R[i].h, Rs._R[i].w],
    'counts': py_string
})

They need a dictionaty in the cython, so we are going to create that dictionary in order to convert the lre string

This is where we extracted the info to understand the KITTI-MOTS dataset: https://www.vision.rwth-aachen.de/page/mots
'''
if __name__ == '__main__':

    number_sequence = '0010'
    with open('../KITTI-MOTS/instances_txt/' + number_sequence + '.txt', 'r') as f:
        old_frame = None

        for line in f:
            line = line.strip()
            lines = line.split(" ")
            

            height = int(lines[3])
            width = int(lines[4])
            lre = lines[5]

            frame = lines[0].zfill(6) + '.png'

            id = lines[1]

            if id != '10000':
            # if True:

                if old_frame is None:
                    old_frame = frame
                    image = cv2.imread('../KITTI-MOTS/training/image_02/' + number_sequence + '/' + frame, cv2.IMREAD_COLOR)

                requirement = {
                    'size': [height, width],
                    'counts': lre.encode('utf-8')
                }

                bboxCoordinates = mask_utils.toBbox(requirement)

                x, y, w, h = bboxCoordinates

                binaryMask = mask_utils.decode(requirement)

                if old_frame != frame:

                    # cv2.imshow('Image', image)
                    # cv2.waitKey(1)
                    cv2.imwrite('GT_008/' + str(old_frame).zfill(6), image)

                    image = cv2.imread('../KITTI-MOTS/training/image_02/' + number_sequence + '/' + frame, cv2.IMREAD_COLOR)

                    cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

                    old_frame = frame
                
                else:
                
                    cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                    




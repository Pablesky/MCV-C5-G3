import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# Import necessary modules
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import cv2

import torch
from PIL import Image

import os
import shutil
import numpy as np

import time

import pickle  
# Load a configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))  # Load your configuration file
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

# Load pretrained weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")

# Create a predictor
predictor = DefaultPredictor(cfg)


dataset_path = '../KITTI-MOTS/testing/image_02'
output_path = 'KITTI-MOTS-predicted-mask/'
kitti_mots = os.listdir(dataset_path)


if os.path.exists(output_path):  
        shutil.rmtree(output_path)

os.mkdir(output_path)

validation = ['0002', '0006', '0007', '0008', '0010', '0013', '0014', '0016', '0018']

timeExecution = 0
nImages = 0
timeLists = []


for folder in kitti_mots:
    if folder in validation:  
        actual_folder = os.path.join(output_path, folder)

        if os.path.exists(actual_folder):  
            shutil.rmtree(actual_folder)

        os.mkdir(actual_folder)

        images = os.listdir(os.path.join(dataset_path, folder))

        for image in images:
            image_path = os.path.join(dataset_path, folder, image)

            image_readed = cv2.imread(image_path, cv2.IMREAD_COLOR)

            start_time = time.time()
            outputs = predictor(image_readed)
            end_time = time.time()
            
            timeExecution += (end_time - start_time)
            nImages += 1

            timeLists.append(end_time - start_time)

            instance_output = outputs["instances"]

            # Car class 1 and Person class 0
            class_ids = outputs["instances"].get("pred_classes").cpu().numpy()

            classes_to_draw = [0, 2]

            mask = [class_id in classes_to_draw for class_id in class_ids]

            instances_to_draw = outputs["instances"][mask]

            v = Visualizer(image_readed, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(instances_to_draw.to("cpu"))

            imgOut = out.get_image()

            image_out_path = os.path.join(actual_folder, image)
            cv2.imwrite(image_out_path, imgOut)

print('Average time: ', timeExecution/nImages)

with open('timeListsSegmentation.pkl', 'wb') as f:
    pickle.dump(timeLists, f)

'''
# Load an image
image = cv2.imread('sigrid.jpg')

# Perform inference
outputs = predictor(image)

# Visualize the predictions (optional)
v = Visualizer(image, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

imgOut = out.get_image()

cv2.imwrite('frame.png', imgOut)
'''


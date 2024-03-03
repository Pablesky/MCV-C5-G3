import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# Import necessary modules
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import random
import cv2

import torch
from PIL import Image

import os
import shutil
import numpy as np

import time

import pickle


from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from kitti2coco import *

"""
EVAL FASTER-RCNN ON COCO METRICS

"""

if __name__ == '__main__':

    datasets_folders = {
        'validation' : ['0002', '0006', '0007', '0008', '0010', '0013', '0014', '0016', '0018'],
        'training' : ['0000', '0001', '0003', '0004', '0005', '0009', '0011', '0012', '0015', '0017', '0019', '0020']
    }

    # Load a configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))  # Load your configuration file
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

    # Load pretrained weights
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = os.path.join('./finetune-detection', "model_final.pth")

    #register your data
    # register_coco_instances("my_dataset_train", {}, "./training_COCO_GT.json", "../KITTI-MOTS/testing/image_02'")
    # register_coco_instances("my_dataset_val", {}, "./validation_COCO_GT.json", "../KITTI-MOTS/testing/image_02'")

    for i in ['training', 'validation']:
        DatasetCatalog.register("KITTI_MOTS_" + i, lambda d=i: get_data_dict(datasets_folders[d]))
        MetadataCatalog.get("KITTI_MOTS_" + i).set(thing_classes=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes)

    dataset_dicts = get_data_dict(datasets_folders['training'])
    dataset_metadata = MetadataCatalog.get("KITTI_MOTS_training")

    # Set to true to visualize the images
    if False:
        for d in random.sample(dataset_dicts, 3):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata, scale=0.5)
            out = visualizer.draw_dataset_dict(d)
            cv2.imshow('frame', out.get_image()[:, :, ::-1])
            cv2.waitKey(0)

        cv2.destroyAllWindows()

    # Create a predictor
    predictor = DefaultPredictor(cfg)


    # my_dataset_training
    # my_dataset_validation

    #Call the COCO Evaluator function and pass the Validation Dataset
    
    output_evaluation_folder = 'detection-evaluation-training'
    if os.path.exists(output_evaluation_folder):  
        shutil.rmtree(output_evaluation_folder)

    os.mkdir(output_evaluation_folder)

    dataset_to_evaluate = "KITTI_MOTS_training"
    # dataset_to_evaluate = "KITTI_MOTS_validation"
    evaluator = COCOEvaluator(dataset_to_evaluate, output_dir = output_evaluation_folder)
    val_loader = build_detection_test_loader(cfg, dataset_to_evaluate)

    #Use the created predicted model in the previous step
    st = inference_on_dataset(predictor.model, val_loader, evaluator)
    

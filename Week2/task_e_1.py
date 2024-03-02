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

from detectron2.engine import DefaultTrainer

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
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")

    #register your data
    # register_coco_instances("my_dataset_train", {}, "./training_COCO_GT.json", "../KITTI-MOTS/testing/image_02'")
    # register_coco_instances("my_dataset_val", {}, "./validation_COCO_GT.json", "../KITTI-MOTS/testing/image_02'")

    for i in ['training', 'validation']:
        DatasetCatalog.register("my_dataset_" + i, lambda d=i: get_data_dict(datasets_folders[d]))
        MetadataCatalog.get("my_dataset_" + i).set(thing_classes=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes)

    coco_labels = MetadataCatalog.get("my_dataset_training").thing_classes

    dataset_dicts = get_data_dict(datasets_folders['training'])
    dataset_metadata = MetadataCatalog.get("my_dataset_training")

    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow('frame', out.get_image()[:, :, ::-1])
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_training")
    cfg.DATASETS.TEST = ("my_dataset_validation")
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(coco_labels)  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    
    if os.path.exists(cfg.OUTPUT_DIR):  
        shutil.rmtree(cfg.OUTPUT_DIR)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()
    

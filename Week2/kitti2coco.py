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

from PIL import Image

import os
import shutil
import numpy as np

from pathlib import Path

import json

mapping_kitti_2_coco = {
    1 : 2,# 1 car in kitti, 2 car in coco
    2 : 0 # 2 pedestrian in kitti, 0 person in coco
}

if __name__ == '__main__':

    # Init annotations
    annotations = {}
    annotations["licenses"] = [ { } ]
    
    # according to provided slides 
    annotations["categories"] =  [
        { 
            "id": mapping_kitti_2_coco[1], # Remember that the class 1 in kitti is the car
            "name": "car",
            "supercategory": "vehicle"
        },

        { 
            "id": mapping_kitti_2_coco[2], # Remember that the class 2 in kitti is the person
            "name": "pedestrian",
            "supercategory": "person"
        }
    ]

    text_notations_folder = '../KITTI-MOTS/instances_txt'

    output_path = 'COCO-annotations'
    if os.path.exists(output_path):  
        shutil.rmtree(output_path)

    os.mkdir(output_path)

    image_id = 1
    for file in Path(text_notations_folder).glob('*'):

        annotations['images'] = []
        annotations['annotations'] = []

        oldFrame = None
        
        object_id = 1

        with open(file, 'r') as f:
            
            for line in f:
                lineClean = line.strip().split(' ')

                framePath = lineClean[0].zfill(6) + '.png'

                ob_id = int(lineClean[1]) % 1000

                class_id = int(lineClean[2])
                id_tracking = int(ob_id)
                
                height = int(lineClean[3])
                width = int(lineClean[4])
                lre = lineClean[5]

                rleObject = {
                    'size': [height, width],
                    'counts': lre.encode('utf-8')
                }

                area = mask_utils.area(rleObject)

                bboxCoordinates = mask_utils.toBbox(rleObject).tolist()
                bboxCoordinates = [int(value) for value in bboxCoordinates]

                if oldFrame is None or oldFrame != framePath:
                    filename = f'../KITTI-MOTS/training/image_02/{file.stem}/{framePath}'

                    annotations['images'].append({
                        "id": image_id,
                        "width": width,
                        "height": height,
                        "file_name": filename
                    })

                    oldFrame = framePath

                    image_id += 1
                    object_id = 1

                annotations['annotations'].append({
                    "id": object_id,
                    "image_id": image_id,
                    "category_id": mapping_kitti_2_coco[class_id],
                    "segmentation": [],
                    "area": float(area),
                    "bbox": bboxCoordinates,
                    "iscrowd": 0
                })

                object_id += 1
            
            with open(f'{output_path}/{file.stem}.json', 'w') as f:
                json.dump(annotations, f)
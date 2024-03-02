from pathlib import Path
from detectron2.structures import BoxMode
import pycocotools.mask as mask_utils
import os
import json
from detectron2.data import MetadataCatalog, DatasetCatalog

mapping_kitti_2_coco = {
    1 : 2,# 1 car in kitti, 2 car in coco
    2 : 0 # 2 pedestrian in kitti, 0 person in coco
}

validation = ['0002', '0006', '0007', '0008', '0010', '0013', '0014', '0016', '0018']
training = ['0000', '0001', '0003', '0004', '0005', '0009', '0011', '0012', '0015', '0017', '0019', '0020']

def get_data_dict(data_folders):
    images_path = '../KITTI-MOTS/training/image_02/'
    text_notations_folder = '../KITTI-MOTS/instances_txt/'

    dataset = []

    image_id = 1

    for folder in data_folders:
        text_file = os.path.join(text_notations_folder, folder + '.txt')

        imagesPath = os.path.join(images_path, folder)
        imagesList = sorted(list(Path(imagesPath).glob('*.png')))


        # dataset_dicts = [] # I think this has to be for each sequence

        # one record for each image
        # we are assuming anotations in each file are ordered by frame

        oldFrame = None
         # i dont think we need it because it is provided at each line

        #print(imagesList)

        with open(text_file, 'r') as f: #we read each line of the annotations
            for line in f:
                cleanLine = line.strip().split(' ')

                frameIndex = int(cleanLine[0])

                ob_id = int(cleanLine[1]) % 1000

                class_id = int(cleanLine[2])
                id_tracking = int(ob_id)
                
                height = int(cleanLine[3])
                width = int(cleanLine[4])
                lre = cleanLine[5]

                rleObject = {
                    'size': [height, width],
                    'counts': lre.encode('utf-8')
                }

                bboxCoordinates = mask_utils.toBbox(rleObject).tolist() # im not sure im doing this right, but im going to suppose this are the coordinates of the bb

                if class_id != 10000 and class_id != 10: # Si es el objeto ese del fondo no queremos meterlo en el dataset, entonces lo skipeamos

                    if oldFrame is None:
                        # Inicializamos un frame nuevo right at the begining case
                        record = {} # initialize new record because it is a new image
                        record["file_name"] = str(imagesList[frameIndex])
                        record["image_id"] = image_id #initial frame
                        record["height"] = height
                        record["width"] = width

                        objs = [] #initialize objects
                        obj = {
                            "id": id_tracking,
                            "bbox": bboxCoordinates,
                            "bbox_mode": BoxMode.XYWH_ABS,
                            "segmentation": rleObject,
                            "category_id": mapping_kitti_2_coco[class_id],
                        }
                        objs.append(obj)

                        oldFrame = frameIndex

                    elif frameIndex != oldFrame:
                        # Cambiamos de frame --> canviar frame id sumant-ne 

                        record["annotations"] = objs #add what we had to objs in the previous record

                        dataset.append(record) #append to this sequence dataset the complete records

                        image_id += 1

                        record = {} #initialzie new records

                        record["file_name"] = str(imagesList[frameIndex])
                        record["image_id"] = image_id #im not sure intex frame is correct
                        record["height"] = height
                        record["width"] = width


                        obj = {
                            "id": id_tracking,
                            "bbox": bboxCoordinates,
                            "bbox_mode": BoxMode.XYWH_ABS,
                            "segmentation": rleObject,
                            "category_id": mapping_kitti_2_coco[class_id],
                        }

                        objs = []
                        objs.append(obj)

                        oldFrame = frameIndex

                    else:
                        # Seguimos en el mismo frame --> continuar fent append de les anotatcions
                        obj = {
                            "id": id_tracking,
                            "bbox": bboxCoordinates,
                            "bbox_mode": BoxMode.XYWH_ABS,
                            "segmentation": rleObject,
                            "category_id": mapping_kitti_2_coco[class_id],
                        }
                        objs.append(obj)

    return dataset
    """with open("./validation_COCO_GT.json", 'w') as f:
        json.dump(dataset, f)"""

                    




                    
if __name__ == '__main__':
    print(get_data_dict(training)[::-1][:3])
    '''
    for d in ["training", "validation"]:
        DatasetCatalog.register("our_" + d, lambda d=d: get_data_dict("./" + d))
        MetadataCatalog.get("our_" + d).set(thing_classes=["balloon"])
    balloon_metadata = MetadataCatalog.get("balloon_train")
    '''
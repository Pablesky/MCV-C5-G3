import os
import shutil

import kitti2coco

validation = ['0002', '0006', '0007', '0008', '0010', '0013', '0014', '0016', '0018']
training = ['0000', '0001', '0003', '0004', '0005', '0009', '0011', '0012', '0015', '0017', '0019', '0020']

kitti2yolo_dict = {
    2 : 0, # 2 car in coco, 0 car in yolo
    0 : 1 # 0 pedestrian in coco, 2 person in yolo
}

def kitti2yolo(dataset_name):

    output_path = 'yolov9/data/'

    output_path = os.path.join(output_path, dataset_name)

    train_folder = os.path.join(output_path, 'train')
    test_folder = os.path.join(output_path, 'test')
    valid_folder = os.path.join(output_path, 'valid')

    if os.path.exists(train_folder):  
        shutil.rmtree(train_folder)

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(os.path.join(train_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_folder, 'labels'), exist_ok=True)

    if os.path.exists(test_folder):  
        shutil.rmtree(test_folder)

    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(os.path.join(test_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(test_folder, 'labels'), exist_ok=True)

    if os.path.exists(valid_folder):  
        shutil.rmtree(valid_folder)

    os.makedirs(valid_folder, exist_ok=True)
    os.makedirs(os.path.join(valid_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(valid_folder, 'labels'), exist_ok=True)

    training_dict = kitti2coco.get_data_dict(training)

    for key in training_dict:
        
        new_image_path = os.path.join(train_folder, 'images', str(key['image_id']).zfill(6) + '.png')
        shutil.copy(key['file_name'], new_image_path)

        annotations = key['annotations']

        height, width = float(key['height']), float(key['width'])

        with open(os.path.join(train_folder, 'labels', str(key['image_id']).zfill(6) + '.txt'), 'w') as f:
            for annotation in annotations:

                w_norm = (float(annotation['bbox'][2]) / width)
                h_norm = (float(annotation['bbox'][3]) / height)

                x_cent = (float(annotation['bbox'][0]) / width) + (w_norm / 2)
                y_cent = (float(annotation['bbox'][1]) / height) + (h_norm / 2)
                
                f.write(f"{kitti2yolo_dict[annotation['category_id']]} {x_cent} {y_cent} {w_norm} {h_norm}\n")
                
    validation_dict = kitti2coco.get_data_dict(validation)

    for key in validation_dict:
        
        new_image_path = os.path.join(test_folder, 'images', str(key['image_id']).zfill(6) + '.png')
        shutil.copy(key['file_name'], new_image_path)

        new_image_path = os.path.join(valid_folder, 'images', str(key['image_id']).zfill(6) + '.png')
        shutil.copy(key['file_name'], new_image_path)

        annotations = key['annotations']

        height, width = float(key['height']), float(key['width'])

        with open(os.path.join(valid_folder, 'labels', str(key['image_id']).zfill(6) + '.txt'), 'w') as x:
            with open(os.path.join(test_folder, 'labels', str(key['image_id']).zfill(6) + '.txt'), 'w') as y:
                for annotation in annotations:

                    w_norm = (float(annotation['bbox'][2]) / width)
                    h_norm = (float(annotation['bbox'][3]) / height)

                    x_cent = (float(annotation['bbox'][0]) / width) + (w_norm / 2)
                    y_cent = (float(annotation['bbox'][1]) / height) + (h_norm / 2)
                    
                    x.write(f"{kitti2yolo_dict[annotation['category_id']]} {x_cent} {y_cent} {w_norm} {h_norm}\n")
                    
                    y.write(f"{kitti2yolo_dict[annotation['category_id']]} {x_cent} {y_cent} {w_norm} {h_norm}\n")
    
    with open(os.path.join(output_path, 'data.yaml'), 'w') as f:
        f.write('names:\n')
        f.write('- car\n')
        f.write('- person\n')
        f.write('nc: 2\n')
        f.write(f'train: data/KITTI-MOTS/train/images\n')
        f.write(f'val: data/KITTI-MOTS/valid/images\n')
        f.write(f'test: data/KITTI-MOTS/test/images\n')

if __name__ == '__main__':
    kitti2yolo('KITTI-MOTS')


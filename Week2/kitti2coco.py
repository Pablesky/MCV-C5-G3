from pathlib import Path

mapping_kitti_2_coco = {
    1 : 2,# 1 car in kitti, 2 car in coco
    2 : 0 # 2 pedestrian in kitti, 0 person in coco
}

validation = ['0002', '0006', '0007', '0008', '0010', '0013', '0014', '0016', '0018']
training = ['0000', '0001', '0003', '0004', '0005', '0009', '0011', '0012', '0015', '0017', '0019', '0020']

def get_data_dict(data_folders):
    images_path = '../KITTI-MOTS/training/image_02/'
    text_notations_folder = '../KITTI-MOTS/instances_txt/'

    dataset_dicts = []

    for folder in data_folders:
        
        print(folder)


if __name__ == '__main__':
    get_data_dict(validation)
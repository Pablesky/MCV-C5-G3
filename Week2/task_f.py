import os

if __name__ == "__main__":
    print('IMPORTANT TO BE IN THE YOLOV9 DIRECTORY')

    # We had to clone the yolov9 repository to the current directory and install the requirements and roboflow
    # git clone https://github.com/SkalskiP/yolov9.git

    # Also we need to get the weights from the yolov9 repository
    # mkdir weights
    # wget -P weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt
    # wget -P weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt
    # wget -P weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt
    # wget -P weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-e.pt

    # Notebook to train the model with the code: https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov9-object-detection-on-custom-dataset.ipynb#scrollTo=N68Bdf4FsMYW

    # To train the model on a custom dataset use the kitti2yolo.py and this command
    # python train_dual.py --batch 8 --epochs 25 --img 640 --device 0 --min-items 0 --close-mosaic 15 --data data/KITTI-MOTS/data.yaml --weights weights/yolov9-c.pt --cfg models/detect/yolov9-c.yaml --hyp hyp.scratch-high.yaml
    # With --bath 16 -> 20 GB of VRAM

    # Also the BBox of yolo are in the shape of (x_center, y_center, width, height) and normalized to the image size


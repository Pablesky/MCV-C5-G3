import os
from roboflow import Roboflow
import subprocess
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

    # python detect.py --img 1280 --conf 0.1 --device 0 --weights {HOME}/yolov9/runs/train/exp/weights/best.pt --source {dataset.location}/test/images

    # Path to your detect.py script
    detect_script = "detect.py"

    # Path to the directory containing your images
    image_dir = "../../KITTI-MOTS/training/image_02/0010/"

    # Path to the best weights file
    weights_file = "runs/train/exp/weights/best.pt"

    # Command to execute
    command = f"python {detect_script} --img 640 --conf 0.1 --device 0 --weights {weights_file}"

    # List all image files in the directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]

    # Iterate over each image file and execute the command
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        full_command = f"{command} --source {image_path}"
        subprocess.run(full_command, shell=True)
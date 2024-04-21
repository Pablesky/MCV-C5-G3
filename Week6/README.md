# Week 5: Multimodal model

## Folder structure 
The code and data is structured as follows:

        .
        ├── data_augmentation.ipynb         # Is the code to aggregate the augemented data to the dataset.
        ├── plot_embedding.ipynb            # Code to plot the embedding of the metric learning.
        ├── task_a.ipynb                    # Code to the data science part.
        ├── task_b.py                       # Code to train the model for image classification.
        ├── task_b_metric_learning.py       # Code to train the model for image classification using metric learning.
        └── task_e.py                       # Code to train the multimodal model.

All the other files are necessary for the execution of the main ones that are explained above.

## Requirements
Standard Computer Vision python packages are used. Regarding the python version, Python >= 3.6 is needed.

- PyTorch:
  ```pip install torch```
- TorchVision:
  ```pip install torchvision```
- TorchInfo:
  ```pip install torchinfo```
- WandB:
  ```pip install wandb```
- UMAP
  ```pip install umap-learn```
- FAISS
```conda install -c pytorch faiss-cpu=1.8.0```
- pytorch-metric-learning
  ```pip install pytorch-metric-learning```
- Diffusers
  ```pip install diffusers```

## Running the code
Each task corresponds to a separate file named after the task. To execute them, simply specify the desired hyperparameter values within the "main" section of the respective file and run it using Python 3, as demonstrated below. If the file is a .ipynb file, just run the code cell by cell.

```bash
python3 task.py
 ```
The notebooks, just run them as normal.

## Tasks
The main goal of this project is to try and understand how multimodal models work and to see if having a lot of dimensions is also a good idea.

- a) Explore the data.
- b) Train a simple image classification model.
- c) Training strategy
- d) Training strategy.
- e), f) Feature extraction for audio and text.
- g) Multimodal model training. 

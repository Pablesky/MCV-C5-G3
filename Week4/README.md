# Week 3: Image Retrieval

## Folder structure 
The code and data is structured as follows:

        .
        ├── generate_data.py             # Creates a pkl file of all the data of the datasets.
        ├── task_a.py                    # Code to train the models for the task a.
        ├── task_a_retrieval.py          # Performs retrieval and evaluation for task a.
        ├── task_n.py                    # Code to train the models for the task b.
        ├── task_b_retrieval.py          # Performs retrieval and evaluation for task b.
        ├── utils.py                     # Some utility functions used across the code.
        └── task_e/                      # A folder with all the files regarding the task.

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


## Tasks
The main goal of this project is to perform image retrieval in a pre-trained model and compare it to other methods trained with metric learning (siamese network and triplet network). Moreover, we see how the retrieval performs with a different dataset (COCO - multiobject). Thus, the main tasks are:

- a) Image to text retrieval.
- b) Text to image retrieval.
- c) Image to text retrieval using BERT.
- d) Text to image retrieval using BERT.

All the hyperparameters are optimized using wandb.ai.

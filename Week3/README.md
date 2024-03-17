# Week 3: Image Retrieval

## Folder structure 
The code and data is structured as follows:

        .
        ├── task_a_generateData.py             # Creates a pkl file of all embeddings of the datasets.
        ├── task_a_retrieval.py                # Performs retrieval and evaluation for task a.
        ├── task_a_retrieval_optimizer.py      # Calls for WandB optimization.
        ├── task_b.py                          # Siamese net and optimization of it.
        ├── task_c.py                          # Triplet net and optimization of it.
        ├── task_d.ipynb                       # Visualizations of all the previous results.
        └── task_whatever.py

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

- a) Image retrieval with pre-trained image classification model.
- b) Train the model on metric learning (Siamese network).
- c) Train the model on metric learning (Triplet network).
- d) Visualize the learned image representation of each of the previous tasks a-c.
- e) Image Retrieval on COCO with Faster R-CNN or Mask R-CNN.


All the hyperparameters are optimized using wandb.ai.

# Week 5: Diffusion models

## Folder structure 
The code and data is structured as follows:

        .
        ├── generate_data.py             # Creates a pkl file of all the data of the datasets.
        ├── task_a.py                    # Code to train the models for the task a.
        ├── task_a_retrieval.py          # Performs retrieval and evaluation for task a.
        ├── task_b.py                    # Code to train the models for the task b.
        ├── task_b_retrieval.py          # Performs retrieval and evaluation for task b.
        ├── task_c_a1.py                 # Performs the training for the BERT model for task img2txt.
        ├── task_c_a2.py                 # Performs retrieval and evaluation for task c.
        ├── task_c_b1.py                 # Performs the training for the BERT model for task txt2img.
        ├── task_c_b2.py                 # Performs retrieval and evaluation for task d.
        ├── utils.py                     # Some utility functions used across the code.
        └── utils_c.py                   # Some utility functions used across the code.

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
The main goal of this project is to try and understand how diffusion models work and the use of their parameters. Moreover, to give some use to these kind of architecture and find a problem from our previous retrievals and solve it using diffusion. Thus, the main tasks are:

- a) Using open-sourced models.
- b) Exploration of inference with diffusion models.
- c) Analysis and problem statement.
- d) Building a complex pipeline.
- e) Example of an application of generative AI. 

All the hyperparameters are optimized using wandb.ai.

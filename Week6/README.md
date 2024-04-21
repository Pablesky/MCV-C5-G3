# Week 5: Diffusion models

## Folder structure 
The code and data is structured as follows:

        .
        ├── task_a.ipynb            # All the code to generate the data and try with the diffusion models.
        ├── task_b.ipynb            # All the code to experiment with the stable diffusion parameters.
        ├── week4_cb3.py            # Code reused from previous weeks to perform the retrieval.
        └── utils.py                # Some utility functions used across the code and to generate the pkl files in the main of this file.

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

## Tasks
The main goal of this project is to try and understand how diffusion models work and the use of their parameters. Moreover, to give some use to these kind of architecture and find a problem from our previous retrievals and solve it using diffusion. Thus, the main tasks are:

- a) Using open-sourced models.
- b) Exploration of inference with diffusion models.
- c) Analysis and problem statement.
- d) Building a complex pipeline.
- e) Example of an application of generative AI. 

All the hyperparameters are optimized using wandb.ai.

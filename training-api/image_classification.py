
import os
# import torch
# import torch.nn as nn
# import torchvision.models as models
# from torch.nn import functional as F
from dotenv import load_dotenv
import requests
import json

from clearml import Task

# from ml_trainer import AutoTrainer
# from ml_trainer.base import AbstractModelArchitecture
from aipmodel.model_registry import MLOpsManager



load_dotenv()
data_model_reg_cfg= {
    'ceph': 'http://s3.cloud-ai.ir',
    's3_access_key': '8HZE0U3P5VOSCPEOUN4G',
    's3_secret_key': 'ihLqlXsauVYmiV83uu5kDdzAZzjlLlXYx05OIOwg',
}

print(data_model_reg_cfg)
# --------- fetch model from model registry --------
manager = MLOpsManager(
    clearml_url=os.environ["CLEARML_URL"],
    clearml_access_key=os.environ["CLEARML_API_ACCESS_KEY"],
    clearml_secret_key=os.environ["CLEARML_API_SECRET_KEY"],
    clearml_username=os.environ["CLEARML_USERNAME"]
)

# ---------- Variables -------------
dataset='cifar-10'
epochs = 1
batch_size = 64
split_ratio = 0.8
lr = 0.01
save_model = False
load_model = False
model_name = "resnet50"
model_id = "resnet50"
model_save_name="resenet50_save1"
transform = {
            "resize": [32, 32],
            "normalize_mean": [0.5, 0.5, 0.5],
            "normalize_std": [0.5, 0.5, 0.5]
        }

# --------- ClearML task initialization --------
task = Task.init(
    project_name="API training",  # Name of the ClearML project
    task_name=f"{dataset} - {model_name} - API Training",  # Name of the task
    task_type=Task.TaskTypes.optimizer,  # Type of the task (could also be "training", "testing", etc.)
    reuse_last_task_id=False  # Whether to reuse the last task ID (set to False for a new task each time)
)
# ---------
# Ensure valid model name/id

# Dataset configuration
dataset_sources = {
    "cifar-10": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "stl10": "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
}


# --------------     to load model -----------------

if load_model: 
    model_id = manager.get_model_id_by_name(model_id)

    manager.get_model(
        model_name= model_name,  # or any valid model ID
        local_dest="."
)

#----------------- main config ----------------
cfg = {
    # Training Params
    "task": "image_classification",
    "batch_size": batch_size,
    "split_ratio": split_ratio,
    "lr": lr,
    "epochs": epochs,
    "num_classes": 10,  

    # Dataset
    "dataset_config": {
        "name": dataset,             # <-- keep dataset name
        "source": dataset_sources[dataset],  # <-- source resolved from mapping
        "transform_config": transform
    },
    
    # Model save
    "save_model": save_model,
    "model_dir": "model/",

    "load_model": load_model,  
    "model_dir": f"./{model_id}/",
    

    # Model load
    "model_config": {
        "type": "timm",
        "name": model_name,
        "pretrained": True
    }
}

# Connect hyperparameters and other configurations to the ClearML task
task.connect(cfg)

if cfg["dataset_config"]["name"] not in dataset_sources:
    raise ValueError(f"Invalid dataset: {dataset}. Choose from {list(dataset_sources.keys())}")

if cfg["model_config"]["name"] not in ["resnet50", "efficientnet_b0"]:
    raise ValueError("Invalid model name/id: choose from resnet50 or efficientnet_b0")

cfg["dataset_config"]["source"] = dataset_sources[cfg["dataset_config"]["name"]]

data_url= "http://172.15.30.79:8169/download-dataset"




# trainer = AutoTrainer(config=cfg)

# trainer.run()

if save_model:
    local_model_id = manager.add_model(
        source_type="local",
        source_path="model/",
        model_name=model_save_name,
        code_path="." , # ← Replace with the path to your model.py if you have it
    )

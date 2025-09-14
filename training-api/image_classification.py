
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

# --------- ClearML task initialization --------
task = Task.init(
    project_name="API training",  # Name of the ClearML project
    task_name=f"API Training",  # Name of the task
    task_type=Task.TaskTypes.optimizer,  # Type of the task (could also be "training", "testing", etc.)
    reuse_last_task_id=False  # Whether to reuse the last task ID (set to False for a new task each time)
)

load_dotenv()

data_model_reg_cfg= {
    #ceph related
    'CEPH_ENDPOINT': 'http://172.15.20.153',
    'CEPH_ACCESS_KEY': '8HZE0U3P5VOSCPEOUN4G',
    'CEPH_SECRET_KEY': 'ihLqlXsauVYmiV83uu5kDdzAZzjlLlXYx05OIOwg',
    'CEPH_BUCKET': 'bucket',

    #clearml
    'clearml_access': '6C8CD3D76920C6D1F81FBBBCD188734C',
    'clearml_secret': 'F3F98D9B5361768BF497DFFB8F63F6E9115C5314FDED76FF18E3B82997DFB6D5',
    'clearml_username': 'testdario7',
    'clearml_url': 'http://web.mlops.ai-lab.ir/api_old'
}
data_model_reg_cfg= {
    #ceph related
    'CEPH_ENDPOINT': 'url',
    'CEPH_ACCESS_KEY': 'access',
    'CEPH_SECRET_KEY': 'secret',
    'CEPH_BUCKET': 'bucket',

    #clearml
    'clearml_url': 'url',
    'clearml_access_key': 'access',
    'clearml_secret_key': 'secret',
    'clearml_username': 'testdario7',
}

data_model_reg_cfg['CEPH_BUCKET'] = f"{data_model_reg_cfg['clearml_username']}-bucket"
task.connect(data_model_reg_cfg, name='model_data_cfg')

print(data_model_reg_cfg)

# print("CEPH_BUCKET:", os.environ["CEPH_BUCKET"])
# print("CEPH_ENDPOINT:", os.environ["CEPH_ENDPOINT"])
# print("CEPH_ACCESS_KEY:", os.environ["CEPH_ACCESS_KEY"])
# print("CEPH_SECRET_KEY:", os.environ["CEPH_SECRET_KEY"])
# --------- fetch model from model registry --------
manager = MLOpsManager(
    clearml_url=data_model_reg_cfg['clearml_url'],
    clearml_access_key=data_model_reg_cfg['clearml_access'],
    clearml_secret_key=data_model_reg_cfg['clearml_secret'],
    clearml_username=data_model_reg_cfg['clearml_username']
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

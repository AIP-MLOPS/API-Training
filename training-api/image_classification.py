
import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from dotenv import load_dotenv
import requests
import json

from clearml import Task

from ml_trainer import AutoTrainer
from ml_trainer.base import AbstractModelArchitecture
from aipmodel.model_registry import MLOpsManager

#  ====================== Data Registry =========================

def get_dataset_download_urls(
    url: str,
    dataset_name: str,
    user_name: str,
    clearml_access_key: str,
    clearml_secret_key: str,
    s3_access_key: str,
    s3_secret_key: str,
    s3_endpoint_url: str,
    download_method: str = "presigned_urls"
):
    """
    Request presigned download URLs for a dataset.

    Returns:
        List of download URLs for .tar.gz and .csv files only
    """
    base = "http://data-ingestion-api-service.aip-mlops-service.svc.cluster.local:8169/download-dataset"

    # 1) Confirm what you’re actually hitting
    r = requests.post(f"{base}/download-dataset", json={}, timeout=10,
                  proxies={"http": None, "https": None})
    payload = {
        "dataset_name": dataset_name,
        "user_name": user_name,
        "clearml_access_key": clearml_access_key,
        "clearml_secret_key": clearml_secret_key,
        "s3_access_key": s3_access_key,
        "s3_secret_key": s3_secret_key,
        "s3_endpoint_url": s3_endpoint_url,
        "download_method": download_method
    }
    print(payload)
    
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, json=payload, headers=headers, timeout=10, proxies={"http": None, "https": None})
    # response.raise_for_status()
    data = response.json()
    print(data)
    
    # Filter for .tar.gz and .csv files only
    download_urls = [
        file["download_url"]
        for file in data.get("download_info", {}).get("files", [])
        if file["filename"].endswith(".tar.gz") or file["filename"].endswith(".csv")
    ]
    return download_urls[0]



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

task.connect(data_model_reg_cfg, name='model_data_cfg')

print(data_model_reg_cfg)
os.environ['CEPH_ENDPOINT'] = data_model_reg_cfg['CEPH_ENDPOINT']
os.environ['CEPH_ACCESS_KEY'] = data_model_reg_cfg['CEPH_ACCESS_KEY']
os.environ['CEPH_SECRET_KEY'] = data_model_reg_cfg['CEPH_SECRET_KEY']
os.environ['CEPH_BUCKET'] = data_model_reg_cfg['CEPH_BUCKET']

# print("CEPH_BUCKET:", os.environ["CEPH_BUCKET"])
# print("CEPH_ENDPOINT:", os.environ["CEPH_ENDPOINT"])
# print("CEPH_ACCESS_KEY:", os.environ["CEPH_ACCESS_KEY"])
# print("CEPH_SECRET_KEY:", os.environ["CEPH_SECRET_KEY"])

# --------- fetch model from model registry --------
manager = MLOpsManager(
    clearml_url=data_model_reg_cfg['clearml_url'],
    clearml_access_key=data_model_reg_cfg['clearml_access_key'],
    clearml_secret_key=data_model_reg_cfg['clearml_secret_key'],
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
# dataset_sources = {
#     "cifar-10": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
#     "stl10": "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
# }


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
        "source": 'url',  # <-- source resolved from datalayer
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
# import requests

# # base = "http://172.15.30.79:8169"
# base = "http://data-ingestion-api-service.aip-mlops-service.svc.cluster.local:8169"

# # 1) Confirm what you’re actually hitting
# r = requests.post(f"{base}/download-dataset", json={}, timeout=10,
#                   proxies={"http": None, "https": None})
# print("REQ:", r.request.method, r.request.url)
# print("STATUS:", r.status_code)
# print("BODY:", r.text[:400])

# # 2) Probe the API surface
# print("openapi:", requests.get(f"{base}/openapi.json",
#                               proxies={"http": None, "https": None}, timeout=5).status_code)
# print("docs:", requests.get(f"{base}/docs",
#                            proxies={"http": None, "https": None}, timeout=5).status_code)
# Data URL
url = get_dataset_download_urls(
    # url="https://api.mlops.ai-lab.ir/data/download-dataset",
    url="http://data-ingestion-api-service.aip-mlops-service.svc.cluster.local:8169/download-dataset",
    # url="https://data-ingestion-api-service:8169/download-dataset",
    dataset_name=cfg["dataset_config"]["name"],
    user_name=data_model_reg_cfg['clearml_username'],
    clearml_access_key=data_model_reg_cfg['clearml_access_key'],
    clearml_secret_key=data_model_reg_cfg['clearml_secret_key'],
    s3_access_key=data_model_reg_cfg['CEPH_ACCESS_KEY'],
    s3_secret_key=data_model_reg_cfg['CEPH_SECRET_KEY'],
    s3_endpoint_url=data_model_reg_cfg['CEPH_ENDPOINT']
)

# Connect hyperparameters and other configurations to the ClearML task
task.connect(cfg)

# if cfg["dataset_config"]["name"] not in dataset_sources:
#     raise ValueError(f"Invalid dataset: {dataset}. Choose from {list(dataset_sources.keys())}")

# if cfg["model_config"]["name"] not in ["resnet50", "efficientnet_b0"]:
#     raise ValueError("Invalid model name/id: choose from resnet50 or efficientnet_b0")

# cfg["dataset_config"]["source"] = dataset_sources[cfg["dataset_config"]["name"]]
cfg["dataset_config"]["source"] = url




trainer = AutoTrainer(config=cfg)

trainer.run()

if save_model:
    local_model_id = manager.add_model(
        source_type="local",
        source_path="model/",
        model_name=model_save_name,
        code_path="." , # ← Replace with the path to your model.py if you have it
    )

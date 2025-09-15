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
dataset='airline-passengers'
epochs = 10
batch_size = 16
save_model = False
load_model = False
model_name = "Autoformer"
model_save_name = "Autoformer_save1"
model_id = "Autoformer"

# Ensure valid model name/id
# if model_name not in ["Autoformer", "TimesNet"] and model_id not in ["Autoformer", "TimesNet"]:
#     raise ValueError("Invalid model name/id: choose from Autoformer or TimesNet")

# # Dataset configuration
# if dataset == 'airline-passengers':
#     sources = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
# elif dataset == "oil-spill":
#     sources = "https://raw.githubusercontent.com/jbrownlee/Datasets/refs/heads/master/oil-spill.csv"
# else:
#     raise ValueError("Invalid dataset: choose either 'airline-passengers' or 'oil-spill'")

# --------------     to load model -----------------
if load_model: 
    model_id = manager.get_model_id_by_name(model_name)

    manager.get_model(
        model_name= model_name,
        local_dest="."
    )

#----------------- main config ----------------

cfg = {
    "task": "timeseries",
    "epochs": epochs,
    "batch_size": batch_size,
    "seq_len": 12,
    "pred_len": 1,
    "input_channels": 1,  
    "output_size": 1,
    "device": "cpu",       

    "dataset_config": {
        "name": 'dataset',
        "source": 'url',
        # "target_column": 'target',
    },

    # Model save
    "save_model": save_model,
    "model_dir": "model/",

    "load_model": load_model,  
    "model_dir": f"./{model_id}/",




    "model_config": {
        "type": "tslib",
        "name": model_name, 
        "task_name": "long_term_forecast", 

    }
}

print(cfg)

task.connect(cfg)
url = get_dataset_download_urls(
    # url="http://api.mlops.ai-lab.ir/data/download-dataset",
    url="http://data-ingestion-api-service.aip-mlops-service.svc.cluster.local:8169/download-dataset",
    # url="http://data-ingestion-api-service:8169/download-dataset",
    dataset_name=cfg["dataset_config"]["name"],
    user_name=data_model_reg_cfg['clearml_username'],
    clearml_access_key=data_model_reg_cfg['clearml_access_key'],
    clearml_secret_key=data_model_reg_cfg['clearml_secret_key'],
    s3_access_key=data_model_reg_cfg['CEPH_ACCESS_KEY'],
    s3_secret_key=data_model_reg_cfg['CEPH_SECRET_KEY'],
    s3_endpoint_url=data_model_reg_cfg['CEPH_ENDPOINT']
)

cfg["dataset_config"]["source"] = url

trainer = AutoTrainer(config=cfg)

trainer.run()

if save_model:
    local_model_id = manager.add_model(
        source_type="local",
        source_path="model/",
        model_name=model_save_name,
        code_path="." , 
    )
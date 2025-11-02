
import os
from dotenv import load_dotenv
import requests
from pathlib import Path
import time

from clearml import Task

from ml_trainer import AutoTrainer
from aipmodel.model_registry import MLOpsManager
from data.sdk.download_sdk import s3_download

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

# --------- fetch model from model registry --------
manager = MLOpsManager(
    CLEARML_API_SERVER_URL=data_model_reg_cfg['clearml_url'],
    CLEARML_ACCESS_KEY=data_model_reg_cfg['clearml_access_key'],
    CLEARML_SECRET_KEY=data_model_reg_cfg['clearml_secret_key'],
    CLEARML_USERNAME=data_model_reg_cfg['clearml_username']
)



#----------------- main config ----------------
cfg = {
        "task": "image_classification",
        "model_name": "model registry",

        "dataset_config": {
            "source": "path/to/dataset",
            "split_ratio": 0.2,
            "batch_size": 32,
        },
        "model_config": {
            "num_classes": 10,  # !Required
            "input_channels": 3,
            "input_size": (32, 32),
            "type": "timm",            # "timm" for pretrained models, or omit for custom CNN
            "name": "resnet18",        # any supported TIMM model
            "pretrained": True
        },
        "trainer_config": {
            "lr": 1e-2,
            "load_model": None,
            "save_model": None,
            "epochs": 20,
            "device": "cuda",
            "checkpoint_path": None,
            "callbacks": None,
            "resume_from_checkpoint": None,
        },
    }

task.connect(cfg)

print(cfg)

model_reg = cfg["model_name"]

if cfg["trainer_config"]["save_model"] is not None:
    cfg["trainer_config"]["save_model"] = "model/" 

# --------------     to load model -----------------

if cfg["trainer_config"]["load_model"] is not None: 
    model_id = manager.get_model_id_by_name(model_reg)

    manager.get_model(
        model_name= model_reg,  # or any valid model ID
        local_dest="."
    )
    cfg["trainer_config"]["load_model"] = f"./{model_id}/"


s3_download(
    clearml_access_key=data_model_reg_cfg['clearml_access_key'],
    clearml_secret_key=data_model_reg_cfg['clearml_secret_key'],
    s3_access_key=data_model_reg_cfg['CEPH_ACCESS_KEY'],
    s3_secret_key=data_model_reg_cfg['CEPH_SECRET_KEY'],
    s3_endpoint_url=data_model_reg_cfg['CEPH_ENDPOINT'],
    dataset_name=cfg["dataset_config"]["source"],
    absolute_path=Path(__file__).parent/"dataset",
    user_name=data_model_reg_cfg['clearml_username']
)

absolute_path = Path(__file__).parent / "dataset" / cfg["dataset_config"]["source"]



files = list(absolute_path.rglob("*.[jc][so][nv]*"))  # matches .json or .csv

# TODO the .json and .csv is not going to work needs refinement
# Or, more explicitly: 
files = [f for f in absolute_path.rglob("*") if f.suffix in [".json", ".csv"]]

# Print absolute paths
for file_path in files:
    print(file_path.resolve())
    file_path = file_path.resolve()

# Connect hyperparameters and other configurations to the ClearML task


# config["dataset_config"]["source"] = "/home/dario/mlops/datasets/medical_qa/"
# config["dataset_config"]["source"] = '/home/dario/mlops/datasets/sample_instruction.json'

cfg["dataset_config"]["source"] = file_path


trainer = AutoTrainer(config=cfg)

trainer.run()

if cfg["trainer_config"]["save_model"] is not None:
    local_model_id = manager.add_model(
        source_type="local",
        source_path="model/",
        model_name = model_reg + "_" + str(int(time.time())),
        code_path="." , # ← Replace with the path to your model.py if you have it
    )

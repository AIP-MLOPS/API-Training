
import os
from dotenv import load_dotenv
import requests
import time
from pathlib import Path

from clearml import Task

from ml_trainer import AutoTrainer
from aipmodel.model_registry import MLOpsManager
from data.sdk.download_sdk import s3_download

# --------- ClearML task initialization --------
task = Task.init(
    project_name="Local API training",  # Name of the ClearML project
    task_name=f"API Training",  # Name of the task
    task_type=Task.TaskTypes.optimizer,  # Type of the task (could also be "training", "testing", etc.)
    reuse_last_task_id=False  # Whether to reuse the last task ID (set to False for a new task each time)
)

load_dotenv()

data_model_reg_cfg= {
    #ceph related
    'CEPH_ENDPOINT': 'default',
    'CEPH_ACCESS_KEY': 'default',
    'CEPH_SECRET_KEY': 'default',
    'CEPH_BUCKET': 'default',

    #clearml
    'clearml_url': 'default',
    'clearml_access_key': 'default',
    'clearml_secret_key': 'default',
    'clearml_username': 'default',
}

data_model_reg_cfg= {
    # Ceph 
    'CEPH_ENDPOINT': 'http://144.172.105.98:7000',
    'CEPH_ACCESS_KEY': '3386LN5KA2OFQXPTYM9S',
    'CEPH_SECRET_KEY': 'AALvi6KexAeSNCsOMRqDHTRf10BQzNyy5BQnGIfO',
    'CEPH_BUCKET': 'mlopsadminv2-bucket',

    # ClearML
    'clearml_url': 'http://144.172.105.98:30003',
    'clearml_access_key': '65592A380E9EB6F013881A57E0FE6389',
    'clearml_secret_key': '1FE51DAD67FB066710CF935911A84058FE9279B1122E0CC4719C505B932DAE81',
    'clearml_username': 'datauserv2',
}


task.connect(data_model_reg_cfg, name='model_data_cfg')

print(data_model_reg_cfg)
os.environ['CEPH_ENDPOINT_URL'] = data_model_reg_cfg['CEPH_ENDPOINT']
os.environ['S3_ACCESS_KEY'] = data_model_reg_cfg['CEPH_ACCESS_KEY']
os.environ['S3_SECRET_KEY'] = data_model_reg_cfg['CEPH_SECRET_KEY']
os.environ['S3_BUCKET_NAME'] = data_model_reg_cfg['CEPH_BUCKET']

# --------- fetch model from model registry --------
manager = MLOpsManager(
    CLEARML_API_SERVER_URL=data_model_reg_cfg['clearml_url'],
    CLEARML_ACCESS_KEY=data_model_reg_cfg['clearml_access_key'],
    CLEARML_SECRET_KEY=data_model_reg_cfg['clearml_secret_key'],
    CLEARML_USERNAME=data_model_reg_cfg['clearml_username']
)


class PrintSaveDirCallback():
    def on_save(self):
        import os
        print(f"\n[Callback] Model saved to: {self.output_dir}")
        print("[Callback] Files inside:")
        for f in os.listdir(self.output_dir):
            print("  -", f)

#----------------- main config ----------------
config = {
        "task": "image_classification",
        "model_name": "model registry",

        "dataset_config": {
            "source": "microsoft-cats_vs_dogs",  # !Required
            "split_ratio": 0.2,
            "batch_size": 32,
        },
        "model_config": {
            "num_classes": 2,  # !Required
            "input_channels": 3,
            "input_size": (32, 32),
            "type": "timm",            # "timm" for pretrained models, or omit for custom CNN
            "name": "resnet18",        # any supported TIMM model
            "pretrained": True
        },
        "trainer_config": {
            "lr": 1e-2,
            "load_model": None,
            # "save_model": None,
            "save_model": True,
            "epochs": 20,
            "device": "cuda",
            "checkpoint_path": None,
            "callbacks": None,
            "resume_from_checkpoint": None,
        },
    }

task.connect(config)

print(config)

model_reg = config["model_name"]

if config["trainer_config"]["save_model"] is not None:
    config["trainer_config"]["save_model"] = "model/"

# --------------     to load model -----------------

if config["trainer_config"]["load_model"] is not None: 
    model_id = manager.get_model_id_by_name(model_reg)

    manager.get_model(
        model_name= model_reg,  # or any valid model ID
        local_dest="."
    )
    # !!! ask from the data team to load from the right path
    config["trainer_config"]["load_model"] = f"./{model_id}/"


s3_download(
    clearml_access_key=data_model_reg_cfg['clearml_access_key'],
    clearml_secret_key=data_model_reg_cfg['clearml_secret_key'],
    clearml_host=data_model_reg_cfg['clearml_url'],
    s3_access_key=data_model_reg_cfg['CEPH_ACCESS_KEY'],
    s3_secret_key=data_model_reg_cfg['CEPH_SECRET_KEY'],
    s3_endpoint_url=data_model_reg_cfg['CEPH_ENDPOINT'],
    dataset_name=config["dataset_config"]["source"],
    absolute_path=Path(__file__).parent/"dataset",
    user_name=data_model_reg_cfg['clearml_username']
)

# absolute_path = Path(__file__).parent / "dataset" / cfg["dataset_config"]["source"]

# files = list(absolute_path.rglob("*.[jc][so][nv]*"))  # matches .json or .csv

# # TODO the .json and .csv is not going to work needs refinement
# # Or, more explicitly: 
# files = [f for f in absolute_path.rglob("*") if f.suffix in [".json", ".csv"]]

# # Print absolute paths
# for file_path in files:
#     print(file_path.resolve())
#     file_path = file_path.resolve()

# # Connect hyperparameters and other configurations to the ClearML task


# # config["dataset_config"]["source"] = "/home/dario/mlops/datasets/medical_qa/"
# # config["dataset_config"]["source"] = '/home/dario/mlops/datasets/sample_instruction.json'

# cfg["dataset_config"]["source"] = file_path


trainer = AutoTrainer(config=config)

trainer.run()

if config["trainer_config"]["save_model"] is not None:
    local_model_id = manager.add_model(
        source_type="local",
        source_path="model/",
        model_name = model_reg + "_" + str(int(time.time())),
        code_path="." , # ← Replace with the path to your model.py if you have it
    )

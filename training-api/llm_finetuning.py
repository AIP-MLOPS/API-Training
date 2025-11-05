import os
from dotenv import load_dotenv
import time
from pathlib import Path

import torch
from clearml import Task
from transformers import TrainerCallback

from ml_trainer import AutoTrainer
from aipmodel.model_registry import MLOpsManager
from data.sdk.download_sdk import s3_download

torch._dynamo.config.disable = True

# import the torch callback for checkpointing
# import os
# import shutil

#‍‍‍‍ let's plan this out
# Three parts: one data another is the model and the new one is the retrying & resuming
# for the model it's simple you load and save it then give the path? or if it's retrying you do the resume_checkpoint
# then another thing is 
# --------- ClearML task initialization --------
task = Task.init(
    project_name="API training",  # Name of the ClearML project
    task_name=f"API Training",  # Name of the task
    task_type=Task.TaskTypes.optimizer,  # Type of the task (could also be "training", "testing", etc.)
    reuse_last_task_id=False  # Whether to reuse the last task ID (set to False for a new task each time)
)
## ====================== Data Registry =========================
load_dotenv()

data_model_reg_cfg= {
    #ceph related
    'CEPH_ENDPOINT': 'http://144.172.105.98:7000',
    'CEPH_ACCESS_KEY': '3386LN5KA2OFQXPTYM9S',
    'CEPH_SECRET_KEY': 'AALvi6KexAeSNCsOMRqDHTRf10BQzNyy5BQnGIfO',
    'CEPH_BUCKET': 'datauserv2-bucket-bucket',

    #clearml
    'clearml_url': 'http://144.172.105.98:30003',
    'clearml_access_key': '8113C94C6A387E90477E58B89CCE0547',
    'clearml_secret_key': 'C60E7BD316A59D867D083BABD50B161EF2BFB8BDAADD59935978E546009F923E',
    'clearml_username': 'datauserv2',
}
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

task.connect(data_model_reg_cfg, name='model_data_cfg')

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

class PrintSaveDirCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        import os
        print(f"\n[Callback] Model saved to: {args.output_dir}")
        print("[Callback] Files inside:")
        for f in os.listdir(args.output_dir):
            print("  -", f)

        models =  manager.list_models()

        if model_id in [m['id'] for m in models]: 
            print(f"Model with ID {model_id} already exists. Deleting the old model before adding the new one.")
            manager.delete_model(model_id=local_model_id)

        model_id =  manager.add_model(
            source_type="local",
            source_path="model/",
            model_name = model_reg + "_" + str(int(time.time())),
        )
        print (f"[Callback] Model uploaded to registry with ID: {model_id}\n")
        


config = {
    "task": "llm_finetuning",
    # "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct_1762009294_1762010845",

    # -----------------------------
    # DATASET CONFIG
    # -----------------------------
    "system_prompt": "You are a helpful assistant.",
    "dataset_config": {
        "source": "medical_qaa",
        "format_fn": None,
        # "format_fn": "default",
        "test_size": None,
    },

    # -----------------------------
    # TRAINER CONFIG
    # -----------------------------
    "trainer_config": {
        "dataset_text_field": "text",
        "batch_size": 2, # *
        # "epochs": 1, # *
        "epochs": 1, # *
        "learning_rate": 1e-4, # *

        

        "optim": "adamw_8bit",
        "save_strategy": "epoch",
        "save_steps": 0.5,
        "save_total_limit": 1,
        "output_dir": "./model",
        "resume_from_checkpoint": None,
        "callbacks": None,

        "load_model": True,  # set to True to load model from model registry
        "save_model": None,  # set to True to save model to model registry
    },
}
task.connect(config)

print(config)

model_reg = config["model_name"]

# --------------     to load model -----------------
if config["trainer_config"]["load_model"] is not None: 
    model_id = manager.get_model_id_by_name(model_reg)
    print(manager.ceph.is_folder("models/eba075dfabed4f7fbecbfeb7e54871ca/"))
    key = "models/eba075dfabed4f7fbecbfeb7e54871ca/"
    contents = manager.ceph.check_if_exists(key)
    result = bool(contents) and any(obj["Key"] != key for obj in contents)
    print("CONTENTS:", contents)
    print("RESULTS:", result)
    print("s3 client information:", manager.CEPH_ENDPOINT_URL, manager.CEPH_USER_BUCKET, manager.CEPH_ADMIN_ACCESS_KEY, manager.CEPH_ADMIN_SECRET_KEY)
    os.makedirs("loaded_model", exist_ok=True)

    manager.get_model(
        model_name= model_reg,  # or any valid model ID
        local_dest="."
    )
    # model_dir = f'./loaded_model/{model_id}/'
    model_dir = f'./{model_id}/'
    # model_dir = "loaded_model/model_files/"

    # Find the first folder inside model_dir
    subfolders = [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, f))]

    if not subfolders:
        print(f"No checkpoint folders found in {model_dir}")

    # You can choose the first one or specify logic (e.g., latest modified)

    if subfolders:
        checkpoint_folder = subfolders[0]  # or sorted(subfolders)[-1] for the last alphabetically
        config["model_name"] = f'./{model_id}/{checkpoint_folder}'
        print(f"Checkpoint folder found: {checkpoint_folder}")
        print(f"Model path set to: {config['model_name']}")
    else:
        config["model_name"] = f'./{model_id}/'
        print(f"Model path set to: {config['model_name']}")

    # Set the config
    # config["model_name"] = f'./{model_id}/{checkpoint_folder}'
    # config["model_name"] = f'loaded_model/{model_id}/{checkpoint_folder}'

    # config["model_name"] = f'loaded_model/{model_id}/checkpoint-1'  
    


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

absolute_path = Path(__file__).parent / "dataset" / config["dataset_config"]["source"]

files = list(absolute_path.rglob("*.[jc][so][nv]*"))  # matches .json or .csv

# Or, more explicitly:
files = [f for f in absolute_path.rglob("*") if f.suffix in [".json", ".csv"]]

# Print absolute paths
for file_path in files:
    print(file_path.resolve())
    file_path = file_path.resolve()

# Connect hyperparameters and other configurations to the ClearML task


# config["dataset_config"]["source"] = "/home/dario/mlops/datasets/medical_qa/"
# config["dataset_config"]["source"] = '/home/dario/mlops/datasets/sample_instruction.json'

config["dataset_config"]["source"] = file_path


trainer = AutoTrainer(config=config)

trainer.run()

if config["trainer_config"]["save_model"] is not None:
    local_model_id = manager.add_model(
        source_type="local",
        source_path="model/",
        model_name = model_reg + "_" + str(int(time.time())),
    )

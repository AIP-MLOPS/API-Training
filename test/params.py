
import os
from dotenv import load_dotenv

from clearml import Task


load_dotenv()

# ---------- Variables -------------
dataset='cifar-10'
epochs = 1
batch_size = 64
split_ratio = 0.8
lr = 0.01
save_model = False
load_model = True
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
        "source": sources,
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

print(cfg)


print(os.environ["CLEARML_API_ACCESS_KEY"])
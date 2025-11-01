
import os
# from dotenv import load_dotenv

from clearml import Task


# load_dotenv()

# ---------- Variables -------------

# --------- ClearML task initialization --------
task = Task.init(
    project_name="API training",  # Name of the ClearML project
    task_name=f" API Training",  # Name of the task
    task_type=Task.TaskTypes.optimizer,  # Type of the task (could also be "training", "testing", etc.)
    reuse_last_task_id=False  # Whether to reuse the last task ID (set to False for a new task each time)
)


cfg= {
    "task_name" : "test_task",
    "model_config" : {
        "model_name": "test_model",
        "model_params": "test_params",
    }
    

}


# Connect hyperparameters and other configurations to the ClearML task
task.connect(cfg)
user_config = {
    'ceph': 'ceph_data'
}
task.connect(user_config, name= "user")

# dataset handling


print(cfg)
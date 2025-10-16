from clearml import Task

task = Task.init(
    project_name="API training",
    task_name= "API Training",  
    task_type=Task.TaskTypes.optimizer, 
    reuse_last_task_id=False  
)

cfg = {
    # Training Params
    "task": "default",
    "batch_size": "default",
    "split_ratio": "default",
    "lr": "default",
    "epochs": "default",
    "num_classes": "default",
}
task.connect(cfg)

print(cfg)
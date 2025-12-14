from clearml import Task
from clearml.automation import HyperParameterOptimizer, UniformIntegerParameterRange, DiscreteParameterRange
from clearml.automation.optuna import OptimizerOptuna
from datetime import datetime

task = Task.init(
    project_name="API training",  # Name of the ClearML project
    task_name=f"API HPO",  # Name of the task
    # task_type=Task.TaskTypes.optimizer,  # Type of the task (could also be "training", "testing", etc.)
    reuse_last_task_id=False  # Whether to reuse the last task ID (set to False for a new task each time)
)

training_config = {
    "task_id" : 'default',
    "queue_name": 'queue',
    "total_max_jobs": 10,
    "metric": 'loss'
}

hpo_config = {
    "lr" : None,
    "epochs":None,
    'batch_size':None,
    "weight_decay": None,

}

task.connect(training_config)
task.connect(hpo_config)

# Create optimizer
# hyper_parameters = []
# hyper_parameters.append(DiscreteParameterRange('General/trainer_config/lr',values=[0.0001,0.0005,0.001]))
# # if hpo_config.get('lr',None): 
# if hpo_config.get('epochs',None): 
#     hyper_parameters.append(DiscreteParameterRange('General/trainer_config/epochs',values=[3,8,10]))
# if hpo_config.get('batch_size',None): 
#     hyper_parameters.append(DiscreteParameterRange('General/dataset_config/batch_size',values=[2,4]))

# Create optimizer
hyper_parameters = []
print(f"HPO config is: {hpo_config}")
# Case 1: If *all three* are None → only append LR search
if (
    hpo_config.get("lr") is None and
    hpo_config.get("epochs") is None and
    hpo_config.get("batch_size") is None and 
    hpo_config.get("weight_decay") is None
):
    print("No Value was detected for `lr`/`epochs`/`batch_size`")
    hyper_parameters.append(
        DiscreteParameterRange(
            'General/trainer_config/lr',
            values=[0.0001, 0.0005, 0.001]
        )
    )
else:
    # Case 2: Append only params that are NOT None
    if hpo_config.get("lr") is not None:
        hyper_parameters.append(
            DiscreteParameterRange(
                'General/trainer_config/lr',
                values=hpo_config['lr']
            )
        )

    if hpo_config.get("epochs") is not None:
        hyper_parameters.append(
            DiscreteParameterRange(
                'General/trainer_config/epochs',
                values=hpo_config['epochs']
            )
        )

    if hpo_config.get("batch_size") is not None:
        hyper_parameters.append(
            DiscreteParameterRange(
                'General/dataset_config/batch_size',
                values=hpo_config['batch_size']
            )
        )
        
    if hpo_config.get("weight_decay") is not None:
        hyper_parameters.append(
            DiscreteParameterRange(
                'General/trainer_config/weight_decay',
                values=hpo_config['weight_decay']
            )
        )
print(f"hyper_parameters for HPO are: {hyper_parameters}")

optimizer = HyperParameterOptimizer(
    base_task_id=training_config["task_id"],
    hyper_parameters=hyper_parameters, 
    objective_metric_title=training_config["metric"],
    objective_metric_series='Validation',
    objective_metric_sign='max',
    max_number_of_concurrent_tasks=1,
    optimizer_class=OptimizerOptuna,
    execution_queue=training_config["queue_name"],
    total_max_jobs=training_config["total_max_jobs"],
    min_iteration_per_job=0,
    max_iteration_per_job=0,
)

optimizer.set_report_period(0.1)
optimizer.start()
optimizer.wait()

top_experiments = optimizer.get_top_experiments(top_k=3)
print("Top experiments:", [t.id for t in top_experiments])
optimizer.stop()
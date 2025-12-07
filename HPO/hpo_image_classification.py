from clearml import Task
from clearml.automation import HyperParameterOptimizer, UniformIntegerParameterRange, DiscreteParameterRange
from clearml.automation.optuna import OptimizerOptuna
from datetime import datetime


training_config = {
    "task_id" : 'default',
    "queue_name": 'default',
    "total_max_jobs": '10',
}

hpo_config = {}
task.connect(training_config)
task.connect(hpo_config)
# Create optimizer
hyper_parameters = []
if hpo_config accuracy
    h append UniformIntegerParameterRange('General/n_estimators', min_value=50, max_value=200, step_size=50),
hyper_parameters = [
    UniformIntegerParameterRange('General/n_estimators', min_value=50, max_value=200, step_size=50),
    DiscreteParameterRange('General/max_depth', values=[2, 3, 5, 7]),
]
optimizer = (
    base_task_id=training_config["task_id"],
    hyper_parameters=[
        UniformIntegerParameterRange('General/n_estimators', min_value=50, max_value=200, step_size=50),
        DiscreteParameterRange('General/max_depth', values=[2, 3, 5, 7]),
    ],
    objective_metric_title='accuracy',
    objective_metric_series='test',
    objective_metric_sign='max',
    max_number_of_concurrent_tasks=2,
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
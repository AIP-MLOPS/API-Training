import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import yaml

from clearml import Task

from data.sdk.download_sdk import s3_download
from aipmodel.model_registry import MLOpsManager

# Initialize ClearML Task
task = Task.init(
    project_name='distributed-training',
    task_name='distributed-finetuning',
)

load_dotenv()

print("\n[STEP 1] Initialize MLOps Manager")
data_model_reg_cfg= {
    'clearml_username': 'default',
    'token': 'default'
}
config = {
    # "task": "distributed_training_llm_finetuning",
    "model_name": "qwen2.5-0.5b-base",

    # -----------------------------
    # DATASET CONFIG
    # -----------------------------
    "system_prompt": "You are a helpful assistant.",
    "dataset_config": {
        "source": "medical_qaa",
        # "format_fn": None,
        # "format_fn": "default",
        # "test_size": None,
    },

    # -----------------------------
    # TRAINER CONFIG
    # -----------------------------
    "trainer_config": {
        # "dataset_text_field": "text",
        "batch_size": 2, # *
        # "epochs": 1, # *
        "epochs": 1.0, # *
        "learning_rate": 1e-4, # *
        "weight_decay": 0.01,
        
        # "save_steps": 0.5,
        # "save_strategy": "epoch",
        # "log_callbacks": [llm_logger],
        

        # "optim": "adamw_8bit",
        # "save_total_limit": 1,
        # "output_dir": "./model",
        # "resume_from_checkpoint": None,
        # "callbacks": [PrintSaveDirCallback()],

        # "load_model": None,  # set to True to load model from model registry
        # "save_model": None,  # set to True to save model to model registry
    },
}


manager = MLOpsManager(
    user_token=data_model_reg_cfg['token'],
    # CLEARML_API_HOST=os.getenv("CLEARML_API_HOST"),
    # CEPH_ENDPOINT_URL=os.getenv("CEPH_ENDPOINT_URL"),
    # USER_MANAGEMENT_API=os.getenv("USER_MANAGEMENT_API"),
)


# print("\n[STEP 3] Get Model Info")
# manager.get_model_info("qwen2.5-0.5b-base")

print("Download Model")
result = manager.get_model(model_name=config["model_name"], local_dest="./model/")
print(f"Model download result: {result}")
print("Model download completed successfully!")

# Find the actual model path
print("Finding model configuration...")
find_result = subprocess.run(
    ["find", "./model/", "-name", "config.json", "-type", "f"],
    capture_output=True, text=True
)
if find_result.stdout:
    model_path = find_result.stdout.strip().split('\n')[0]
    model_dir = str(Path(model_path).parent)
    print(f"Found model at: {model_dir}")
else:
    print("WARNING: No config.json found in model directory")
    model_dir = "./model/"

# Download dataset
print("Download Dataset using Data Layer SDK")
dataset_path = None
dataset_dir = Path("./dataset")
dataset_dir.mkdir(parents=True, exist_ok=True)

print("Downloading dataset: mshojaei_mini_v1")
dataset_object = s3_download(
    dataset_name=config["dataset_config"]["source"],
    absolute_path=dataset_dir,
    user_token=data_model_reg_cfg['token'],
    user_management_url=os.getenv("USER_MANAGEMENT_API"),
    clearml_api_host=os.getenv("CLEARML_API_HOST"),
    s3_endpoint_url=os.getenv("CEPH_ENDPOINT_URL"),
    dataset_type="text_generation",
)

print("✓ Dataset download completed successfully!")

import os as os_module
if os_module.path.exists(dataset_dir):
    items = os_module.listdir(dataset_dir)
    print(f"Dataset contents ({len(items)} items): {items[:10]}...")
    
    for item in items:
        item_path = dataset_dir / item
        if item.endswith(('.jsonl', '.json', '.parquet', '.csv')):
            dataset_path = str(item_path)
            print(f"✓ Found dataset file: {dataset_path}")
            break
    
    if not dataset_path:
        find_data_result = subprocess.run(
            ["find", str(dataset_dir), "-type", "f", "-name", "*.jsonl", "-o", "-name", "*.json", "-o", "-name", "*.parquet"],
            capture_output=True, text=True
        )
        if find_data_result.stdout.strip():
            dataset_path = find_data_result.stdout.strip().split('\n')[0]
            print(f"✓ Found dataset file (recursive search): {dataset_path}")
        else:
            dataset_path = str(dataset_dir)
            print(f"⚠ No specific data file found, using directory: {dataset_path}")
    
# Update config with actual paths
config_updates = {
    'base_model': model_dir,
    'datasets': [{'path': dataset_path, 'type': 'completion'}]
}

# Connect configuration to ClearML
task.connect(config_updates)

# Create config.yml with updated paths
config = {
    'base_model': model_dir,
    'model_type': 'AutoModelForCausalLM',
    'tokenizer_type': 'AutoTokenizer',
    'load_in_8bit': False,
    'load_in_4bit': True,
    'strict': False,
    'datasets': [{
        'path': dataset_path,
        'type': 'completion',
        'format_fn': 'default',
        'field_instruction': 'instruction',
        'field_output': 'response'
    }],
    'dataset_prepared_path': 'last_run_prepared',
    'val_set_size': 0.05,
    'output_dir': './outputs',
    'sequence_len': 512,
    'sample_packing': True,
    'pad_to_sequence_len': True,
    'adapter': 'qlora',
    'lora_r': 8,
    'lora_alpha': 16,
    'lora_dropout': 0.05,
    'lora_target_modules': ['q_proj', 'v_proj'],
    'wandb_project': None,
    'gradient_accumulation_steps': 4,
    'micro_batch_size': 1,
    'num_epochs': 1,
    'optimizer': 'adamw_bnb_8bit',
    'lr_scheduler': 'cosine',
    'learning_rate': 0.0002,
    'train_on_inputs': False,
    'group_by_length': False,
    'bf16': 'auto',
    'fp16': None,
    'tf32': False,
    'gradient_checkpointing': True,
    'logging_steps': 1,
    'warmup_steps': 10,
    'evals_per_epoch': 4,
    'saves_per_epoch': 1,
    'weight_decay': 0.0
}

with open('config.yml', 'w') as f:
    yaml.dump(config, f)

print("\n[STEP 7] Starting Axolotl Training")
result = subprocess.run(
    ['accelerate', 'launch', '--num_processes=1', '-m', 'axolotl.cli.train', 'config.yml'],
    check=True
)

print("Training completed!")
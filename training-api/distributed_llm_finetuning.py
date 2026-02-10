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
    'token': 'eyJhbGciOiJSUzI1NiIsImtpZCI6InNoYXJpZl9jZXJ0IiwidHlwIjoiSldUIn0.eyJvd25lciI6Im9yZ2FuaXphdGlvbl9zaGFyaWYiLCJuYW1lIjoiZGVwbG95bWVudCIsImNyZWF0ZWRUaW1lIjoiMjAyNi0wMi0wN1QyMDowNTowM1oiLCJ1cGRhdGVkVGltZSI6IiIsImRlbGV0ZWRUaW1lIjoiIiwiaWQiOiIyYzQzYmYyMC1kNTFiLTQ1ZGUtYWIxOS0xY2VlZDU2NWM3ZTciLCJ0eXBlIjoibm9ybWFsLXVzZXIiLCJwYXNzd29yZCI6IiIsInBhc3N3b3JkU2FsdCI6IiIsInBhc3N3b3JkVHlwZSI6InBsYWluIiwiZGlzcGxheU5hbWUiOiIiLCJmaXJzdE5hbWUiOiIiLCJsYXN0TmFtZSI6IiIsImF2YXRhciI6Imh0dHBzOi8vd2ViLnJheWVuYWkuaXIvd3AtY29udGVudC91cGxvYWRzLzIwMjUvMTEvYS0xLnBuZyIsImF2YXRhclR5cGUiOiIiLCJwZXJtYW5lbnRBdmF0YXIiOiIiLCJlbWFpbCI6ImRlcGxveW1lbnRAc2hhcmlmLmVkdSIsImVtYWlsVmVyaWZpZWQiOmZhbHNlLCJwaG9uZSI6IiIsImNvdW50cnlDb2RlIjoiIiwicmVnaW9uIjoiIiwibG9jYXRpb24iOiIiLCJhZGRyZXNzIjpbXSwiYWZmaWxpYXRpb24iOiIiLCJ0aXRsZSI6IiIsImlkQ2FyZFR5cGUiOiIiLCJpZENhcmQiOiIiLCJob21lcGFnZSI6IiIsImJpbyI6IiIsImxhbmd1YWdlIjoiIiwiZ2VuZGVyIjoiIiwiYmlydGhkYXkiOiIiLCJlZHVjYXRpb24iOiIiLCJzY29yZSI6MCwia2FybWEiOjAsInJhbmtpbmciOjE3MiwiaXNEZWZhdWx0QXZhdGFyIjpmYWxzZSwiaXNPbmxpbmUiOmZhbHNlLCJpc0FkbWluIjpmYWxzZSwiaXNGb3JiaWRkZW4iOmZhbHNlLCJpc0RlbGV0ZWQiOmZhbHNlLCJzaWdudXBBcHBsaWNhdGlvbiI6ImFwcGxpY2F0aW9uX3BhbmVsIiwiaGFzaCI6IiIsInByZUhhc2giOiIiLCJhY2Nlc3NLZXkiOiIiLCJhY2Nlc3NTZWNyZXQiOiIiLCJnaXRodWIiOiIiLCJnb29nbGUiOiIiLCJxcSI6IiIsIndlY2hhdCI6IiIsImZhY2Vib29rIjoiIiwiZGluZ3RhbGsiOiIiLCJ3ZWlibyI6IiIsImdpdGVlIjoiIiwibGlua2VkaW4iOiIiLCJ3ZWNvbSI6IiIsImxhcmsiOiIiLCJnaXRsYWIiOiIiLCJjcmVhdGVkSXAiOiIiLCJsYXN0U2lnbmluVGltZSI6IiIsImxhc3RTaWduaW5JcCI6IiIsInByZWZlcnJlZE1mYVR5cGUiOiIiLCJyZWNvdmVyeUNvZGVzIjpudWxsLCJ0b3RwU2VjcmV0IjoiIiwibWZhUGhvbmVFbmFibGVkIjpmYWxzZSwibWZhRW1haWxFbmFibGVkIjpmYWxzZSwibGRhcCI6IiIsInByb3BlcnRpZXMiOnt9LCJyb2xlcyI6W10sInBlcm1pc3Npb25zIjpbXSwiZ3JvdXBzIjpbXSwibGFzdFNpZ25pbldyb25nVGltZSI6IiIsInNpZ25pbldyb25nVGltZXMiOjAsIm1hbmFnZWRBY2NvdW50cyI6bnVsbCwidG9rZW5UeXBlIjoiYWNjZXNzLXRva2VuIiwidGFnIjoiIiwic2NvcGUiOiJwcm9maWxlIiwiYXpwIjoiYzYzYzQ4ZmZiMTgwMmVjYWFkYWQiLCJpc3MiOiJodHRwczovL2lhbS5yYXllbmFpLmlyIiwic3ViIjoiMmM0M2JmMjAtZDUxYi00NWRlLWFiMTktMWNlZWQ1NjVjN2U3IiwiYXVkIjpbImM2M2M0OGZmYjE4MDJlY2FhZGFkIl0sImV4cCI6MTc3MTIzMTIwNywibmJmIjoxNzcwNjI2NDA3LCJpYXQiOjE3NzA2MjY0MDcsImp0aSI6ImFkbWluLzQ2ZGUyMjAwLTU2YjItNDcwNy05ZWZmLTMyYjU5YWI3Njg0OCJ9.pdl2qoDT4nf_s4_hI-6riPExAjuBmPduanyBMAbXhvmiUdEb0jR10-m-rRH_-ZfGGUHmI62wkqtoTO_qISp80lP-NnuzEw6ZaZv2EKGzSWLP3InPZiwzddl_gSrPkb9RYH0EFcu60n3w75FY2SSZh8Kj4mLrH7ZkSkRXFUMvgqmvfyIamCoF7yc3kUyApv_lOxNxlTaTfvI2yeHeM4LgheLRmKMJ8GT8tgwwY5XT6kwkyzaZnNc5p-rSkPwEncdhc6eJVRznZ_0DGaSSNWpNWF5DKDGIHuLNZ6cRaZxu_npREM3m0ZFKE2xiUoeHvYJ5A6RGIeDAGEpGQZb1HfSQ13E0Bz_P2r-FDh6qA-54C0BgbOAoLx1JrUlxTqCT9DCdgioesEzW_pkR0cR1dvvXWtx04SiSWnK7wQj5WaED8TDuqa5ONmK3QQmUmcFLpzwWe0exfevDaptOJbOUK64v3TBefMV2rH0XvgnuhFt3jqYHq93hgirHC5a7RGvFs30q7IzqEgfpHAcOVwtm1fMWnUWlCZ8ielAFAMy57mkn3tkaqjrkWxMZk_bfs1brOfdGrvdJREnFbaaZjQkdKl6wXGWY1Neo7zbzgkEiRvtUH5pK4t1M4umceTuVXAoQ6nf9PPRIxi3e2O08KSQKFK3LhoxPCfFYi_Fa2ZHzfMiWU3E'
}

config = {
    # "task": "distributed_training_llm_finetuning",
    "model_name": "qwen2.5-0.5b-base",

    # -----------------------------
    # DATASET CONFIG
    # -----------------------------
    "system_prompt": "You are a helpful assistant.",
    "dataset_config": {
        "source": "mshojaei_mini_v1",
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
    "gpu_num_proc" : "2"
}

task.connect(data_model_reg_cfg, name='model_data_cfg')
task.connect(config)

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
# dataset_object = s3_download(
#     dataset_name=config["dataset_config"]["source"],
#     absolute_path=dataset_dir,
#     user_token=data_model_reg_cfg['token'],
#     user_management_url=os.getenv("USER_MANAGEMENT_API"),
#     clearml_api_host=os.getenv("CLEARML_API_HOST"),
#     s3_endpoint_url=os.getenv("CEPH_ENDPOINT_URL"),
#     dataset_type="text_generation",
# )

dataset_object = s3_download(
        dataset_name=config["dataset_config"]["source"],
        absolute_path=Path(__file__).parent/"dataset",
        token=data_model_reg_cfg['token'],
        user_management_url=os.getenv("USER_MANAGEMENT_API"),
        clearml_api_host=os.getenv("CLEARML_API_HOST"),
        s3_endpoint_url=os.getenv("CEPH_ENDPOINT_URL"),
        dataset_type="text_generation",
        # user_name=data_model_reg_cfg['clearml_username'],
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
axolotl_config = {
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
    yaml.dump(axolotl_config, f)

print("\n[STEP 7] Starting Axolotl Training")
# result = subprocess.run(
#     ['accelerate', 'launch', f'--num_processes={config["gpu_num_proc"]}', '-m', 'axolotl.cli.train', 'config.yml'],
#     check=True
# )
# result = subprocess.run(
#     ['python', '-m', 'accelerate.commands.launch', f'--num_processes={config["gpu_num_proc"]}', '-m', 'axolotl.cli.train', 'config.yml'],
#     check=True
# )
print("\n[STEP 7] Starting Axolotl Training")
result = subprocess.run(
    ['bash', '-c', f'source /opt/mlops/bin/activate && accelerate launch --num_processes={config["gpu_num_proc"]} -m axolotl.cli.train config.yml'],
    check=True
)

print("Training completed!")
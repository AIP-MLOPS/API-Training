from ml_trainer import AutoTrainer
from transformers import TrainerCallback
# import the torch callback for checkpointing
# import os
# import shutil

‍‍‍‍# let's plan this out
# Three parts: one data another is the model and the new one is the retrying & resuming
# for the model it's simple you load and save it then give the path? or if it's retrying you do the resume_checkpoint
# then another thing is 

class PrintSaveDirCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        import os
        print(f"\n[Callback] Model saved to: {args.output_dir}")
        print("[Callback] Files inside:")
        for f in os.listdir(args.output_dir):
            print("  -", f)


class PrintSaveDirCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        # Print original folder contents
        print(f"\n[Callback] Model saved to: {args.output_dir}")
        print("[Callback] Files inside:")
        for f in os.listdir(args.output_dir):
            print("  -", f)
        
        # Define destination folder
        dest_dir = "saved_checkpoint"
        os.makedirs(dest_dir, exist_ok=True)
        
        # Copy all files from output_dir to dest_dir
        for item in os.listdir(args.output_dir):
            s = os.path.join(args.output_dir, item)
            d = os.path.join(dest_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        
        print(f"[Callback] All files copied to: {dest_dir}")

full_config = {
    # -----------------------------
    # MODEL CONFIG
    # -----------------------------
    "task": "llm_finetuning",
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct", 

    # -----------------------------
    # DATASET CONFIG
    # -----------------------------
    "dataset_config": {
        "source": f"/home/dario/mlops/datasets/sample_instruction.json",
        "format_fn": "default",               
        "test_size": 0.1,
    },
    "system_prompt": "You are a helpful assistant.",

    # -----------------------------
    # TRAINER CONFIG
    # -----------------------------
    "trainer_config": {
        "epochs": 100,
        "resume_from_checkpoint": "/home/dario/mlops/trainer_save/output/checkpoint-60",
    }
}

trainer = AutoTrainer(config=full_config)

trainer.run()

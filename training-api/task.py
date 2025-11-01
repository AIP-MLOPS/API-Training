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




trainer = AutoTrainer(config=full_config)

trainer.run()

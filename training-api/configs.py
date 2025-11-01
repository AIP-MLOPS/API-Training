unified_config = {
    "llm_finetuning": {
        # -----------------------------
        # MODEL CONFIG
        # -----------------------------
        "task": "llm_finetuning",
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",

        # -----------------------------
        # DATASET CONFIG
        # -----------------------------
        "system_prompt": "You are a helpful assistant.",
        "dataset_config": {
            "source": "path/to/dataset",
            "format_fn": None,
            "test_size": 0.1,
        },

        # -----------------------------
        # TRAINER CONFIG
        # -----------------------------
        "trainer_config": {
            "dataset_text_field": "text",
            "batch_size": 2, # *
            "epochs": 1, # *
            "learning_rate": 1e-4, # *
            "optim": "adamw_8bit",

            "save_strategy": "epoch",
            "save_steps": 0.5,
            "save_total_limit": 1,
            "output_dir": None,
            "resume_from_checkpoint": None,
            "callbacks": None,
        },
    },

    "image_classification": {
        "task": "image_classification",
        "dataset_config": {
            "source": "path/to/dataset",
            "split_ratio": 0.2,
            "batch_size": 32,
        },
        "model_config": {
            "num_classes": 10,  # !Required
            "input_channels": 3,
            "input_size": (32, 32),
            "type": "timm",            # "timm" for pretrained models, or omit for custom CNN
            "name": "resnet18",        # any supported TIMM model
            "pretrained": True
        },
        "trainer_config": {
            "lr": 1e-2,
            "load_model": "path/to/load/",
            "save_model": "path/to/save/",
            "epochs": 20,
            "device": "cuda",
            "checkpoint_path": None,
            "callbacks": None,
            "resume_from_checkpoint": None,
        },
    },

    "timeseries": {
        "task": "timeseries",
        "dataset_config": {
            "source": "path/to/dataset",
            "split_ratio": 0.2,
            "batch_size": 32,
            "seq_len": 1,
            "pred_len": 1,
            # "class_num": "num",
            "target_column": "last",
        },
        "model_config": {
            "type": "tslib",
            "name": "Autoformer",
            "task_name": "long_term_forecast",
            "hidden_size": 64,
            "input_size": 1,
            "output_size": 1,
            "batch_size": 32,
        },
        "trainer_config": {
            "epochs": 50,
            "device": "cuda",
            "checkpoint_path": None,
            "callbacks": None,
            "resume_from_checkpoint": None,
            "lr": 0.001,
            "load_model": "path/to/load/",
            "save_model": "path/to/save/",
        },
    },
}
    
image_classification = {
    "task": "image_classification",
    "dataset_config": {
        "source": "path/to/dataset",
        "split_ratio": 0.2,
        "batch_size": 32,
    },
    "model_config": {
        "num_classes": 10, # !Required
        "input_channels": 3,
        "input_size": (32, 32),
        "type": "timm",            # "timm" for pretrained models, or omit for custom CNN
        "name": "resnet18",        # any supported TIMM model
        "pretrained": True
    },



    "model_dir": "/path/to/load/",
    "trainer_config": {
        "lr":1e-2,
        # "loss_fn": None,

        "load_model": "path/to/load/",
        "save_model": "path/to/save/",

        "epochs": 20,
        "device": "cuda",
        "checkpoint_path": None,

        # "checkpoint": True,
        "callbacks": None,
        "resume_from_checkpoint": None,
    },
}

timeseries=config = {
    "task": "timeseries",

    "dataset_config": {
        "source": "path/to/dataset",
        "split_ratio": 0.2,
        "batch_size": 32,
        "seq_len": 1,
        "pred_len": 1,
        # "class_num": "num",
        "target_column": "last",
    },
    "model_config": {
        "type": "tslib",
        "name": "Autoformer",
        "task_name": "long_term_forecast",

        "hidden_size": 64,
        "input_size": 1,
        "output_size": 1,
        "batch_size": 32,

    },
    
    "trainer_config": {
        "epochs": 50,
        "device": "cuda",
        "checkpoint_path": None,
        "callbacks": None,
        "resume_from_checkpoint": None,
        "lr": 0.001,

        "load_model": "path/to/load/",
        "save_model": "path/to/save/",
    },
}

"""QLoRA fine-tuning of Mistral-7B-Instruct-v0.2 on the prepared medical Q&A
dataset. Requires a GPU (designed for a free-tier Colab T4, 15-16GB VRAM).

Run (after src/prepare_dataset.py has produced data/processed/*.jsonl):
    python src/finetune.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402


def get_bnb_config():
    import torch
    from transformers import BitsAndBytesConfig

    compute_dtype = getattr(torch, config.BNB_4BIT_COMPUTE_DTYPE)
    return BitsAndBytesConfig(
        load_in_4bit=config.BNB_LOAD_IN_4BIT,
        bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=config.BNB_4BIT_USE_DOUBLE_QUANT,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_base_model_and_tokenizer():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    bnb_config = get_bnb_config()

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # avoids fp16/bf16 overflow issues during training

    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    model.config.use_cache = False
    return model, tokenizer


def build_lora_config():
    from peft import LoraConfig

    return LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias=config.LORA_BIAS,
        task_type=config.LORA_TASK_TYPE,
        target_modules=config.LORA_TARGET_MODULES,
    )


def load_processed_datasets():
    from datasets import load_dataset

    train_ds = load_dataset("json", data_files=config.TRAIN_FILE, split="train")
    val_ds = load_dataset("json", data_files=config.VAL_FILE, split="train")
    return train_ds, val_ds


def main():
    if not os.path.exists(config.TRAIN_FILE):
        raise FileNotFoundError(
            f"{config.TRAIN_FILE} not found. Run `python src/prepare_dataset.py` first."
        )

    from peft import prepare_model_for_kbit_training
    from transformers import TrainingArguments
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = load_base_model_and_tokenizer()
    model = prepare_model_for_kbit_training(model)
    lora_config = build_lora_config()

    train_ds, val_ds = load_processed_datasets()
    print(f"Train examples: {len(train_ds)} | Val examples: {len(val_ds)}")

    training_args = SFTConfig(
        output_dir=config.ADAPTER_DIR,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        optim=config.OPTIM,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        warmup_ratio=config.WARMUP_RATIO,
        weight_decay=config.WEIGHT_DECAY,
        max_grad_norm=config.MAX_GRAD_NORM,
        logging_steps=config.LOGGING_STEPS,
        save_steps=config.SAVE_STEPS,
        bf16=True,
        report_to="tensorboard",
        max_seq_length=config.MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.model.save_pretrained(config.ADAPTER_DIR)
    tokenizer.save_pretrained(config.ADAPTER_DIR)
    print(f"Saved LoRA adapter + tokenizer -> {config.ADAPTER_DIR}")


if __name__ == "__main__":
    main()

# 底座模型预训练

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 屏蔽libiomp5md.dll重复初始化错误
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # 继续关闭HF警告
os.environ["HF_TOKEN"] = "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"

from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments

from my_model import LLMIEForCausalLM, build_student_config, get_model_stats
from tokenizer_utils import load_qwen_tokenizer


DATASET_DIR = "./artifacts/pretrain_dataset"
OUTPUT_DIR = "./artifacts/pretrain_runs"
FINAL_MODEL_DIR = "./artifacts/base_model"


def main():
    tokenizer, tokenizer_source = load_qwen_tokenizer()

    dataset = load_from_disk(DATASET_DIR)
    config = build_student_config(tokenizer)
    model = LLMIEForCausalLM(config)
    model.gradient_checkpointing_enable()

    stats = get_model_stats(model)
    print(f"Loaded tokenizer from {tokenizer_source}")
    print(f"Tokenizer size: {len(tokenizer)}")
    print(f"Student params: {stats.total_params / 1e6:.2f}M")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=3e-4,
        num_train_epochs=1,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=20,
        eval_steps=200,
        save_steps=200,
        save_strategy="steps",
        eval_strategy="steps",
        save_total_limit=2,
        weight_decay=0.1,
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )
    trainer.train()

    Path(FINAL_MODEL_DIR).mkdir(parents=True, exist_ok=True)
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print(f"Base model saved to {FINAL_MODEL_DIR}")


if __name__ == "__main__":
    main()

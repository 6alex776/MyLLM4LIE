# 底座模型预训练

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 屏蔽libiomp5md.dll重复初始化错误
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # 继续关闭HF警告
os.environ["HF_TOKEN"] = "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"

from pathlib import Path

import torch

# RTX 4090 (Ampere) 加速设置
torch.backends.cuda.matmul.allow_tf32 = True  # 启用 TF32 矩阵运算加速
torch.backends.cudnn.allow_tf32 = True        # 启用 cuDNN TF32 加速

from datasets import load_from_disk
from transformers import Trainer, TrainingArguments

from my_model import LLMIEForCausalLM, build_student_config, get_model_stats
from tokenizer_utils import load_qwen_tokenizer


DATASET_DIR = "./artifacts/pretrain_dataset"
OUTPUT_DIR = "./artifacts/pretrain_runs"
FINAL_MODEL_DIR = "./artifacts/base_model"


def find_latest_checkpoint(output_dir: str) -> str:
    """查找最新的有效checkpoint（包含模型权重文件）"""
    checkpoint_dir = Path(output_dir)
    if not checkpoint_dir.exists():
        return None
    
    checkpoint_dirs = []
    for item in checkpoint_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            # 检查是否包含模型权重文件
            model_file = item / "model.safetensors"
            if model_file.exists():
                checkpoint_dirs.append(item)
    
    if not checkpoint_dirs:
        return None
    
    # 按checkpoint编号排序，返回最新的
    checkpoint_dirs.sort(key=lambda x: int(x.name.split("-")[-1]), reverse=True)
    return str(checkpoint_dirs[0])


def main():
    tokenizer, tokenizer_source = load_qwen_tokenizer()
    dataset = load_from_disk(DATASET_DIR)

    # 查找最新的有效checkpoint
    latest_checkpoint = find_latest_checkpoint(OUTPUT_DIR)
    
    if latest_checkpoint is not None:
        print(f"找到checkpoint，从 {latest_checkpoint} 恢复训练")
        model = LLMIEForCausalLM.from_pretrained(
            latest_checkpoint,
            ignore_mismatched_sizes=True,  # 忽略权重形状不匹配（如 tie_word_embeddings 导致的 lm_head 缺失）
        )
    
    if latest_checkpoint is None:
        print("未找到有效checkpoint，从头开始训练")
        config = build_student_config(tokenizer)
        model = LLMIEForCausalLM(config)
    
    # 关闭梯度检查点以提速（显存有余量 14G/24G，关掉可提速 20-30%）
    # model.gradient_checkpointing_enable()

    stats = get_model_stats(model)
    print(f"Loaded tokenizer from {tokenizer_source}")
    print(f"Tokenizer size: {len(tokenizer)}")
    print(f"Student params: {stats.total_params / 1e6:.2f}M")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,        # 加大到16（显存有余量）
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=3,          # 16×3=48，与原来12×4=48等价
        learning_rate=2e-4,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        warmup_steps=200,
        logging_steps=100,
        eval_steps=5000,
        save_steps=2000,
        save_strategy="steps",
        eval_strategy="steps",
        save_total_limit=2,
        weight_decay=0.1,
        fp16=False,
        bf16=True,
        report_to="none",
        dataloader_num_workers=min(4, os.cpu_count() // 2),
        dataloader_prefetch_factor=2,           # 预取数据，减少GPU等待
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )
    
    # 如果有checkpoint，从checkpoint继续训练
    trainer.train(resume_from_checkpoint=latest_checkpoint)

    Path(FINAL_MODEL_DIR).mkdir(parents=True, exist_ok=True)
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print(f"Base model saved to {FINAL_MODEL_DIR}")


if __name__ == "__main__":
    main()

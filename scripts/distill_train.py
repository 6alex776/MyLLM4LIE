# SFT 微调训练（基于自训练底座模型 + LoRA）

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import torch

# 4090 加速设置
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import LLMIEForCausalLM, get_model_stats
from src.tokenizer import load_tokenizer
from src.optimizer import create_optimizer


BASE_MODEL_DIR = "./artifacts/base_model"          # 预训练底座模型
SFT_DATA_PATH = "./artifacts/distill_dataset.jsonl"  # SFT 数据集（JSONL 格式）
OUTPUT_DIR = "./artifacts/sft_runs"                 # 训练中间 checkpoint
FINAL_MODEL_DIR = "./artifacts/sft_model"            # 最终微调模型
MAX_SEQ_LEN = 768                                    # ChatGLM3 中文编码更长，放宽截断

# ========== 优化器配置 ==========
# 设置为 True 使用 Muon（推荐用于 200M 小模型），False 使用 AdamW
USE_MUON = True
# Muon 学习率：SFT 阶段可以比预训练稍小
MUON_LR = 2e-4
# AdamW 学习率（用于 1D 参数，或当 USE_MUON=False 时）
ADAMW_LR = 1e-4
# 权重衰减
WEIGHT_DECAY = 0.01


def build_chat_text(system_prompt: str, instruction: str, user_input: str, assistant_output: str) -> str:
    """构建聊天格式（与预训练底座兼容）"""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}\n\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_output}<|im_end|>"
    )


def preprocess_dataset(tokenizer):
    """加载并 tokenize SFT 数据集"""
    dataset = load_dataset("json", data_files=SFT_DATA_PATH, split="train")
    dataset = dataset.train_test_split(test_size=0.02, seed=42)

    assistant_prefix = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    def tokenize_batch(examples):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for system_prompt, instruction, user_input, assistant_output in zip(
            examples["system"],
            examples["instruction"],
            examples["input"],
            examples["output"],
        ):
            full_text = build_chat_text(system_prompt, instruction, user_input, assistant_output)
            encoded = tokenizer(
                full_text,
                truncation=True,
                max_length=MAX_SEQ_LEN,
                padding=False,              # 不做 padding，交给 DataCollator 动态处理
                return_attention_mask=True,
            )

            # labels：只对 assistant 部分计算 loss，其余置为 -100
            labels = [-100] * len(encoded["input_ids"])
            prefix_start = -1
            token_ids = encoded["input_ids"]
            for index in range(0, len(token_ids) - len(assistant_prefix) + 1):
                if token_ids[index : index + len(assistant_prefix)] == assistant_prefix:
                    prefix_start = index + len(assistant_prefix)
                    break

            if prefix_start != -1:
                for index in range(prefix_start, len(token_ids)):
                    if encoded["attention_mask"][index] == 1:
                        labels[index] = token_ids[index]

            input_ids_list.append(encoded["input_ids"])
            attention_mask_list.append(encoded["attention_mask"])
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    tokenized.set_format("torch")
    return tokenized


def main():
    # 1. 加载分词器
    tokenizer, tokenizer_source = load_tokenizer()
    print(f"分词器来源: {tokenizer_source}")
    print(f"词表大小: {len(tokenizer)}")

    # 2. 加载自训练底座模型
    print(f"\n加载底座模型: {BASE_MODEL_DIR}")
    model = LLMIEForCausalLM.from_pretrained(BASE_MODEL_DIR)
    # ~191M 底座模型不需要梯度检查点
    # model.gradient_checkpointing_enable()

    stats = get_model_stats(model)
    print(f"底座模型参数量: {stats.total_params / 1e6:.2f}M")

    # 3. 配置 LoRA（适配 ~191M 模型）
    lora_config = LoraConfig(
        r=12,               # ~191M 模型适当增大秩
        lora_alpha=24,      # 缩放系数，通常为 r 的 2 倍
        lora_dropout=0.05,  # 轻微 dropout 防止过拟合
        bias="none",        # 不训练 bias
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. 加载数据集
    dataset = preprocess_dataset(tokenizer)
    print(f"训练集: {len(dataset['train'])} 条, 验证集: {len(dataset['test'])} 条")

    # 5. 训练参数（4090 24GB）
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,       # 4090 绰绰有余
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,        # 16×2=32 有效 batch
        learning_rate=ADAMW_LR if not USE_MUON else MUON_LR,
        num_train_epochs=5,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        logging_steps=10,
        eval_steps=100,
        save_steps=200,
        save_strategy="steps",
        eval_strategy="steps",
        save_total_limit=2,
        weight_decay=WEIGHT_DECAY,
        fp16=False,
        bf16=True,
        report_to="none",
        dataloader_num_workers=4,             # Linux 服务器可用多进程
        remove_unused_columns=False,
    )

    # 创建自定义优化器（Muon 或 AdamW）
    # 注意：LoRA 参数中，A/B 矩阵是 2D 的，会用 Muon；bias 等 1D 参数用 AdamW
    optimizer = create_optimizer(
        model,
        use_muon=USE_MUON,
        muon_lr=MUON_LR,
        adamw_lr=ADAMW_LR,
        weight_decay=WEIGHT_DECAY,
    )

    # 6. 训练（DataCollatorForSeq2Seq 动态 padding，每 batch 只 pad 到该 batch 最长长度）
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        optimizers=(optimizer, None),
    )
    trainer.train()

    # 7. 合并 LoRA 权重并保存
    final_model = model.merge_and_unload()
    Path(FINAL_MODEL_DIR).mkdir(parents=True, exist_ok=True)
    final_model.save_pretrained(FINAL_MODEL_DIR, safe_serialization=True)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print(f"\nSFT 模型已保存到: {FINAL_MODEL_DIR}")


if __name__ == "__main__":
    main()

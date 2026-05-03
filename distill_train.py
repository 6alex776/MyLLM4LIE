# SFT 微调训练（基于自训练底座模型 + LoRA）

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

from my_model import LLMIEForCausalLM, get_model_stats  # 自定义底座模型
from tokenizer_utils import load_tokenizer


BASE_MODEL_DIR = "./artifacts/base_model"          # 预训练底座模型
SFT_DATA_PATH = "./artifacts/distill_dataset.jsonl"  # SFT 数据集（JSONL 格式）
OUTPUT_DIR = "./artifacts/sft_runs"                 # 训练中间 checkpoint
FINAL_MODEL_DIR = "./artifacts/sft_model"            # 最终微调模型
MAX_SEQ_LEN = 512                                    # 序列截断长度


def build_chat_text(system_prompt: str, instruction: str, user_input: str, assistant_output: str) -> str:
    """构建 Qwen 风格聊天格式（与预训练底座相同）"""
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
    # 118M 小模型不需要梯度检查点，关掉可大幅提速（从 ~26s/it 降到 ~1-2s/it）

    stats = get_model_stats(model)
    print(f"底座模型参数量: {stats.total_params / 1e6:.2f}M")

    # 3. 配置 LoRA（适配 118M 小模型）
    lora_config = LoraConfig(
        r=8,                # 低秩维度，小模型不需要太大
        lora_alpha=16,      # 缩放系数，通常为 r 的 2 倍
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

    # 5. 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,#4,        # 适配 8GB 显存 + 151K 大词表
        per_device_eval_batch_size=8,#4,
        gradient_accumulation_steps=2,#8,        # 4×8=32 有效 batch
        learning_rate=2e-4,
        num_train_epochs=5,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        logging_steps=10,
        eval_steps=100,
        save_steps=200,
        save_strategy="steps",
        eval_strategy="steps",
        save_total_limit=2,
        weight_decay=0.01,
        fp16=False,
        bf16=torch.cuda.is_available(),       # bf16 将 logits 从 4.6GB 降到 2.3GB
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
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

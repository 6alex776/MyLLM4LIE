# 知识蒸馏训练

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

from my_model import LLMIEForCausalLM
from tokenizer_utils import load_qwen_tokenizer


BASE_MODEL_DIR = "./artifacts/base_model"
DISTILL_DATA_PATH = "./artifacts/distill_dataset.jsonl"
OUTPUT_DIR = "./artifacts/distill_runs"
FINAL_MODEL_DIR = "./artifacts/final_student_model"
MAX_SEQ_LEN = 512


def build_chat_text(system_prompt: str, instruction: str, user_input: str, assistant_output: str) -> str:
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}\n\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_output}<|im_end|>"
    )


def preprocess_dataset(tokenizer):
    dataset = load_dataset("json", data_files=DISTILL_DATA_PATH, split="train")
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
                padding="max_length",
                return_attention_mask=True,
            )

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
    tokenizer, tokenizer_source = load_qwen_tokenizer()

    model = LLMIEForCausalLM.from_pretrained(BASE_MODEL_DIR)
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        print(
            f"Resizing token embeddings from {model.get_input_embeddings().weight.shape[0]} "
            f"to {len(tokenizer)} to cover tokenizer special tokens."
        )
        model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    print(f"Loaded tokenizer from {tokenizer_source}")
    model.print_trainable_parameters()

    dataset = preprocess_dataset(tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=20,
        eval_steps=100,
        save_steps=100,
        save_strategy="steps",
        eval_strategy="steps",
        save_total_limit=2,
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

    final_model = model.merge_and_unload()
    Path(FINAL_MODEL_DIR).mkdir(parents=True, exist_ok=True)
    final_model.save_pretrained(FINAL_MODEL_DIR, safe_serialization=True)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print(f"Final student model saved to {FINAL_MODEL_DIR}")


if __name__ == "__main__":
    main()

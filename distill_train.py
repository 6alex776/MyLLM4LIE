# 知识蒸馏训练（中间层 + 输出结合）

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn import MSELoss
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    Trainer, 
    TrainingArguments, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from tokenizer_utils import load_qwen_tokenizer


BASE_MODEL_DIR = "./artifacts/qwen_model/Qwen2.5-1.5B-Instruct"  # 使用 Qwen3.5-0.8B-Base
TEACHER_MODEL_DIR = "./artifacts/qwen_model/Qwen2.5-7B-Instruct"  # 教师模型路径
DISTILL_DATA_PATH = "./artifacts/distill_dataset.jsonl"
OUTPUT_DIR = "./artifacts/distill_runs"
FINAL_MODEL_DIR = "./artifacts/final_student_model"
MAX_SEQ_LEN = 512  # Qwen 默认支持 512 序列长度

# 蒸馏配置
USE_ENHANCED_DISTILLATION = True  # 是否使用增强版蒸馏
OUTPUT_LOSS_WEIGHT = 0.7  # 输出损失权重
INTERMEDIATE_LOSS_WEIGHT = 0.3  # 中间层损失权重
TEMPERATURE = 2.0  # 温度系数

# 量化配置
TEACHER_QUANTIZATION_ENABLED = True  # 教师模型启用量化以节省显存
STUDENT_QUANTIZATION_ENABLED = False  # 学生模型不建议量化（因为需要训练）


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


class EnhancedDistillationTrainer(Trainer):
    """自定义 Trainer，支持中间层蒸馏与输出蒸馏结合"""
    
    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not USE_ENHANCED_DISTILLATION:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        
        # 学生模型前向传播（不需要隐藏状态，节省显存）
        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        
        # 教师模型前向传播（无梯度，不输出隐藏状态避免CPU卸载卡顿）
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        
        # 输出损失（KL散度）- 对齐词汇表维度
        min_vocab_size = min(student_outputs.logits.size(-1), teacher_outputs.logits.size(-1))
        student_logits = student_outputs.logits[..., :min_vocab_size]
        teacher_logits = teacher_outputs.logits[..., :min_vocab_size]
        
        logits_loss = F.kl_div(
            F.log_softmax(student_logits / TEMPERATURE, dim=-1),
            F.softmax(teacher_logits / TEMPERATURE, dim=-1),
            reduction="batchmean"
        ) * (TEMPERATURE * TEMPERATURE)
        
        return (logits_loss, student_outputs) if return_outputs else logits_loss


def main():
    tokenizer, tokenizer_source = load_qwen_tokenizer()

    # 1. 加载学生模型（Qwen3.5-0.8B-Base）
    print("Loading student model...")
    if STUDENT_QUANTIZATION_ENABLED:
        # 对学生模型进行量化（一般不建议，因为需要训练）
        print("Loading student model with quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        student_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_DIR,
            quantization_config=quantization_config,
            device_map="auto",
        )
        print("Student model loaded with 4-bit quantization.")
    else:
        # 对于学生模型，使用标准 FP16 加载，因为需要训练
        student_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_DIR,
            torch_dtype=torch.float16,  # 使用 FP16 加速训练
            device_map="auto",  # 自动分配到可用设备
        )
        print("Student model loaded with FP16 precision.")
    
    # 检查并调整 token embeddings
    if student_model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        print(
            f"Resizing token embeddings from {student_model.get_input_embeddings().weight.shape[0]} "
            f"to {len(tokenizer)} to cover tokenizer special tokens."
        )
        student_model.resize_token_embeddings(len(tokenizer))
    
    student_model.gradient_checkpointing_enable()

    # 2. 加载教师模型（如果启用增强蒸馏）
    teacher_model = None
    enhanced_distillation_enabled = USE_ENHANCED_DISTILLATION  # 保存原始值
    
    if enhanced_distillation_enabled and TEACHER_QUANTIZATION_ENABLED:
        try:
            print("Loading teacher model with quantization...")
            # 使用配置的量化方式加载教师模型以节省显存
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            teacher_model = AutoModelForCausalLM.from_pretrained(
                TEACHER_MODEL_DIR,
                quantization_config=quantization_config,
                device_map="auto",
            )
            print("Teacher model loaded with 8-bit quantization.")
            
            teacher_model.eval()
            print("Teacher model loaded successfully!")
        except:
            # 如果 8-bit 量化失败，尝试 4-bit 量化
            try:
                print("8-bit quantization failed, trying 4-bit...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                # 不再限制 GPU 显存（只算 logits 不占内存）
                teacher_model = AutoModelForCausalLM.from_pretrained(
                    TEACHER_MODEL_DIR,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
                print("Teacher model loaded with 4-bit quantization.")
                
                teacher_model.eval()
                print("Teacher model loaded successfully!")
            except Exception as e:
                print(f"Failed to load teacher model: {e}")
                print("Falling back to standard distillation (output-only).")
                enhanced_distillation_enabled = False
    elif enhanced_distillation_enabled:
         # 不使用量化（当 TEACHER_QUANTIZATION_ENABLED=False 时）
         try:
             print("Loading teacher model...")
             teacher_model = AutoModelForCausalLM.from_pretrained(
                 TEACHER_MODEL_DIR,
                 torch_dtype=torch.float16,
                 device_map="auto",
             )
             teacher_model.eval()
             print("Teacher model loaded successfully!")
         except Exception as e:
             print(f"Failed to load teacher model: {e}")
             print("Falling back to standard distillation (output-only).")
             enhanced_distillation_enabled = False

    # 3. 配置 LORA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    student_model = get_peft_model(student_model, lora_config)
    print(f"Loaded tokenizer from {tokenizer_source}")
    student_model.print_trainable_parameters()

    # 4. 加载数据集
    dataset = preprocess_dataset(tokenizer)

    # 5. 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,  # 降低学习率以适应增强蒸馏
        num_train_epochs=2,  # 增加训练轮数
        lr_scheduler_type="cosine",
        warmup_steps=200,
        logging_steps=20,
        eval_steps=200,
        save_steps=200,
        save_strategy="steps",
        eval_strategy="steps",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    # 6. 选择 Trainer
    if enhanced_distillation_enabled and teacher_model is not None:
        print("Using enhanced distillation (output + intermediate layers)")
        trainer = EnhancedDistillationTrainer(
            teacher_model=teacher_model,
            model=student_model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )
    else:
        print("Using standard distillation (output-only)")
        trainer = Trainer(
            model=student_model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

    # 7. 开始训练
    trainer.train()

    # 8. 保存最终模型
    final_model = student_model.merge_and_unload()
    Path(FINAL_MODEL_DIR).mkdir(parents=True, exist_ok=True)
    final_model.save_pretrained(FINAL_MODEL_DIR, safe_serialization=True)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print(f"Final student model saved to {FINAL_MODEL_DIR}")


if __name__ == "__main__":
    main()

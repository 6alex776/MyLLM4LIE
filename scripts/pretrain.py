# 底座模型预训练

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 屏蔽libiomp5md.dll重复初始化错误
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # 继续关闭HF警告
# HF_TOKEN 请通过环境变量或 .env 文件设置，不要硬编码在代码中

from pathlib import Path

import torch

# RTX 4090 (Ampere) 加速设置
torch.backends.cuda.matmul.allow_tf32 = True  # 启用 TF32 矩阵运算加速
torch.backends.cudnn.allow_tf32 = True        # 启用 cuDNN TF32 加速

from datasets import load_from_disk, load_dataset, concatenate_datasets
from transformers import Trainer, TrainingArguments

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import LLMIEForCausalLM, build_student_config, get_model_stats
from src.tokenizer import load_tokenizer
from src.optimizer import create_optimizer


DATASET_DIR = "./artifacts/pretrain_dataset"
OUTPUT_DIR = "./artifacts/pretrain_runs"
FINAL_MODEL_DIR = "./artifacts/base_model"

# ========== 预训练数据源配置 ==========
# 200M 小模型预训练数据集选择（大小适中、高质量）
# 主数据集：中文维基百科（事实性强、语法规范）
WIKI_DATASET = "pleisto/wikipedia-cn-20230720-filtered"
WIKI_SAMPLES = 50000  # 取前 5 万条（约 100MB）

# 辅助数据集：TigerBot 预训练数据（增加多样性）
TIGER_DATASET = "TigerResearch/pretrain_zh"
TIGER_SAMPLES = 20000  # 取前 2 万条

# 本地数据集回退（如果 HuggingFace 下载失败）
USE_LOCAL_FALLBACK = True

# ========== 优化器配置 ==========
# 设置为 True 使用 Muon（推荐用于 200M 小模型），False 使用 AdamW
USE_MUON = True
# Muon 学习率：比 AdamW 大 2-5 倍（AdamW 用 1e-4，Muon 推荐 2e-4~5e-4）
MUON_LR = 3e-4
# AdamW 学习率（用于 1D 参数，或当 USE_MUON=False 时）
ADAMW_LR = 1e-4
# 权重衰减（Muon 必须设置，推荐 0.01~0.1）
WEIGHT_DECAY = 0.01


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


def load_hf_pretrain_dataset(tokenizer):
    """
    从 HuggingFace 加载高质量中文预训练数据集。
    
    200M 小模型数据策略：
    - 主数据：维基百科（事实性强、语法规范）
    - 辅助数据：TigerBot（增加多样性）
    - 总量控制在 ~7 万条，避免数据过多导致训练时间过长
    """
    datasets_list = []
    
    # 1. 加载维基百科数据（主数据集）
    try:
        print(f"加载维基百科数据集: {WIKI_DATASET} (前 {WIKI_SAMPLES} 条)")
        wiki_data = load_dataset(WIKI_DATASET, split=f"train[:{WIKI_SAMPLES}]")
        
        # 检查字段名并统一为 "text"
        if "text" not in wiki_data.column_names:
            # 尝试常见字段名
            for col in ["content", "article", "body", "paragraph"]:
                if col in wiki_data.column_names:
                    wiki_data = wiki_data.rename_column(col, "text")
                    break
        
        # 过滤空文本和太短的文本
        wiki_data = wiki_data.filter(lambda x: len(x.get("text", "").strip()) >= 50)
        datasets_list.append(wiki_data)
        print(f"  维基百科加载完成: {len(wiki_data)} 条")
    except Exception as e:
        print(f"  维基百科加载失败: {e}")
    
    # 2. 加载 TigerBot 数据（辅助数据集）
    try:
        print(f"加载 TigerBot 数据集: {TIGER_DATASET} (前 {TIGER_SAMPLES} 条)")
        tiger_data = load_dataset(TIGER_DATASET, split=f"train[:{TIGER_SAMPLES}]")
        
        # 统一字段名
        if "text" not in tiger_data.column_names:
            for col in ["content", "article", "body", "paragraph"]:
                if col in tiger_data.column_names:
                    tiger_data = tiger_data.rename_column(col, "text")
                    break
        
        tiger_data = tiger_data.filter(lambda x: len(x.get("text", "").strip()) >= 50)
        datasets_list.append(tiger_data)
        print(f"  TigerBot 加载完成: {len(tiger_data)} 条")
    except Exception as e:
        print(f"  TigerBot 加载失败: {e}")
    
    # 3. 合并数据集
    if len(datasets_list) == 0:
        raise RuntimeError("所有在线数据集加载失败，请检查网络连接")
    
    combined = concatenate_datasets(datasets_list)
    print(f"\n数据集合并完成: 共 {len(combined)} 条")
    
    # 4. Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding=False,
        )
    
    tokenized = combined.map(
        tokenize_function,
        batched=True,
        remove_columns=combined.column_names,
    )
    
    # 5. 划分训练集和验证集
    dataset = tokenized.train_test_split(test_size=0.01, seed=42)
    print(f"训练集: {len(dataset['train'])} 条, 验证集: {len(dataset['test'])} 条")
    
    return dataset


def main():
    tokenizer, tokenizer_source = load_tokenizer()
    
    # 尝试从 HuggingFace 加载数据集，失败则回退到本地
    try:
        dataset = load_hf_pretrain_dataset(tokenizer)
    except Exception as e:
        print(f"在线数据集加载失败: {e}")
        if USE_LOCAL_FALLBACK and Path(DATASET_DIR).exists():
            print(f"回退到本地数据集: {DATASET_DIR}")
            dataset = load_from_disk(DATASET_DIR)
        else:
            raise

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

    # 如果使用 Muon，TrainingArguments 的 learning_rate 和 weight_decay 会被忽略
    # 实际值由 create_optimizer 控制
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,       # 4090 24GB，~191M 模型绰绰有余
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,        # 16×2=32 有效 batch
        learning_rate=ADAMW_LR if not USE_MUON else MUON_LR,  # Trainer 需要这个值，实际由优化器控制
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        warmup_steps=200,
        logging_steps=100,
        eval_steps=5000,
        save_steps=2000,
        save_strategy="steps",
        eval_strategy="steps",
        save_total_limit=2,
        weight_decay=WEIGHT_DECAY,
        fp16=False,
        bf16=True,
        report_to="none",
        dataloader_num_workers=min(4, os.cpu_count() // 2),
        dataloader_prefetch_factor=2,
        remove_unused_columns=False,
    )

    # 创建自定义优化器（Muon 或 AdamW）
    optimizer = create_optimizer(
        model,
        use_muon=USE_MUON,
        muon_lr=MUON_LR,
        adamw_lr=ADAMW_LR,
        weight_decay=WEIGHT_DECAY,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        optimizers=(optimizer, None),  # (optimizer, lr_scheduler)，None 表示使用 Trainer 默认的 scheduler
    )
    
    # 如果有checkpoint，从checkpoint继续训练
    trainer.train(resume_from_checkpoint=latest_checkpoint)

    Path(FINAL_MODEL_DIR).mkdir(parents=True, exist_ok=True)
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print(f"Base model saved to {FINAL_MODEL_DIR}")


if __name__ == "__main__":
    main()

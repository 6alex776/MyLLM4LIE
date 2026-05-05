# 数据集预处理

import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # 关闭Windows符号链接警告
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"  # 关闭HF未登录警告

# 开启高速下载（仅在 hf_transfer 已安装时启用，否则自动回退普通下载）
try:
    import hf_transfer  # noqa: F401
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
except ImportError:
    pass  # hf_transfer 未安装，使用 huggingface_hub 默认下载方式

from datasets import Dataset, concatenate_datasets, load_dataset

from tokenizer_utils import load_tokenizer

MAX_SEQ_LEN = 512
OUTPUT_DIR = "./artifacts/pretrain_dataset"

# 预训练数据集（中英混合，中文为主）
PRETRAIN_DATASETS = [
    # 中文维基百科 + 书籍 (占比约 60%)
    {
        "path": "yuhuanstudio/wikipedia-pretrain-zh",
        "split": "train[:80%]",
        "fields": ["title", "text"],
    },
    # 中文新闻语料 (占比约 20%)
    {
        "path": "cc100",
        "name": "zh",
        "split": "train[:5%]",
        "fields": ["text"],
    },
    # 英文核心语料 (占比约 18%)
    {
        "path": "Salesforce/wikitext",
        "name": "wikitext-2-raw-v1",
        "split": "train",
        "fields": ["text"],
    },
    # 中文对话，增加语言多样性 (占比约 2%)
    {
        "path": "BelleGroup/train_0.5M_CN",
        "split": "train[:15%]",
        "fields": ["instruction", "output"],
    },
]


def build_text(example, fields):
    parts = []
    for field in fields:
        value = example.get(field)
        if value:
            parts.append(str(value).strip())
    return "\n".join(parts).strip()


def load_source_dataset(spec) -> Dataset:
    name = spec.get("name", None)
    dataset = load_dataset(spec["path"], name, split=spec["split"])
    text_dataset = dataset.map(
        lambda row: {"text": build_text(row, spec["fields"])},
        remove_columns=dataset.column_names,
    )
    text_dataset = text_dataset.filter(lambda row: len(row["text"]) >= 20)
    return text_dataset


def tokenize_and_chunk(dataset: Dataset, tokenizer) -> Dataset:
    def tokenize_function(examples):
        return tokenizer(examples["text"], add_special_tokens=False)

    def group_texts(examples):
        concatenated = {key: sum(examples[key], []) for key in examples}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // MAX_SEQ_LEN) * MAX_SEQ_LEN
        result = {
            key: [values[index: index + MAX_SEQ_LEN] for index in range(0, total_length, MAX_SEQ_LEN)]
            for key, values in concatenated.items()
        }
        result["labels"] = [ids[:] for ids in result["input_ids"]]
        return result

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized.map(group_texts, batched=True)


def main():
    tokenizer, tokenizer_source = load_tokenizer()

    datasets = [load_source_dataset(spec) for spec in PRETRAIN_DATASETS]
    merged_dataset = concatenate_datasets(datasets).shuffle(seed=42)
    tokenized = tokenize_and_chunk(merged_dataset, tokenizer)
    tokenized = tokenized.train_test_split(test_size=0.02, seed=42)
    tokenized.save_to_disk(OUTPUT_DIR)

    print(f"Loaded tokenizer from {tokenizer_source}")
    print(f"Saved tokenized pretrain dataset to {OUTPUT_DIR}")
    print(tokenized)


if __name__ == "__main__":
    main()

# 数据集预处理

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # 关闭Windows符号链接警告
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"     # 关闭HF未登录警告
os.environ["HF_TOKEN"] = "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"  # 设置HF_TOKEN

from datasets import Dataset, concatenate_datasets, load_dataset

from tokenizer_utils import load_qwen_tokenizer

MAX_SEQ_LEN = 512
OUTPUT_DIR = "./artifacts/pretrain_dataset"

# 这两个数据集都可以直接用 Hugging Face datasets.load_dataset() 加载。
PRETRAIN_DATASETS = [
    {
        "path": "yuhuanstudio/wikipedia-pretrain-zh",
        "split": "train[:1%]",
        "fields": ["title", "text"],
    },
    {
        "path": "BelleGroup/train_0.5M_CN",
        "split": "train[:2%]",
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
    dataset = load_dataset(spec["path"], split=spec["split"])
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
            key: [values[index : index + MAX_SEQ_LEN] for index in range(0, total_length, MAX_SEQ_LEN)]
            for key, values in concatenated.items()
        }
        result["labels"] = [ids[:] for ids in result["input_ids"]]
        return result

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized.map(group_texts, batched=True)


def main():
    tokenizer, tokenizer_source = load_qwen_tokenizer()

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

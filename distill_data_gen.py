import json
from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from openai import OpenAI


OUTPUT_PATH = Path("./artifacts/distill_dataset.jsonl")
MAX_SAMPLES = 20_000

SYSTEM_PROMPTS = {
    "polish": "你是专业的中文文本润色助手，只输出润色后的最终文本，不要解释。",
    "translate": "你是专业的中英互译助手，只输出翻译结果，不要解释。",
    "expand": "你是专业的中文扩写助手，在保留原意的前提下适度扩写，只输出结果。",
    "summarize": "你是专业的中文缩写与总结助手，只输出精炼后的结果。",
    "general": "你是专业的文本处理助手，严格按照用户要求完成任务，只输出结果。",
}

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")


def load_instruction_pool():
    belle = load_dataset("BelleGroup/train_0.5M_CN", split="train[:12000]").map(
        lambda row: {
            "task_type": "general",
            "instruction": row["instruction"].strip(),
            "input_text": row["instruction"].strip(),
        }
    )

    adgen = load_dataset("HasturOfficial/adgen", split="train[:4000]").map(
        lambda row: {
            "task_type": "summarize",
            "instruction": "请将下面广告文案压缩成一句更短的表达。",
            "input_text": row.get("content", "") or row.get("title", "") or row.get("text", ""),
        }
    )

    polish_corpus = load_dataset("yuhuanstudio/wikipedia-pretrain-zh", split="train[:3000]").map(
        lambda row: {
            "task_type": "polish",
            "instruction": "请将下面这句话润色得更自然、更通顺。",
            "input_text": (row.get("title", "") + " " + row.get("text", "")).strip()[:120],
        }
    )

    expand_corpus = load_dataset("yuhuanstudio/wikipedia-pretrain-zh", split="train[3000:6000]").map(
        lambda row: {
            "task_type": "expand",
            "instruction": "请在保持原意的前提下，将下面内容扩写得更完整一些。",
            "input_text": (row.get("title", "") + " " + row.get("text", "")).strip()[:80],
        }
    )

    pool = concatenate_datasets(
        [
            belle.select_columns(["task_type", "instruction", "input_text"]),
            adgen.select_columns(["task_type", "instruction", "input_text"]),
            polish_corpus.select_columns(["task_type", "instruction", "input_text"]),
            expand_corpus.select_columns(["task_type", "instruction", "input_text"]),
        ]
    )
    pool = pool.filter(lambda row: len(row["input_text"]) >= 8)
    return pool.shuffle(seed=42).select(range(min(MAX_SAMPLES, len(pool))))


def query_teacher(task_type: str, instruction: str, input_text: str) -> str:
    response = client.chat.completions.create(
        model="qwen3.5-0.8b",
        temperature=0.3,
        max_tokens=256,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS[task_type]},
            {"role": "user", "content": f"{instruction}\n\n输入文本：\n{input_text}"},
        ],
    )
    return response.choices[0].message.content.strip()


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pool = load_instruction_pool()
    records = []

    for index, sample in enumerate(pool, start=1):
        teacher_output = query_teacher(
            task_type=sample["task_type"],
            instruction=sample["instruction"],
            input_text=sample["input_text"],
        )
        records.append(
            {
                "system": SYSTEM_PROMPTS[sample["task_type"]],
                "instruction": sample["instruction"],
                "input": sample["input_text"],
                "output": teacher_output,
            }
        )

        if index % 200 == 0:
            with OUTPUT_PATH.open("w", encoding="utf-8") as file:
                for row in records:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"Generated {index} / {len(pool)} distilled samples")

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for row in records:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved distilled dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

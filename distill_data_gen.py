import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

from datasets import concatenate_datasets, load_dataset
from openai import OpenAI

# 可选：消除Windows符号链接警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ========== 路径配置 ==========
OUTPUT_PATH = Path("./artifacts/distill_dataset.jsonl")
PROGRESS_PATH = Path("./artifacts/distill_progress.json")  # 进度保存文件
FAILED_PATH = Path("./artifacts/failed_samples.jsonl")  # 失败样本记录

# 保留任务：精简样本
TASK_SAMPLE_LIMITS = {
    "polish": 1200,  # 核心：文本润色
    "expand": 600,  # 保留：文本扩写
    "translate": 1200,  # 核心：中英翻译
    "general": 600,  # 保留：通用指令
    "summarize": 600,  # 保留：总结缩写
}

SYSTEM_PROMPTS = {
    "polish": "你是专业的中文文本润色助手，只输出润色后的最终文本，不要解释。",
    "translate": "你是专业的中英互译助手，只输出翻译结果，不要解释。",
    "expand": "你是专业的中文扩写助手，保留原意适度扩写，只输出结果。",
    "summarize": "你是专业的中文总结助手，精炼核心内容，只输出结果。",
    "general": "你是专业的文本处理助手，按要求完成任务，只输出结果。",
}

# 连接本地Qwen教师模型
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")


def load_instruction_pool():
    """加载原始指令池（无变化，仅作为基础数据）"""
    # 1. 通用任务 (保留，少量)
    belle = load_dataset("BelleGroup/train_0.5M_CN", split="train[:1000]").map(
        lambda x: {
            "task_type": "general",
            "instruction": x["instruction"].strip(),
            "input_text": x["instruction"].strip(),
        }
    ).select(range(min(TASK_SAMPLE_LIMITS["general"], 1000)))  # 提前限制最大长度

    # 2. 总结任务 (保留，少量)
    adgen = load_dataset("HasturOfficial/adgen", split="train[:1000]").map(
        lambda x: {
            "task_type": "summarize",
            "instruction": "请总结下面的内容",
            "input_text": x["content"],
        }
    ).select(range(min(TASK_SAMPLE_LIMITS["summarize"], 1000)))

    # 3. 核心：润色任务
    polish = load_dataset("yuhuanstudio/wikipedia-pretrain-zh", split="train[:2000]").map(
        lambda x: {
            "task_type": "polish",
            "instruction": "请润色这句话，让它更通顺自然",
            "input_text": (x["title"] + " " + x["text"])[:100].strip(),
        }
    ).select(range(min(TASK_SAMPLE_LIMITS["polish"], 2000)))

    # 4. 核心：扩写任务
    expand = load_dataset("yuhuanstudio/wikipedia-pretrain-zh", split="train[1000:2000]").map(
        lambda x: {
            "task_type": "expand",
            "instruction": "请扩写下面的内容，让它更丰富完整",
            "input_text": (x["title"] + " " + x["text"])[:80].strip(),
        }
    ).select(range(min(TASK_SAMPLE_LIMITS["expand"], 1000)))

    # 5. 核心：翻译任务
    translate = load_dataset("opus100", "en-zh", split="train[:2000]").map(
        lambda x, i: {
            "task_type": "translate",
            "instruction": "请进行中英互译，只输出翻译结果",
            # 交替使用中→英/英→中，保证翻译任务多样性
            "input_text": x["translation"]["zh"] if i % 2 == 0 else x["translation"]["en"],
        },
        with_indices=True
    ).select(range(min(TASK_SAMPLE_LIMITS["translate"], 2000)))

    # 合并所有5个任务
    dataset = concatenate_datasets([belle, adgen, polish, expand, translate])

    # 过滤脏数据
    dataset = dataset.filter(lambda x: len(x["input_text"]) >= 5)

    return dataset.shuffle(seed=42)


def load_progress() -> Dict[str, int]:
    """加载已完成的进度（按任务类型记录已完成数量）"""
    if not PROGRESS_PATH.exists():
        return {task: 0 for task in TASK_SAMPLE_LIMITS.keys()}

    try:
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            progress = json.load(f)
        # 确保进度字典包含所有任务类型，且不超过配置上限
        for task in TASK_SAMPLE_LIMITS.keys():
            if task not in progress:
                progress[task] = 0
            # 防止进度值超过配置上限
            progress[task] = min(progress[task], TASK_SAMPLE_LIMITS[task])
        return progress
    except Exception as e:
        print(f"加载进度文件失败，将重置进度：{e}")
        return {task: 0 for task in TASK_SAMPLE_LIMITS.keys()}


def save_progress(progress: Dict[str, int]):
    """保存当前进度（原子写入，避免损坏）"""
    temp_path = PROGRESS_PATH.with_suffix(".tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
        # 原子替换，防止写入中断导致文件损坏
        temp_path.replace(PROGRESS_PATH)
    except Exception as e:
        print(f"保存进度失败：{e}")


def query_teacher(task_type, instruction, input_text, retry=2):
    """调用教师模型，增加重试机制"""
    for attempt in range(retry + 1):
        try:
            resp = client.chat.completions.create(
                model="qwen2.5-7b",
                temperature=0.3,
                max_tokens=200,
                top_p=0.9,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPTS[task_type]},
                    {"role": "user", "content": f"{instruction}\n{input_text}"},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"调用教师模型失败（第{attempt + 1}次重试）：{e}")
            if attempt < retry:
                time.sleep(1)  # 重试前等待1秒
    return None


def write_sample_to_file(sample: dict, file_path: Path):
    """追加写入单条样本到文件（避免内存缓存所有数据）"""
    file_path.parent.mkdir(exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def process_single_sample(sample: dict) -> Optional[dict]:
    """处理单条样本，返回结果或None（失败）"""
    try:
        output = query_teacher(
            sample["task_type"],
            sample["instruction"],
            sample["input_text"]
        )
        if not output:
            return None

        result = {
            "system": SYSTEM_PROMPTS[sample["task_type"]],
            "instruction": sample["instruction"],
            "input": sample["input_text"],
            "output": output
        }
        # 实时写入结果文件
        write_sample_to_file(result, OUTPUT_PATH)
        return result
    except Exception as e:
        print(f"样本处理失败：{e}")
        # 记录失败样本
        write_sample_to_file({
            "error": str(e),
            "task_type": sample["task_type"],
            "instruction": sample["instruction"],
            "input_text": sample["input_text"]
        }, FAILED_PATH)
        return None


def main():
    import concurrent.futures

    # 1. 初始化路径和进度
    progress = load_progress()
    total_completed = sum(progress.values())
    total_samples = sum(TASK_SAMPLE_LIMITS.values())

    print("=== 任务样本配置 ===")
    for task, limit in TASK_SAMPLE_LIMITS.items():
        done = progress[task]
        print(f"{task:10s}: 总{limit}条 | 已完成{done}条 | 剩余{limit - done}条")
    print(f"\n总计：已完成{total_completed}/{total_samples}条")

    # 2. 加载原始数据并按任务分组
    pool = load_instruction_pool()
    task_groups = {
        task: pool.filter(lambda x: x["task_type"] == task)
        for task in TASK_SAMPLE_LIMITS.keys()
    }

    # 3. 配置并发（可根据硬件调整）
    MAX_WORKERS = 1
    completed = total_completed

    # 4. 按任务类型分批处理（跳过已完成的样本）
    for task_type, task_dataset in task_groups.items():
        # 关键修复：取配置上限和实际数据集长度的最小值
        task_config_limit = TASK_SAMPLE_LIMITS[task_type]
        task_actual_length = len(task_dataset)
        task_total = min(task_config_limit, task_actual_length)
        task_done = progress[task_type]

        # 防止已完成进度超过实际可处理的样本数
        if task_done >= task_total:
            print(f"\n【{task_type}】任务已完成（实际可处理{task_total}条），跳过")
            continue

        # 打印数据集实际长度，方便调试
        print(f"\n【{task_type}】任务 - 配置上限：{task_config_limit} | 实际可用：{task_actual_length} | 需处理：{task_total - task_done}")

        # 切片：只处理剩余未完成的样本（确保不越界）
        end_idx = min(task_done + (task_total - task_done), task_actual_length)
        remaining_dataset = task_dataset.select(range(task_done, end_idx))
        print(f"开始处理【{task_type}】任务，剩余样本数：{len(remaining_dataset)}")

        # 并发处理剩余样本
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_single_sample, sample)
                       for sample in remaining_dataset]

            for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                result = future.result()
                if result:
                    # 更新进度（每处理1条更新1次，保证断点准确性）
                    new_progress = task_done + idx + 1
                    # 确保进度不超过实际可处理的样本数
                    progress[task_type] = min(new_progress, task_total)
                    completed += 1

                    # 每10条保存一次进度（减少IO）
                    if idx % 10 == 0:
                        save_progress(progress)

                    # 打印进度
                    if completed % 100 == 0:
                        print(f"全局进度：{completed}/{total_samples}")

        # 任务完成后强制保存进度
        save_progress(progress)
        print(f"【{task_type}】任务处理完成，当前进度：{progress[task_type]}/{task_total}")

    # 5. 最终统计
    print("\n=== 处理完成 ===")
    final_progress = load_progress()
    total_success = sum(final_progress.values())
    # 重新计算实际总样本数（基于各任务的实际可处理数量）
    actual_total_samples = sum(
        min(TASK_SAMPLE_LIMITS[task], len(task_groups[task]))
        for task in TASK_SAMPLE_LIMITS.keys()
    )
    print(f"最终完成样本数：{total_success}/{actual_total_samples}（配置总样本数：{total_samples}）")
    print(f"结果文件路径：{OUTPUT_PATH}")
    print(f"失败样本路径：{FAILED_PATH if FAILED_PATH.exists() else '无'}")

    # 验证结果文件行数
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)
        print(f"结果文件实际行数：{line_count}")


if __name__ == "__main__":
    main()
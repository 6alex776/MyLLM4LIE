# 蒸馏数据生成

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

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
    "polish": 1800,  # 核心：文本润色
    "expand": 1500,  # 核心：文本扩写
    "translate": 1800,  # 核心：中英翻译
    "general": 600,  # 保留：通用指令
    "summarize": 1000,  # 保留：总结缩写
}

SYSTEM_PROMPTS = {
    "polish": "你是专业的中文文本润色助手，只输出润色后的最终文本，不要解释。",
    "translate": "你是专业的中英互译助手，只输出翻译结果，不要解释。",
    "expand": "你是专业的中文扩写助手，保留原意适度扩写，只输出结果。",
    "summarize": "你是专业的中文总结助手，精炼核心内容，只输出结果。",
    "general": "你是专业的文本处理助手，按要求完成指定的任务，只输出结果。",
}

# 连接本地Qwen教师模型
client = None  # 延迟初始化，避免多进程问题


def init_worker():
    """初始化工作进程（每个进程独立创建客户端）"""
    global client
    client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")


def load_instruction_pool():
    """加载原始指令池（无变化，仅作为基础数据）"""
    # 1. 通用任务 (保留，少量)
    belle = load_dataset("BelleGroup/train_0.5M_CN", split="train[:1200]").map(
        lambda x: {
            "task_type": "general",
            "instruction": x["instruction"].strip(),
            "input_text": x["instruction"].strip(),
        }
    ).select(range(min(TASK_SAMPLE_LIMITS["general"], 1000)))

    # 2. 总结任务 (保留，少量)
    adgen = load_dataset("HasturOfficial/adgen", split="train[:3000]").map(
        lambda x: {
            "task_type": "summarize",
            "instruction": "请总结下面的内容",
            "input_text": x["content"],
        }
    ).select(range(min(TASK_SAMPLE_LIMITS["summarize"], 1000)))

    # 3. 核心：润色任务（混合高质量数据集）
    # 3.1 Belle 通用指令（适合润色）
    polish_belle = load_dataset("BelleGroup/train_0.5M_CN", split="train[:1500]").map(
        lambda x: {
            "task_type": "polish",
            "instruction": "请润色这句话，让它更通顺自然",
            "input_text": x["instruction"][:100].strip(),
        }
    ).select(range(min(TASK_SAMPLE_LIMITS["polish"] // 2, 1000)))
    
    # 3.2 维基百科数据（补充润色）
    polish_wiki = load_dataset("yuhuanstudio/wikipedia-pretrain-zh", split="train[:2000]").map(
        lambda x: {
            "task_type": "polish",
            "instruction": "请润色这句话，让它更通顺自然",
            "input_text": (x["title"] + " " + x["text"])[:100].strip(),
        }
    ).select(range(min(TASK_SAMPLE_LIMITS["polish"] // 2, 1000)))
    
    # 合并润色数据集
    polish = concatenate_datasets([polish_belle, polish_wiki])

    # 4. 核心：扩写任务（混合数据集）
    # 4.1 Adgen 广告生成（适合扩写训练）
    expand_adgen = load_dataset("HasturOfficial/adgen", split="train[:1500]").map(
        lambda x: {
            "task_type": "expand",
            "instruction": "请扩写下面的内容，让它更丰富完整",
            "input_text": x["content"][:80].strip(),
        }
    ).select(range(min(TASK_SAMPLE_LIMITS["expand"] // 2, 800)))
    
    # 4.2 维基百科数据（补充扩写）
    expand_wiki = load_dataset("yuhuanstudio/wikipedia-pretrain-zh", split="train[1500:3000]").map(
        lambda x: {
            "task_type": "expand",
            "instruction": "请扩写下面的内容，让它更丰富完整",
            "input_text": (x["title"] + " " + x["text"])[:80].strip(),
        }
    ).select(range(min(TASK_SAMPLE_LIMITS["expand"] // 2, 800)))
    
    # 合并扩写数据集
    expand = concatenate_datasets([expand_adgen, expand_wiki])

    # 5. 核心：翻译任务
    translate = load_dataset("opus100", "en-zh", split="train[:4000]").map(
        lambda x, i: {
            "task_type": "translate",
            "instruction": "请进行中英互译，只输出翻译结果",
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
        for task in TASK_SAMPLE_LIMITS.keys():
            if task not in progress:
                progress[task] = 0
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
        temp_path.replace(PROGRESS_PATH)
    except Exception as e:
        print(f"保存进度失败：{e}")


def query_teacher(task_type, instruction, input_text, retry=2):
    """调用教师模型，增加重试机制"""
    global client
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
                time.sleep(1)
    return None


def write_samples_batch(samples: list, file_path: Path):
    """分批写入样本（批量写入，提高效率）"""
    file_path.parent.mkdir(exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def process_sample_wrapper(args):
    """多进程包装函数，处理单条样本"""
    sample = args
    try:
        output = query_teacher(
            sample["task_type"],
            sample["instruction"],
            sample["input_text"]
        )
        if not output:
            return None, {
                "error": "教师模型返回空",
                "task_type": sample["task_type"],
                "instruction": sample["instruction"],
                "input_text": sample["input_text"]
            }

        result = {
            "system": SYSTEM_PROMPTS[sample["task_type"]],
            "instruction": sample["instruction"],
            "input": sample["input_text"],
            "output": output
        }
        return result, None
    except Exception as e:
        error_info = {
            "error": str(e),
            "task_type": sample["task_type"],
            "instruction": sample["instruction"],
            "input_text": sample["input_text"]
        }
        return None, error_info


def main():
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

    # 3. 配置多进程（根据CPU核心数自动设置）
    MAX_WORKERS = min(4, cpu_count() // 2)
    print(f"\n使用 {MAX_WORKERS} 个进程并行处理")

    # 4. 按任务类型分批处理
    completed = total_completed

    for task_type, task_dataset in task_groups.items():
        task_config_limit = TASK_SAMPLE_LIMITS[task_type]
        task_actual_length = len(task_dataset)
        task_total = min(task_config_limit, task_actual_length)
        task_done = progress[task_type]

        if task_done >= task_total:
            print(f"\n【{task_type}】任务已完成，跳过")
            continue

        print(f"\n【{task_type}】任务 - 需处理：{task_total - task_done}条")

        # 切片：只处理剩余未完成的样本
        end_idx = min(task_done + (task_total - task_done), task_actual_length)
        remaining_samples = [dict(sample) for sample in task_dataset.select(range(task_done, end_idx))]

        # 多进程处理
        with Pool(processes=MAX_WORKERS, initializer=init_worker) as pool:
            results = list(tqdm(
                pool.imap(process_sample_wrapper, remaining_samples),
                total=len(remaining_samples),
                desc=f"Processing {task_type}"
            ))

        # 分批写入结果
        batch_size = 100
        success_results = [r[0] for r in results if r[0] is not None]
        failed_results = [r[1] for r in results if r[1] is not None]

        # 写入成功样本（分批）
        for i in range(0, len(success_results), batch_size):
            batch = success_results[i:i+batch_size]
            write_samples_batch(batch, OUTPUT_PATH)

        # 写入失败样本（分批）
        if failed_results:
            for i in range(0, len(failed_results), batch_size):
                batch = failed_results[i:i+batch_size]
                write_samples_batch(batch, FAILED_PATH)

        # 更新进度
        progress[task_type] = min(task_done + len(success_results), task_total)
        completed += len(success_results)
        save_progress(progress)

        print(f"【{task_type}】任务完成，成功{len(success_results)}条，失败{len(failed_results)}条")

    # 5. 最终统计
    print("\n=== 处理完成 ===")
    final_progress = load_progress()
    total_success = sum(final_progress.values())
    print(f"最终完成样本数：{total_success}/{total_samples}")
    print(f"结果文件路径：{OUTPUT_PATH}")

    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)
        print(f"结果文件实际行数：{line_count}")


if __name__ == "__main__":
    main()
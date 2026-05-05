# 蒸馏数据生成

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Optional
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from datasets import concatenate_datasets, load_dataset, Dataset
from openai import OpenAI

# 消除Windows符号链接警告（必须在import datasets之前设置）
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Hugging Face 国内镜像（解决连接超时问题）
# 如果网络正常可注释掉；如果超时则取消下面这行的注释
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ========== 路径配置 ==========
OUTPUT_PATH = Path("./artifacts/distill_dataset.jsonl")
PROGRESS_PATH = Path("./artifacts/distill_progress.json")  # 进度保存文件
FAILED_PATH = Path("./artifacts/failed_samples.jsonl")  # 失败样本记录

# ========== 200M 小模型优化配置 ==========
# 核心原则：质量 > 数量，总样本控制在 6K-8K，output 长度控制在 150 字以内
TASK_SAMPLE_LIMITS = {
    "polish_and_correct": 1800,  # 核心任务：润色纠错，短文本输出，适合小模型
    "summarize": 1200,  # 核心任务：输入长输出短，小模型容易学好
    "translate": 1600,  # 核心任务：输入输出长度相近，格式固定
    "general": 1000,  # 辅助：通用对话能力，控制数量避免偏科
    "expand": 800,  # 降级：扩写输出长，小模型容易生成质量差，减少数量
}

# 小模型 system prompt 必须更精简，减少认知负担
SYSTEM_PROMPTS = {
    "polish_and_correct": "修正文本中的错别字和语法错误，保持原意。只输出修正后的文本。",
    "translate": "进行中英互译，只输出翻译结果。",
    "expand": "扩写下面的内容，保持原意，输出控制在150字以内。",
    "summarize": "总结内容，精炼核心，输出控制在100字以内。",
    "general": "按要求完成任务，只输出结果。",
}

# 小模型输出长度限制（字符数）
MAX_OUTPUT_LENGTH = {
    "polish_and_correct": 200,
    "translate": 150,
    "expand": 150,
    "summarize": 100,
    "general": 150,
}

# 连接本地Qwen教师模型
client = None  # 延迟初始化，避免多进程问题


def init_worker():
    """初始化工作进程（每个进程独立创建客户端）"""
    global client
    client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")


def load_instruction_pool(incomplete_tasks: set = None):
    """加载原始指令池：使用更高质量的数据集替代 yuhuanstudio/wikipedia-pretrain-zh"""
    if incomplete_tasks is None:
        incomplete_tasks = set(TASK_SAMPLE_LIMITS.keys())

    datasets_list = []

    # 1. 通用指令 — BELLE 3.5M 指令集（对话形式，更口语化自然）
    if "general" in incomplete_tasks:
        belle_general = load_dataset(
            "BelleGroup/train_3.5M_CN", split="train[:5000]"
        ).map(
            lambda x: {
                "task_type": "general",
                "instruction": x["conversations"][0]["value"].strip(),
                "input_text": x["conversations"][0]["value"].strip(),
                # 提取assistant的第一轮回复作为输出
                "output": x["conversations"][1]["value"].strip() if len(x["conversations"]) > 1 else "",
            }
        ).filter(lambda x: len(x["output"]) > 0).select(range(min(TASK_SAMPLE_LIMITS["general"], 2000)))
        datasets_list.append(belle_general)

    # 2. 总结任务 — LCSTS 数据集（字段名：text=原文, summary=摘要）
    if "summarize" in incomplete_tasks:
        lcsts_data = load_dataset(
            "hugcyp/LCSTS", split="train[:2000]"
        ).map(
            lambda x: {
                "task_type": "summarize",
                "instruction": "请总结下面的内容，只输出总结结果",
                "input_text": x["text"].strip(),
                "output": x["summary"].strip(),
            }
        ).filter(lambda x: 50 <= len(x["input_text"]) <= 300)
        summarize = lcsts_data.select(range(min(TASK_SAMPLE_LIMITS["summarize"], 1200)))
        datasets_list.append(summarize)

    # 3. 润色并纠错任务 — twnlp/cgc_data（中文语法纠错数据集，含病句和正确句子）
    if "polish_and_correct" in incomplete_tasks:
        # CGC数据集：包含中文语法错误和修正后的句子，每行格式为"错误句子 正确句子"
        cgc_data = load_dataset(
            "twnlp/cgc_data", split="train"
        ).map(
            lambda x: {
                "task_type": "polish_and_correct",
                "instruction": "请修正下面这段文字中的错别字和语法错误，使其通顺规范",
                "input_text": x["text"].strip(),
            }
        ).filter(lambda x: len(x["input_text"]) >= 20)

        # 取前1800条作为润色纠错任务数据
        polish_correct = cgc_data.select(range(min(TASK_SAMPLE_LIMITS["polish_and_correct"], 1800)))
        datasets_list.append(polish_correct)

    # 4. 扩写任务 — 200M 小模型优化：使用更短的输入，降低扩写难度
    # 策略：从 chinese-cosmopedia 中提取短句（20-50字），要求扩写到 80-120 字
    if "expand" in incomplete_tasks:
        # 只取 wikihow 中适合扩写的短段落（实用性强，结构清晰）
        wikihow_expand = load_dataset(
            "opencsg/chinese-cosmopedia", split="train",
            streaming=True
        ).filter(
            lambda x: x["data_format"] == "wikihow"
        ).take(3000)

        wikihow_expand = list(wikihow_expand)
        wikihow_expand = (
            Dataset.from_list(wikihow_expand)
            .map(lambda x: {
                "task_type": "expand",
                "instruction": "将下面的短句扩写成一段通顺的话，80字左右",
                # 200M 模型：输入控制在 20-50 字，降低难度
                "input_text": x["text"][:50].strip(),
            })
            .filter(lambda x: 20 <= len(x["input_text"]) <= 50)
        )

        expand = wikihow_expand.select(range(min(TASK_SAMPLE_LIMITS["expand"], 800)))
        datasets_list.append(expand)

    # 5. 翻译任务 — IWSLT 英中口语翻译数据集（本地文件，质量更高）
    if "translate" in incomplete_tasks:
        import pandas as pd
        import json

        iwslt_data = []
        local_iwslt_path = Path("datasets/iwslt_en_zh")

        # 优先读取本地 CSV 文件
        csv_file = local_iwslt_path / "damo_mt_iwslt1617_testset_en2zh.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            for idx, row in df.iterrows():
                # CSV 列名：Source（英文原文）, Reference（中文翻译）
                en_text = str(row.get("Source", "")).strip().strip('"')
                zh_text = str(row.get("Reference", "")).strip().strip('"')
                if en_text and zh_text:
                    # 交替进行英翻中和中翻英，实现互译
                    if idx % 2 == 0:
                        # 英翻中
                        iwslt_data.append({
                            "task_type": "translate",
                            "instruction": "请进行中英互译，只输出翻译结果",
                            "input_text": en_text,
                            "output": zh_text,
                        })
                    else:
                        # 中翻英
                        iwslt_data.append({
                            "task_type": "translate",
                            "instruction": "请进行中英互译，只输出翻译结果",
                            "input_text": zh_text,
                            "output": en_text,
                        })
        else:
            # 回退到 opus100 在线数据集
            print("本地 IWSLT 数据集未找到，回退到 opus100...")
            opus_data = load_dataset("opus100", "en-zh", split="train[:6000]")
            for i, x in enumerate(opus_data):
                iwslt_data.append({
                    "task_type": "translate",
                    "instruction": "请进行中英互译，只输出翻译结果",
                    "input_text": x["translation"]["zh"] if i % 2 == 0 else x["translation"]["en"],
                })

        translate = Dataset.from_list(iwslt_data[:TASK_SAMPLE_LIMITS["translate"]])
        datasets_list.append(translate)

    # 合并所有已加载的任务数据集
    dataset = concatenate_datasets(datasets_list)
    original_count = len(dataset)

    # ========== 200M 小模型数据质量控制 ==========
    # 1. 基础过滤：输入不能太短
    dataset = dataset.filter(lambda x: len(x["input_text"]) >= 15)
    after_min_len = len(dataset)

    # 2. 长度过滤：输入不能超过 300 字（小模型上下文有限，太长会分散注意力）
    dataset = dataset.filter(lambda x: len(x["input_text"]) <= 300)
    after_max_len = len(dataset)

    # 3. 如果已有 output，过滤过长的 output
    def filter_output_length(x):
        if "output" in x and x["output"]:
            task = x.get("task_type", "general")
            max_len = MAX_OUTPUT_LENGTH.get(task, 150)
            return len(x["output"]) <= max_len * 1.5  # 允许 1.5 倍缓冲
        return True

    dataset = dataset.filter(filter_output_length)
    after_output_len = len(dataset)

    # 4. 去重：基于 input_text 去重，避免重复样本浪费训练资源
    seen_inputs = set()
    def dedup_by_input(x):
        key = x["input_text"].strip()[:50]  # 取前50字作为去重键
        if key in seen_inputs:
            return False
        seen_inputs.add(key)
        return True

    dataset = dataset.filter(dedup_by_input)
    after_dedup = len(dataset)

    # 5. 过滤低质量样本（包含过多特殊字符或乱码）
    def filter_garbage(x):
        text = x["input_text"]
        # 如果特殊字符占比超过 30%，认为是低质量数据
        special_chars = len(re.findall(r'[^\u4e00-\u9fa5a-zA-Z0-9\s\.\,\;\:\!\?\"\'\(\)\-\—\…]', text))
        if len(text) > 0 and special_chars / len(text) > 0.3:
            return False
        return True

    dataset = dataset.filter(filter_garbage)
    after_garbage = len(dataset)

    # 打印数据质量报告
    print("\n=== 数据质量过滤报告 ===")
    print(f"原始样本数：{original_count}")
    print(f"过滤太短输入后：{after_min_len} (-{original_count - after_min_len})")
    print(f"过滤太长输入后：{after_max_len} (-{after_min_len - after_max_len})")
    print(f"过滤太长输出后：{after_output_len} (-{after_max_len - after_output_len})")
    print(f"去重后：{after_dedup} (-{after_output_len - after_dedup})")
    print(f"过滤低质量后：{after_garbage} (-{after_dedup - after_garbage})")
    print(f"最终可用样本：{after_garbage} (保留率 {after_garbage/original_count*100:.1f}%)")

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
            # 200M 小模型：根据任务类型限制输出长度
            max_tokens = MAX_OUTPUT_LENGTH.get(task_type, 150)
            resp = client.chat.completions.create(
                model="qwen2.5-7b",
                temperature=0.3,
                max_tokens=max_tokens,
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
        task_type = sample["task_type"]

        # 以下任务数据集中已包含输出，直接解析使用，跳过教师模型
        if task_type == "polish_and_correct":
            text = sample["input_text"]
            # CGC数据集格式："错误句子\t正确句子"（用制表符分隔）
            parts = text.split("\t")
            if len(parts) >= 2:
                input_text = parts[0].strip()
                output = parts[-1].strip()
            else:
                return None, {
                    "error": "CGC数据格式错误，无法解析",
                    "task_type": task_type,
                    "input_text": text
                }

        elif task_type == "summarize":
            # LCSTS数据集：已包含input和output字段
            input_text = sample["input_text"]
            output = sample.get("output", "")
            if not output:
                return None, {
                    "error": "LCSTS数据缺少output字段",
                    "task_type": task_type,
                    "input_text": input_text
                }

        elif task_type == "general":
            # BELLE数据集：已包含output字段（assistant的回复）
            input_text = sample["input_text"]
            output = sample.get("output", "")
            if not output:
                return None, {
                    "error": "BELLE数据缺少output字段",
                    "task_type": task_type,
                    "input_text": input_text
                }

        elif task_type == "translate":
            # IWSLT数据集：已包含output字段（中文翻译）
            input_text = sample["input_text"]
            output = sample.get("output", "")
            if not output:
                return None, {
                    "error": "IWSLT数据缺少output字段",
                    "task_type": task_type,
                    "input_text": input_text
                }

        else:
            # 其他任务（expand）：调用教师模型生成输出
            output = query_teacher(
                task_type,
                sample["instruction"],
                sample["input_text"]
            )
            if not output:
                return None, {
                    "error": "教师模型返回空",
                    "task_type": task_type,
                    "instruction": sample["instruction"],
                    "input_text": sample["input_text"]
                }
            input_text = sample["input_text"]

        result = {
            "system": SYSTEM_PROMPTS[task_type],
            "instruction": sample["instruction"],
            "input": input_text,
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

    # 2. 只加载尚未完成的任务所需的数据集（跳过已完成任务，省时省流量）
    incomplete_tasks = {
        task for task, limit in TASK_SAMPLE_LIMITS.items()
        if progress[task] < limit
    }
    if not incomplete_tasks:
        print("所有任务已完成，无需继续！")
        return
    print(f"待处理任务：{incomplete_tasks}")
    pool = load_instruction_pool(incomplete_tasks)
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
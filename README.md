# MyLLM4LIE - 轻量级中文文本增强模型

基于知识蒸馏的 ~200M 参数中文语言模型，用于替换 LLMInputEnhancer（https://github.com/6alex776/LLMInputEnhancer） 中的 Qwen3.5-0.8B。

## 项目特点

- **小参数**：~200M 参数，单卡 8G 显存可训练
- **高效率**：集成 Muon 优化器，训练速度比 AdamW 快 ~2 倍
- **高质量**：5 大任务（润色纠错、摘要、翻译、扩写、通用对话），数据均来自高质量开源数据集
- **易部署**：支持导出为 HF 格式和 GGUF 量化

## 模型架构

| 组件 | 配置 |
|------|------|
| 结构 | Decoder-only Transformer (LLaMA 风格) |
| 参数量 | ~200M |
| Hidden Size | 768 |
| Intermediate Size | 4096 |
| Layers | 12 |
| Attention Heads | 12 |
| Max Position | 1024 |
| 激活函数 | SiLU (SwiGLU) |
| 位置编码 | RoPE |
| 归一化 | RMSNorm + Pre-Norm |
| 注意力残差 | 跨层残差连接（可配置） |

## 文件结构

```
MyLLM4LIE/
├── my_model.py              # 模型架构定义（含注意力残差）
├── muon_optimizer.py        # Muon + AdamW 混合优化器
├── tokenizer_utils.py       # 分词器加载工具
│
├── pretrain.py              # 预训练脚本（支持 HuggingFace 数据集）
├── pretrain_data.py         # 预训练数据预处理
│
├── distill_data_gen.py      # 蒸馏数据生成（5 大任务）
├── distill_train.py         # LoRA 指令微调（支持 Muon）
│
├── export_to_hf_llama.py    # 导出为 HuggingFace LLaMA 格式
└── test.py                  # 模型测试脚本
```

## 训练流程

### 1. 预训练

从 HuggingFace 加载高质量中文语料：

```bash
python pretrain.py
```

**数据源**：
- `pleisto/wikipedia-cn-20230720-filtered`（50K 条，主数据）
- `TigerResearch/pretrain_zh`（20K 条，辅助数据）

**优化器**：默认使用 Muon（矩阵参数）+ AdamW（1D 参数）混合优化

### 2. 蒸馏数据生成

使用本地 Qwen 教师模型生成 SFT 数据：

```bash
# 先启动 llama-server（Qwen2.5-7B）
# 然后运行
python distill_data_gen.py
```

**5 大任务**：

| 任务 | 数据集 | 样本数 | 说明 |
|------|--------|--------|------|
| polish_and_correct | twnlp/cgc_data | 1500 | 中文语法纠错 |
| summarize | hugcyp/LCSTS | 1200 | 微博短文本摘要 |
| translate | 本地 IWSLT | 1200 | 中英互译 |
| general | BelleGroup/train_3.5M_CN | 800 | 通用对话 |
| expand | opencsg/chinese-cosmopedia | 600 | 短句扩写 |

### 3. 指令微调

```bash
python distill_train.py
```

- LoRA 秩：12
- 目标模块：q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### 4. 导出与量化

```bash
# 导出为 HF LLaMA 格式
python export_to_hf_llama.py

# 转 GGUF（需 llama.cpp）
python convert_hf_to_gguf.py .\artifacts\final_student_hf_llama --outfile .\artifacts\llmie-student-f16.gguf --outtype f16

# 量化
.\llama-quantize.exe .\artifacts\llmie-student-f16.gguf .\artifacts\llmie-student-q4_k_m.gguf q4_k_m
```

## 关键配置

### Muon 优化器开关

```python
# pretrain.py / distill_train.py
USE_MUON = True          # True: Muon+AdamW, False: 纯 AdamW
MUON_LR = 3e-4          # Muon 学习率（比 AdamW 大 2-5 倍）
ADAMW_LR = 1e-4         # AdamW 学习率
WEIGHT_DECAY = 0.01     # 权重衰减
```

### 注意力残差开关

```python
# my_model.py 中通过 config 控制
use_attention_residual=True      # 启用跨层注意力残差
attention_residual_alpha=0.3     # 残差融合权重
```

### 数据质量控制

```python
# distill_data_gen.py
TASK_SAMPLE_LIMITS = {
    "polish_and_correct": 1500,
    "summarize": 1200,
    "translate": 1200,
    "general": 800,
    "expand": 600,
}
```

## 环境要求

- Python 3.10+
- PyTorch 2.0+
- transformers, datasets, peft
- 可选：flash-attn（训练加速）

## 硬件要求

- **训练**：RTX 4090 24GB（或同等算力）
- **推理**：8GB 显存即可

## 项目演进

- **v1.0**：基础 116M 模型，标准 Transformer
- **v2.0**：升级至 ~200M，添加 Muon 优化器、注意力残差、高质量数据集

## License

MIT

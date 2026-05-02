# LLMInputEnhancer 学生模型训练方案

## 1. 目标

本目录用于训练一个可以替换现有 `Qwen3.5-0.8B` 的轻量化学生模型，满足以下约束：

- Windows + RTX 8G 显存可训练
- Decoder-only、LLaMA 风格
- 使用 RoPE、RMSNorm、Pre-Norm、SwiGLU
- 使用 `Qwen/Qwen-tokenizer`
- 参数量控制在约 116M
- 最终可导出为 HF LLaMA 格式并转 GGUF
- 主项目业务代码无需修改，只替换模型文件

## 2. 模型结构

当前学生模型定义在 [my_model.py](C:\Users\29390\OneDrive\Desktop\project\MyLLM4LIE\my_model.py)。

推荐结构如下：

- `hidden_size = 512`
- `intermediate_size = 1536`
- `num_hidden_layers = 12`
- `num_attention_heads = 8`
- `max_position_embeddings = 1024`
- `tie_word_embeddings = True`

说明：

- 由于 Qwen 分词器词表很大，若不共享输入输出词嵌入，参数量会直接超标。
- 这一版约 `116M` 参数，能卡在你的 100M～120M 硬约束内。

## 3. 数据集选择

### 预训练

优先选择短文本、中文友好、Hugging Face 可直接加载的数据：

1. `yuhuanstudio/wikipedia-pretrain-zh`
   链接：[https://hf.co/datasets/yuhuanstudio/wikipedia-pretrain-zh](https://hf.co/datasets/yuhuanstudio/wikipedia-pretrain-zh)
2. `BelleGroup/train_0.5M_CN`
   链接：[https://hf.co/datasets/BelleGroup/train_0.5M_CN](https://hf.co/datasets/BelleGroup/train_0.5M_CN)

说明：

- 第一份提供稳定中文通用语料。
- 第二份虽然是指令数据，但文本短、中文质量尚可，对小模型很友好。

### 蒸馏 / 指令微调

1. `BelleGroup/train_0.5M_CN`
   链接：[https://hf.co/datasets/BelleGroup/train_0.5M_CN](https://hf.co/datasets/BelleGroup/train_0.5M_CN)
2. `HasturOfficial/adgen`
   链接：[https://hf.co/datasets/HasturOfficial/adgen](https://hf.co/datasets/HasturOfficial/adgen)
3. `yuhuanstudio/wikipedia-pretrain-zh`
   链接：[https://hf.co/datasets/yuhuanstudio/wikipedia-pretrain-zh](https://hf.co/datasets/yuhuanstudio/wikipedia-pretrain-zh)

说明：

- `Belle` 负责一般指令响应。
- `adgen` 可补一点短文本压缩/改写类能力，比较贴近你的桌面文本增强场景。
- `wikipedia-pretrain-zh` 在蒸馏阶段额外切出一部分短句，专门构造 `polish` 和 `expand` 任务，避免学生模型只学到泛化问答却学不到你的核心文本增强功能。

## 4. 训练顺序

### 本地分词器目录

训练脚本现在会优先读取这个本地目录：

`.\artifacts\tokenizer\Qwen-tokenizer`

如果该目录不存在，脚本才会回退到线上 `Qwen/Qwen-tokenizer`。

因此你可以先手动把 Qwen 分词器相关文件放到这里，再离线执行后续训练。

1. 生成预训练数据

```powershell
python pretrain_data.py
```

2. 进行轻量预训练

```powershell
python pretrain.py
```

3. 启动本地 `llama-server`，用老师模型生成蒸馏数据

```powershell
python distill_data_gen.py
```

4. 对学生模型做 LoRA 指令蒸馏微调

```powershell
python distill_train.py
```

5. 导出为标准 HF LLaMA 权重

```powershell
python export_to_hf_llama.py
```

6. 在 `llama.cpp` 中转 GGUF

```powershell
python convert_hf_to_gguf.py .\artifacts\final_student_hf_llama --outfile .\artifacts\llmie-student-f16.gguf --outtype f16
```

7. 如需进一步压缩显存，可继续量化

```powershell
.\llama-quantize.exe .\artifacts\llmie-student-f16.gguf .\artifacts\llmie-student-q4_k_m.gguf q4_k_m
```

## 5. 为什么蒸馏阶段不用 KL 蒸馏

这是刻意的工程取舍：

- 你的硬件只有 8G 显存
- 你要求“最简易落地”的本科方案
- 你明确不希望上复杂蒸馏

因此这里采用最稳定的**响应式蒸馏**：

- 先用老师模型离线生成高质量回答
- 再把这些回答当成监督信号，对学生模型做 LoRA SFT

这本质上仍然是知识蒸馏，但训练链路更容易跑通，也更适合写进毕设。

## 6. 无缝替换主项目的方法

你的主项目当前已经走 OpenAI 兼容接口 + `llama.cpp` 调用链，因此最终替换方式非常简单：

1. 训练完成后得到 `llmie-student-q4_k_m.gguf`
2. 用它替换主项目目前加载的 `Qwen3.5-0.8B` GGUF 文件
3. 保持原有 `llama-server` 启动方式、端口、OpenAI API 调用格式不变

也就是说：

- 不改 UI
- 不改热键
- 不改业务逻辑
- 不改调用接口

只换底层模型文件即可。

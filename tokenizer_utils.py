from pathlib import Path

from transformers import AutoTokenizer


CHATGLM3_LOCAL_DIR = Path("./artifacts/tokenizer/chatglm3-6b")
CHATGLM3_REMOTE_NAME = "THUDM/chatglm3-6b"
QWEN_LOCAL_DIR = Path("./artifacts/tokenizer/Qwen-tokenizer")
QWEN_REMOTE_NAME = "Qwen/Qwen-tokenizer"

# 当前使用的分词器：Qwen（词表 151K，与现有底座模型匹配）
# 如需切换为 ChatGLM3：改为 LOCAL_TOKENIZER_DIR = CHATGLM3_LOCAL_DIR，并重新训练底座模型#TODO
LOCAL_TOKENIZER_DIR = QWEN_LOCAL_DIR
REMOTE_TOKENIZER_NAME = QWEN_REMOTE_NAME


def resolve_tokenizer_source() -> str:
    if LOCAL_TOKENIZER_DIR.exists():
        return str(LOCAL_TOKENIZER_DIR)
    return REMOTE_TOKENIZER_NAME


def load_tokenizer():
    tokenizer_source = resolve_tokenizer_source()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, tokenizer_source


# 兼容旧函数名
def load_qwen_tokenizer():
    return load_tokenizer()

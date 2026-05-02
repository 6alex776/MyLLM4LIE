from pathlib import Path

from transformers import AutoTokenizer


LOCAL_TOKENIZER_DIR = Path("./artifacts/tokenizer/Qwen-tokenizer")
REMOTE_TOKENIZER_NAME = "Qwen/Qwen-tokenizer"


def resolve_tokenizer_source() -> str:
    if LOCAL_TOKENIZER_DIR.exists():
        return str(LOCAL_TOKENIZER_DIR)
    return REMOTE_TOKENIZER_NAME


def load_qwen_tokenizer():
    tokenizer_source = resolve_tokenizer_source()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, tokenizer_source

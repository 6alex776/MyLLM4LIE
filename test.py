import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from transformers import AutoTokenizer

from my_model import LLMIEForCausalLM  # 自定义模型必须直接导入

MODEL_PATH = "./artifacts/sft_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print(f"设备: {DEVICE}")
    print(f"模型路径: {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = LLMIEForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    prompts = [
        "今天天气真好，我想出去",
        "人工智能的发展将会改变",
        "The future of artificial intelligence",
        "从前有座山，山里有座庙，庙里有个",
    ]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.15,       # 抑制重复
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n{'='*60}")
        print(f"输入: {prompt}")
        print(f"润色: {answer}")


if __name__ == "__main__":
    main()
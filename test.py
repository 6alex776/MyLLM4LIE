import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./artifacts/final_student_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_chat_prompt(system_prompt: str, instruction: str, user_input: str) -> str:
    """构建与训练数据一致的聊天格式"""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}\n\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    # 使用与训练数据一致的聊天格式
    system_prompt = "你是专业的中英互译助手，只输出翻译结果，不要解释。"
    instruction = "请进行中英互译，只输出翻译结果"
    user_input = "我很开心今天能够与朋友们相聚，一起分享生活的点点滴滴，感受彼此的陪伴和温暖。今天的阳光明媚，微风轻拂，让人心情愉悦。我们聊起了各自的工作、生活中的趣事，笑声不断，气氛十分融洽。"

    prompt = build_chat_prompt(system_prompt, instruction, user_input)
    input_len = len(tokenizer.encode(prompt, add_special_tokens=False))

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.3,
            top_p=0.9,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 只提取新生成的部分（去掉输入的 prompt）
    generated_ids = outputs[0][input_len:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    print("=" * 50)
    print(f"输入: {user_input}")
    print(f"输出: {answer if answer else '无有效输出'}")


if __name__ == "__main__":
    main()
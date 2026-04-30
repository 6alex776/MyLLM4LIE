import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from transformers import AutoTokenizer

from my_model import LLMIEForCausalLM


MODEL_PATH = "./artifacts/final_student_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = LLMIEForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    prompt = """<|im_start|>system
你是专业的文本处理助手，严格按照用户要求完成任务，只输出结果。<|im_end|>
<|im_start|>user
请润色这句话，让表达更自然：我很开心今天<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            temperature=0.2,
            top_p=0.8,
            repetition_penalty=1.2,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    print("=" * 50)
    print(result)


if __name__ == "__main__":
    main()

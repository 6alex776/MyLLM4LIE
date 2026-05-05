from pathlib import Path

from transformers import LlamaConfig, LlamaForCausalLM

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import LLMIEForCausalLM
from src.tokenizer import load_qwen_tokenizer


SOURCE_MODEL_DIR = "./artifacts/final_student_model"
EXPORT_MODEL_DIR = "./artifacts/final_student_hf_llama"


def main():
    source_model = LLMIEForCausalLM.from_pretrained(SOURCE_MODEL_DIR)
    tokenizer, tokenizer_source = load_qwen_tokenizer()

    config = source_model.config
    llama_config = LlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        tie_word_embeddings=True,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    llama_model = LlamaForCausalLM(llama_config)
    llama_model.load_state_dict(source_model.state_dict(), strict=True)

    Path(EXPORT_MODEL_DIR).mkdir(parents=True, exist_ok=True)
    llama_model.save_pretrained(EXPORT_MODEL_DIR, safe_serialization=True)
    tokenizer.save_pretrained(EXPORT_MODEL_DIR)

    print(f"Loaded tokenizer from {tokenizer_source}")
    print(f"Exported HF LLaMA-format model to {EXPORT_MODEL_DIR}")
    print("Next step: use llama.cpp/convert_hf_to_gguf.py to generate GGUF.")


if __name__ == "__main__":
    main()

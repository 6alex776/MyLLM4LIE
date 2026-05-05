from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

try:
    from flash_attn import flash_attn_qkvpacked_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False


class LLMIEConfig(PretrainedConfig):
    model_type = "llmie_student"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 151_936,
        hidden_size: int = 512,
        intermediate_size: int = 1536,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 8,
        max_position_embeddings: int = 1024,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        initializer_range: float = 0.02,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        tie_word_embeddings: bool = True,
        use_cache: bool = False,
        # ========== 注意力残差配置 ==========
        use_attention_residual: bool = True,  # 是否启用跨层注意力残差
        attention_residual_alpha: float = 0.3,  # 残差连接权重（0~1，越大深层越依赖浅层）
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        # 注意力残差配置
        self.use_attention_residual = use_attention_residual
        self.attention_residual_alpha = attention_residual_alpha
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.einsum("bi,j->bij", position_ids.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    query = (query * cos) + (rotate_half(query) * sin)
    key = (key * cos) + (rotate_half(key) * sin)
    return query, key


class LLMIESelfAttention(nn.Module):
    def __init__(self, config: LLMIEConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)

        cos, sin = self.rotary_emb(position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 使用 Flash Attention 加速训练（仅训练模式，推理时使用标准实现）
        if FLASH_ATTENTION_AVAILABLE and self.training:
            # Flash Attention 要求输入格式为 (batch_size, seq_len, 3 * hidden_size)
            qkv = torch.cat([query_states, key_states, value_states], dim=1)
            qkv = qkv.transpose(1, 2).contiguous()  # (batch, seq_len, 3*num_heads, head_dim)
            attn_output = flash_attn_qkvpacked_func(
                qkv, causal=True, dropout_p=0.0
            )
            attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        else:
            # 标准 Attention 实现（兼容推理和不支持 Flash Attention 的环境）
            attn_scores = torch.matmul(query_states, key_states.transpose(-2, -1))
            attn_scores = attn_scores / (self.head_dim ** 0.5)

            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), torch.finfo(attn_scores.dtype).min, device=hidden_states.device),
                diagonal=1,
            )
            attn_scores = attn_scores + causal_mask

            if attention_mask is not None:
                expanded_mask = attention_mask[:, None, None, :].to(attn_scores.dtype)
                attn_scores = attn_scores.masked_fill(expanded_mask == 0, torch.finfo(attn_scores.dtype).min)

            attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_probs, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        return self.o_proj(attn_output)


class LLMIEMLP(nn.Module):
    def __init__(self, config: LLMIEConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class LLMIEDecoderLayer(nn.Module):
    def __init__(self, config: LLMIEConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = LLMIESelfAttention(config)
        self.mlp = LLMIEMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # 注意力残差：深层可以访问浅层的注意力输出
        self.use_attention_residual = config.use_attention_residual
        if self.use_attention_residual and layer_idx > 0:
            # 投影层：将当前层输入投影到与残差相同的维度
            self.residual_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        shallow_hidden: Optional[torch.Tensor] = None,  # 来自浅层的隐藏状态
    ) -> torch.Tensor:
        # ========== 注意力子层 ==========
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
        
        # 注意力残差：深层添加来自浅层的注意力信息
        if self.use_attention_residual and self.layer_idx > 0 and shallow_hidden is not None:
            # 将浅层信息投影后加权融合
            shallow_contrib = self.residual_proj(shallow_hidden)
            # 使用配置中的 alpha 权重融合（默认 0.3）
            alpha_val = getattr(self, '_residual_alpha', 0.3)
            attn_output = attn_output + alpha_val * shallow_contrib
        
        hidden_states = residual + attn_output

        # ========== MLP 子层 ==========
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class LLMIEPreTrainedModel(PreTrainedModel):
    config_class = LLMIEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LLMIEDecoderLayer"]

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

class LLMIEModel(LLMIEPreTrainedModel):
    def __init__(self, config: LLMIEConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LLMIEDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.use_attention_residual = config.use_attention_residual
        self.attention_residual_alpha = config.attention_residual_alpha
        # 为每一层设置残差 alpha（可以逐层调整，这里统一设置）
        for layer in self.layers:
            layer._residual_alpha = config.attention_residual_alpha
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must provide either input_ids or inputs_embeds.")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot provide both input_ids and inputs_embeds.")

        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
            batch_size, seq_len = input_ids.shape
        else:
            hidden_states = inputs_embeds
            batch_size, seq_len = hidden_states.shape[:2]

        if position_ids is None:
            device = hidden_states.device
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # 注意力残差：记录每隔几层的 hidden_states 作为浅层参考
        # 策略：每 3 层保存一个 shallow_hidden，供后续深层使用
        shallow_hidden = None
        shallow_interval = max(1, len(self.layers) // 4)  # 约每 1/4 层保存一次

        for layer_idx, layer in enumerate(self.layers):
            # 在特定层保存 shallow_hidden
            if self.use_attention_residual and layer_idx % shallow_interval == 0:
                shallow_hidden = hidden_states.detach()  # detach 避免梯度回传太远

            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    shallow_hidden if layer_idx > 0 else None,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    shallow_hidden=shallow_hidden if layer_idx > 0 else None,
                )

        hidden_states = self.norm(hidden_states)

        if not return_dict:
            return (hidden_states,)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=None, hidden_states=None)


class LLMIEForCausalLM(LLMIEPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: LLMIEConfig):
        super().__init__(config)
        self.model = LLMIEModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=return_dict,
            **kwargs,
        )
        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        logits = self.lm_head(hidden_states)  # 保持 bf16/fp16，由 Trainer autocast 管理精度

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is not None:
            model_inputs["inputs_embeds"] = kwargs["inputs_embeds"]
        return model_inputs


def build_student_config(tokenizer) -> LLMIEConfig:
    """构建 ~191M 学生模型配置：hidden=768, intermediate=4096, 12层, ChatGLM3 ~64K 词表"""
    return LLMIEConfig(
        vocab_size=len(tokenizer),
        hidden_size=768,                  # 从 512 升级到 768
        intermediate_size=4096,           # 从 1536 升级到 4096
        num_hidden_layers=12,
        num_attention_heads=12,           # 768/12 = 64 head_dim
        num_key_value_heads=12,
        max_position_embeddings=1024,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=True,
    )


@dataclass
class ModelStats:
    total_params: int
    trainable_params: int


def get_model_stats(model: nn.Module) -> ModelStats:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return ModelStats(total_params=total_params, trainable_params=trainable_params)


if __name__ == "__main__":
    from tokenizer_utils import load_tokenizer
    tokenizer, _ = load_tokenizer()

    config = build_student_config(tokenizer)
    model = LLMIEForCausalLM(config)
    stats = get_model_stats(model)

    prompt = "请把这句话润色得更自然：这个方案基本上可以用。"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    print(f"logits shape: {tuple(outputs.logits.shape)}")
    print(f"total params: {stats.total_params / 1e6:.2f}M")
    print(f"trainable params: {stats.trainable_params / 1e6:.2f}M")

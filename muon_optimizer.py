# Muon 优化器实现
# 参考: https://github.com/KellerJordan/Muon
# 适用于 200M 小模型，比 AdamW 数据效率高 ~2 倍

import torch
from torch.optim.optimizer import Optimizer


class Muon(Optimizer):
    """
    Muon: MomentUm Orthogonalized by Newton-Schulz
    
    核心思想：对动量矩阵进行正交化（通过 Newton-Schulz 迭代近似 SVD），
    使得参数更新方向更加均匀，避免某些方向更新过快/过慢。
    
    适用场景：
    - 200M 以下小模型效果最明显
    - 矩阵参数（如 Linear 层的 weight）
    - 不适用于 1D 参数（如 bias、embedding、LayerNorm），这些仍用 AdamW
    
    使用建议：
    - 学习率通常比 AdamW 大 2-5 倍
    - weight_decay 必须设置（对 Muon 的可扩展性至关重要）
    - momentum 推荐 0.95
    """

    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        weight_decay=0.01,
    ):
        """
        Args:
            params: 模型参数
            lr: 学习率（建议比 AdamW 大 2-5 倍，如 AdamW 用 1e-4，Muon 用 2e-4~5e-4）
            momentum: 动量系数（默认 0.95）
            nesterov: 是否使用 Nesterov 动量（默认 True，效果更好）
            ns_steps: Newton-Schulz 迭代步数（默认 5，精度与速度的平衡点）
            weight_decay: 权重衰减（必须设置，推荐 0.01~0.1）
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """执行单步优化"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # 初始化状态
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]

                # 更新动量: m_t = μ * m_{t-1} + g_t
                buf.mul_(momentum).add_(grad)

                # Nesterov 动量: 使用 μ * m_t + g_t 作为更新方向
                if nesterov:
                    update_dir = grad.add(buf, alpha=momentum)
                else:
                    update_dir = buf

                # 只对 >= 2D 的参数进行正交化（矩阵参数）
                if update_dir.ndim >= 2:
                    # Newton-Schulz 迭代近似正交化
                    # 目标: 计算 (M^T M)^{-1/2} M，即对 M 的行进行白化
                    update_dir = newton_schulz(update_dir, steps=ns_steps)
                
                # 应用权重衰减
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # 参数更新
                p.add_(update_dir, alpha=-lr)

        return loss


def newton_schulz(G, steps=5, eps=1e-7):
    """
    Newton-Schulz 迭代：近似计算 (G^T G)^{-1/2} G
    
    这是 Muon 的核心，用多项式迭代代替昂贵的 SVD。
    经过 ns_steps 次迭代后，G 的行向量会被正交化（白化）。
    
    Args:
        G: 梯度矩阵 (..., m, n)
        steps: 迭代次数（5 次是精度与速度的平衡点）
        eps: 防止除零的小常数
    """
    # 确保是 float32（Newton-Schulz 对精度敏感）
    original_dtype = G.dtype
    if G.dtype != torch.float32:
        G = G.float()

    # 归一化：G' = G / ||G||_F，使迭代稳定
    norm = G.norm(p="fro", dim=(-2, -1), keepdim=True)
    G = G / (norm + eps)

    # 迭代系数（来自论文，已优化）
    # 这些系数使得迭代快速收敛到正交矩阵
    a, b, c = 3.4445, -4.7750, 2.0315

    # 初始值: X_0 = G
    X = G

    for _ in range(steps):
        # X_{k+1} = a * X_k + b * X_k @ X_k^T @ X_k + c * X_k @ X_k^T @ X_k @ X_k^T @ X_k
        # 等价于: X_{k+1} = (a * I + b * (X X^T) + c * (X X^T)^2) @ X
        XTX = X.transpose(-2, -1) @ X
        X = a * X + b * X @ XTX + c * X @ XTX @ XTX

    # 反归一化
    X = X * norm

    # 转回原始精度
    if original_dtype != torch.float32:
        X = X.to(original_dtype)

    return X


class HybridOptimizer(Optimizer):
    """
    Muon + AdamW 混合优化器
    
    策略：
    - 矩阵参数 (>=2D, 如 Linear.weight) → Muon（学习率较大）
    - 1D 参数 (bias, LayerNorm, Embedding) → AdamW（学习率较小）
    """

    def __init__(
        self,
        muon_params,
        adamw_params,
        muon_lr=0.02,
        adamw_lr=1e-4,
        muon_momentum=0.95,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        weight_decay=0.01,
        ns_steps=5,
    ):
        """
        Args:
            muon_params: 矩阵参数列表（>=2D）
            adamw_params: 1D 参数列表
            muon_lr: Muon 学习率（建议比 AdamW 大 2-5 倍）
            adamw_lr: AdamW 学习率
            muon_momentum: Muon 动量
            adamw_betas: AdamW betas
            adamw_eps: AdamW eps
            weight_decay: 权重衰减（两个优化器都应用）
            ns_steps: Newton-Schulz 迭代次数
        """
        # 分别创建两个优化器
        self.muon = Muon(
            muon_params,
            lr=muon_lr,
            momentum=muon_momentum,
            weight_decay=0.0,  # 在 Muon.step 里手动处理
            ns_steps=ns_steps,
        )
        self.adamw = torch.optim.AdamW(
            adamw_params,
            lr=adamw_lr,
            betas=adamw_betas,
            eps=adamw_eps,
            weight_decay=weight_decay,
        )
        self.weight_decay = weight_decay
        self.muon_lr = muon_lr

    def zero_grad(self, set_to_none=False):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        # Muon 的 weight_decay 在 step 里处理，但 Hybrid 统一在外部处理
        # 先让两个优化器各自 step
        loss_muon = self.muon.step(closure)
        loss_adamw = self.adamw.step(closure)
        return loss_muon or loss_adamw

    def state_dict(self):
        return {
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])

    @property
    def param_groups(self):
        """兼容 Trainer 的学习率调度"""
        return self.adamw.param_groups + self.muon.param_groups

    @property
    def defaults(self):
        return self.adamw.defaults


def create_optimizer(model, use_muon=True, muon_lr=0.02, adamw_lr=1e-4, weight_decay=0.01):
    """
    为模型创建优化器。
    
    Muon 策略：
    - 矩阵参数 (>=2D, 如 Linear.weight, Conv.weight) → Muon
    - 1D 参数 (bias, LayerNorm, Embedding) → AdamW
    
    Args:
        model: 待优化的模型
        use_muon: 是否使用 Muon（False 则全部用 AdamW）
        muon_lr: Muon 的学习率（建议比 AdamW 大 2-5 倍）
        adamw_lr: AdamW 的学习率
        weight_decay: 权重衰减
    
    Returns:
        配置好的优化器
    """
    if not use_muon:
        # 纯 AdamW 模式
        print(f"使用 AdamW 优化器: lr={adamw_lr}, weight_decay={weight_decay}")
        return torch.optim.AdamW(
            model.parameters(),
            lr=adamw_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=weight_decay,
        )

    # Muon + AdamW 混合模式
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # 矩阵参数 (>=2D) 给 Muon，1D 参数给 AdamW
        if param.ndim >= 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    print(f"使用 Hybrid 优化器: Muon({len(muon_params)} 个矩阵参数) + AdamW({len(adamw_params)} 个 1D 参数)")
    print(f"Muon lr={muon_lr}, AdamW lr={adamw_lr}, weight_decay={weight_decay}")

    return HybridOptimizer(
        muon_params=muon_params,
        adamw_params=adamw_params,
        muon_lr=muon_lr,
        adamw_lr=adamw_lr,
        weight_decay=weight_decay,
    )

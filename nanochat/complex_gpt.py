"""
Schrödinger GPT - 基于量子力学薛定谔演化的 Transformer
Notable features:
- 复数表示: 直角坐标 (xr, xi)，无 atan2
- RoPE: 复数乘法 (xr + i*xi) * (cos + i*sin)
- 相互作用矩阵 S = QK† (因果掩码，严格单向信息流)
- 薛定谔演化: dz = -i * S @ V
- MLP 只作用于模长 (虚时间演化，保持相位不变)
- 最终测量: 投影到实轴 Re(z) = xr

物理意义:
- S 是量子相互作用强度 (非厄米，因为时间有箭头)
- 因果掩码 = 时间单向性 (过去→未来)
- Attention 改变相位关系 (信息传播)
- MLP 改变幅度 (能量/激活强度)
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW
from torch.utils.checkpoint import checkpoint


@dataclass
class HCConfig:
    """HyperConnections configuration"""
    mhc: bool = False
    hc_geometric: bool = False
    dynamic_H: bool = False
    mhc_h_res_proj: str = "sinkhorn"
    sinkhorn_iters: int = 10
    sinkhorn_tau: float = 0.05
    ns_steps: int = 5
    ns_eps: float = 1e-7
    ns_coeffs: tuple = (3.0, -3.2, 1.2)
    manifold_dim: int = 4
    sigma_init: float = 1.0
    sigma_learnable: bool = True
    H_mode: str = "per-token"
    pool_type: str = "last"


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6  # 暂不支持 GQA，必须等于 n_head
    n_embd: int = 768
    window_pattern: str = "L"
    hc_num_streams: int = 1
    hc_disable: bool = False
    gradient_checkpointing: bool = False
    hc: HCConfig = None

    def __post_init__(self):
        if self.hc is None:
            self.hc = HCConfig()
        # 薛定谔注意力暂不支持 GQA，如果不一致则强制设置为 n_head
        if self.n_kv_head != self.n_head:
            print(f"Warning: Schrödinger attention does not support GQA, forcing n_kv_head={self.n_head}")
            object.__setattr__(self, 'n_kv_head', self.n_head)


def norm(x):
    """RMSNorm without learnable params"""
    return F.rms_norm(x, (x.size(-1),))


def complex_norm(xr, xi):
    """对复数的模长做 RMSNorm，保持相位不变"""
    r = torch.sqrt(xr**2 + xi**2 + 1e-8)
    r_normed = norm(r)
    scale = r_normed / r
    return xr * scale, xi * scale


class SchrodingerAttention(nn.Module):
    """
    薛定谔注意力机制 (全直角坐标版本，无 atan2)

    物理图景:
    1. 构造相互作用矩阵 S = Q @ K† (复数点积)
    2. 因果掩码 (时间箭头，严格单向信息流)
    3. 薛定谔演化 dz = -i * S @ V

    关键改进:
    - 不做厄米化 (避免未来信息泄漏到过去)
    - 先掩码后缩放 (保证严格因果性)
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0

        # 复数投影: 权重分为实部和虚部 (参数量 2x)
        self.c_q = nn.Linear(self.n_embd, self.n_embd * 2, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd * 2, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd * 2, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def complex_linear(self, xr, xi, layer):
        """复数线性变换: (xr + i*xi) @ (Wr + i*Wi)"""
        w = layer.weight  # [2*D, D]
        wr, wi = w[:self.n_embd, :], w[self.n_embd:, :]
        out_r = F.linear(xr, wr) - F.linear(xi, wi)
        out_i = F.linear(xr, wi) + F.linear(xi, wr)
        return out_r, out_i

    def apply_rope(self, xr, xi, cos, sin):
        """
        复数 RoPE: (xr + i*xi) * (cos + i*sin)
        = (xr*cos - xi*sin) + i*(xr*sin + xi*cos)
        无需 atan2!
        """
        out_r = xr * cos - xi * sin
        out_i = xr * sin + xi * cos
        return out_r, out_i

    def forward(self, xr, xi, cos_sin, window_size, kv_cache=None):
        """
        Args:
            xr, xi: 直角坐标 [B, T, D]
            cos_sin: (cos, sin) 用于 RoPE
            window_size: (left, right) 滑动窗口大小
        Returns:
            xr_out, xi_out: 直角坐标输出
        """
        B, T, D = xr.shape
        cos, sin = cos_sin

        # 1. 复数 QKV 投影
        qr, qi = self.complex_linear(xr, xi, self.c_q)
        kr, ki = self.complex_linear(xr, xi, self.c_k)
        vr, vi = self.complex_linear(xr, xi, self.c_v)

        # 2. 多头变换 [B, T, D] -> [B, T, n_head, head_dim] -> [B, n_head, T, head_dim]
        qr = qr.view(B, T, self.n_head, self.head_dim)
        qi = qi.view(B, T, self.n_head, self.head_dim)
        kr = kr.view(B, T, self.n_head, self.head_dim)
        ki = ki.view(B, T, self.n_head, self.head_dim)
        vr = vr.view(B, T, self.n_head, self.head_dim)
        vi = vi.view(B, T, self.n_head, self.head_dim)

        # 3. 复数 RoPE (只对 Q, K)
        # cos, sin: [1, T, 1, head_dim//2] -> 需要扩展
        # 对每个 head 的前半维度应用 RoPE
        d2 = self.head_dim // 2
        # Q
        qr1, qr2 = qr[..., :d2], qr[..., d2:]
        qi1, qi2 = qi[..., :d2], qi[..., d2:]
        qr1, qi1 = self.apply_rope(qr1, qi1, cos, sin)
        qr = torch.cat([qr1, qr2], dim=-1)
        qi = torch.cat([qi1, qi2], dim=-1)
        # K
        kr1, kr2 = kr[..., :d2], kr[..., d2:]
        ki1, ki2 = ki[..., :d2], ki[..., d2:]
        kr1, ki1 = self.apply_rope(kr1, ki1, cos, sin)
        kr = torch.cat([kr1, kr2], dim=-1)
        ki = torch.cat([ki1, ki2], dim=-1)

        # 4. transpose to [B, n_head, T, head_dim]
        qr, qi = qr.transpose(1, 2), qi.transpose(1, 2)
        kr, ki = kr.transpose(1, 2), ki.transpose(1, 2)
        vr, vi = vr.transpose(1, 2), vi.transpose(1, 2)

        # 5. QK norm (对复数模长做 norm，保持相位)
        qr, qi = complex_norm(qr, qi)
        kr, ki = complex_norm(kr, ki)

        # 6. 构造相互作用矩阵 S = Q @ K†
        # 不做厄米化，直接因果掩码，避免未来信息泄漏
        # S = Q @ K†, K† = (kr.T, -ki.T)
        sr = torch.matmul(qr, kr.transpose(-1, -2)) + torch.matmul(qi, ki.transpose(-1, -2))
        si = torch.matmul(qi, kr.transpose(-1, -2)) - torch.matmul(qr, ki.transpose(-1, -2))

        # 7. 因果掩码 (先掩码，保证严格因果性)
        left_window = window_size[0]

        # 创建因果掩码 (下三角)
        mask = torch.ones(T, T, device=xr.device, dtype=torch.bool).tril_()
        if 0 < left_window < T:
            # 滑动窗口: 只保留最近 left_window 个位置
            mask = mask & torch.ones(T, T, device=xr.device, dtype=torch.bool).triu_(1 - left_window)

        sr = sr.masked_fill(~mask, 0)
        si = si.masked_fill(~mask, 0)

        # 归一化: 1/√d，避免累积效应 (在掩码后缩放)
        scale = self.head_dim ** -0.5
        sr = sr * scale
        si = si * scale

        # 对模长做 Softmax 归一化
        S_abs = torch.sqrt(sr ** 2 + si ** 2 + 1e-8)
        A = torch.softmax(S_abs, dim=-1)
        hr = sr * A
        hi = si * A

        # 8. 薛定谔演化: dz = -i * S @ V (不再是厄米特 H，而是因果 S)
        # S @ V = (sr + i*si) @ (vr + i*vi) = (sr@vr - si@vi) + i*(sr@vi + si@vr)
        # -i * (a + ib) = b - ia
        # 所以 dz_r = sr@vi + si@vr, dz_i = si@vi - sr@vr
        dz_r = torch.matmul(hr, vi) + torch.matmul(hi, vr)
        dz_i = torch.matmul(hi, vi) - torch.matmul(hr, vr)

        # 9. 还原维度
        dz_r = dz_r.transpose(1, 2).contiguous().view(B, T, D)
        dz_i = dz_i.transpose(1, 2).contiguous().view(B, T, D)

        # 10. 输出投影
        out_r = self.c_proj(dz_r)
        out_i = self.c_proj(dz_i)

        # 注意: 残差连接已移到 Block 外部，这里只返回 delta
        return out_r, out_i


class SchrodingerMLP(nn.Module):
    """
    MLP 只作用于模长 (虚时间演化) - 直角坐标版本

    物理直觉:
    - Attention (实时间): 改变相位关系 (信息传播)
    - MLP (虚时间): 改变幅度 (能量/激活强度)
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, xr, xi):
        """
        Args:
            xr, xi: 直角坐标 [B, T, D] (已经过 pre-MLP complex_norm)
        Returns:
            delta_xr, delta_xi: 残差 delta (相位不变，只改模长)
        """
        # 计算模长 (输入已被 complex_norm 归一化，但仍需计算 r)
        r = torch.sqrt(xr**2 + xi**2 + 1e-8)

        # MLP 直接作用于 r (已被 complex_norm 归一化，不需要额外 norm)
        delta_r = self.c_fc(r)
        delta_r = F.relu(delta_r).square()  # ReLU^2 激活 (与标准 GPT 一致)
        delta_r = self.c_proj(delta_r)

        # 新模长 = 旧模长 + delta (用 softplus 确保 > 0)
        r_new = F.softplus(r + delta_r, beta=1.0, threshold=20.0)

        # 计算相对变化: (r_new - r) / r，保持相位不变
        # 注意: 残差连接已移到 Block 外部，这里只返回 delta
        scale = (r_new - r) / r

        return xr * scale, xi * scale


class Block(nn.Module):
    """Transformer Block (薛定谔版本) - 直角坐标，无 atan2"""

    def __init__(self, config, layer_idx, init_hc=None):
        super().__init__()
        self.attn = SchrodingerAttention(config, layer_idx)
        self.mlp = SchrodingerMLP(config)

    def forward(self, xr, xi, cos_sin, window_size, kv_cache=None):
        # Attention: pre-norm + residual (对齐标准 GPT)
        xr_n, xi_n = complex_norm(xr, xi)
        dr, di = self.attn(xr_n, xi_n, cos_sin, window_size, kv_cache)
        xr, xi = xr + dr, xi + di  # 残差连接到原始输入

        # MLP: pre-norm + residual
        xr_n, xi_n = complex_norm(xr, xi)
        dr, di = self.mlp(xr_n, xi_n)
        xr, xi = xr + dr, xi + di  # 残差连接到原始输入

        return xr, xi


class GPT(nn.Module):
    """Schrödinger GPT"""

    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config

        # 计算滑动窗口大小
        self.window_sizes = self._compute_window_sizes(config)

        # Pad vocab for efficiency
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")

        # 注意: 薛定谔版本暂不支持 HyperConnections
        # 如果启用了 HC，打印警告
        if config.hc_num_streams > 1 and not config.hc_disable:
            print0(f"Warning: Schrödinger GPT does not support HyperConnections yet, ignoring hc_num_streams={config.hc_num_streams}")

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)

        # 预计算 RoPE 的 cos/sin (与 gpt.py 保持一致)
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        """初始化权重"""
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5

        for block in self.transformer.h:
            # 复数投影层: 初始化 Wr 和 Wi
            for layer in [block.attn.c_q, block.attn.c_k, block.attn.c_v]:
                # Wr (实部): uniform
                torch.nn.init.uniform_(layer.weight[:n_embd, :], -s, s)
                # Wi (虚部): 小一点，让初始时接近实数
                torch.nn.init.uniform_(layer.weight[n_embd:, :], -s * 0.1, s * 0.1)

            # 输出投影: zeros
            torch.nn.init.zeros_(block.attn.c_proj.weight)

            # MLP
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # 重新计算 RoPE cos/sin
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to bf16
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        """预计算 RoPE 的 cos/sin (与 gpt.py 保持一致)"""
        if device is None:
            device = self.transformer.wte.weight.device

        # 频率
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))

        # 位置
        t = torch.arange(seq_len, dtype=torch.float32, device=device)

        # 角度 [seq_len, head_dim//2]
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()

        if device.type == "cuda":
            cos, sin = cos.bfloat16(), sin.bfloat16()

        # [1, seq_len, 1, head_dim//2] 用于广播
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        """计算滑动窗口大小"""
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)

        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}

        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """估算 FLOPs (复数运算约 2x)"""
        nparams = sum(p.numel() for p in self.parameters())
        nparams_exclude = self.transformer.wte.weight.numel()
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len

        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            # 复数注意力约 2x FLOPs
            attn_flops += 24 * h * q * effective_seq

        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        return sum(p.numel() for p in self.parameters())

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        """设置优化器 (与 gpt.py 保持一致)"""
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        matrix_params = []
        embedding_params = []
        lm_head_params = list(self.lm_head.parameters())

        embedding_params.extend(list(self.transformer.wte.parameters()))

        for block in self.transformer.h:
            for name, param in block.named_parameters():
                if "branch.attn" in name or "branch.mlp" in name or \
                   "attn.c_" in name or "mlp.c_" in name or \
                   "proj_head.weight" in name or \
                   "dynamic_alpha_fn" in name or \
                   "dynamic_beta_fn" in name or \
                   "H_res_proj.weight" in name or \
                   "H_pre_proj.weight" in name or \
                   "H_post_proj.weight" in name:
                    matrix_params.append(param)
                else:
                    embedding_params.append(param)

        print0(f"Parameter grouping:")
        print0(f"  Matrix params (Muon): {len(matrix_params)} tensors, {sum(p.numel() for p in matrix_params):,} elements")
        print0(f"  Embedding params (AdamW): {len(embedding_params)} tensors, {sum(p.numel() for p in embedding_params):,} elements")
        print0(f"  LM head params (AdamW): {len(lm_head_params)} tensors, {sum(p.numel() for p in lm_head_params):,} elements")

        total_params_check = sum(p.numel() for p in matrix_params) + sum(p.numel() for p in embedding_params) + sum(p.numel() for p in lm_head_params)
        all_params = sum(p.numel() for p in self.parameters())
        print0(f"  Total: {total_params_check:,} / {all_params:,} parameters")
        assert total_params_check == all_params

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling LR ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=adam_betas, eps=1e-10, weight_decay=0.0)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)

        muon_kwargs = dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)

        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # 获取 RoPE cos/sin
        assert T <= self.cos.size(1), f"Sequence length {T} exceeds rotary cache {self.cos.size(1)}"
        # NOTE: kv_cache 接口保留但当前被忽略 ——
        # 由于当前归一化（complex_norm）依赖于时间序列长度 T，增量缓存会改变行为。
        # 因此暂时不使用缓存，始终从序列开头计算 RoPE 和 K/V。若传入 kv_cache，则发出警告。
        if kv_cache is not None:
            import warnings as _warnings
            _warnings.warn("kv_cache passed to GPT.forward is currently ignored; recomputing K/V each call")
        T0 = 0
        cos_sin = (self.cos[:, T0:T0+T], self.sin[:, T0:T0+T])

        # Embedding
        x = self.transformer.wte(idx)  # [B, T, D]

        # 初始化: 实部 = embedding, 虚部 = 0
        xr = x
        xi = torch.zeros_like(x)

        # Transformer forward
        def transformer_forward(xr, xi, blocks, cos_sin, window_sizes, kv_cache):
            # 逐层 (每个 block 内部会先做 complex_norm)
            for i, block in enumerate(blocks):
                xr, xi = block(xr, xi, cos_sin, window_sizes[i], kv_cache)

            # 最终 norm
            xr, xi = complex_norm(xr, xi)

            return xr, xi

        if self.config.gradient_checkpointing and self.training and kv_cache is None:
            xr, xi = checkpoint(
                transformer_forward, xr, xi, self.transformer.h,
                cos_sin, self.window_sizes, kv_cache,
                use_reentrant=False
            )
        else:
            xr, xi = transformer_forward(xr, xi, self.transformer.h, cos_sin, self.window_sizes, kv_cache)

        # 测量: 投影到实轴 Re(z) = xr
        observable = xr

        # lm_head
        softcap = 15
        logits = self.lm_head(observable)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction
            )
            return loss
        else:
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """自回归生成"""
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        ids = torch.tensor([tokens], dtype=torch.long, device=device)

        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)

            ids = torch.cat((ids, next_ids), dim=1)
            yield next_ids.item()

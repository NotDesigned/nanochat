"""
Geometric-Induced Diffusion Hyper-Connections

核心思想：用几何归纳偏置替代全参数预测
- 步骤1：坐标投影 - 将高维流投影到低维流形空间
- 步骤2：核函数构建 - 基于坐标距离计算亲密度（热核/RBF）
- 步骤3：Sinkhorn归一化 - 保证双随机性（质量守恒）

ein notation:
b - batch
d - feature dimension
s - residual streams
k - manifold dimension (projection dim)
"""

from __future__ import annotations
from typing import Callable

from functools import partial
from random import randrange
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Module
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.utils.checkpoint import checkpoint

from einops import rearrange, einsum
from einops.layers.torch import Rearrange, Reduce

from hyper_connections.hyper_connections import Residual, StreamEmbed, RMSNorm


# helper functions

def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def add(x, y):
    return x + y


# Sinkhorn algorithm in log-space for numerical stability

def sinkhorn_log(logits: Tensor, num_iters: int = 10, tau: float = 0.05) -> Tensor:
    """
    Sinkhorn-Knopp algorithm in log-space.
    Projects a matrix onto the doubly-stochastic manifold.
    
    Args:
        logits: (n, n) matrix of log-affinities
        num_iters: number of Sinkhorn iterations
        tau: temperature parameter
    
    Returns:
        (n, n) doubly-stochastic matrix
    """
    n = logits.shape[-1]
    Z = logits / tau
    log_marginal = torch.full((n,), -math.log(n), device=logits.device, dtype=logits.dtype)

    u = torch.zeros(n, device=Z.device, dtype=Z.dtype)
    v = torch.zeros(n, device=Z.device, dtype=Z.dtype)

    for _ in range(num_iters):
        u = log_marginal - torch.logsumexp(Z + v.unsqueeze(0), dim=1)
        v = log_marginal - torch.logsumexp(Z + u.unsqueeze(1), dim=0)

    return torch.exp(Z + u.unsqueeze(1) + v.unsqueeze(0)) * n


# main functions

def get_expand_reduce_stream_functions(
    num_streams, add_stream_embed=False, dim=None, disable=False
):
    if num_streams == 1 or disable:
        return (nn.Identity(), nn.Identity())

    if add_stream_embed:
        assert exists(dim), (
            "`dim` must be passed into get_init_and_expand_reduce_stream_functions "
            "for returning an expansion function with stream embeddings added"
        )
        expand_fn = StreamEmbed(num_streams, dim, expand_to_streams=True)
    else:
        expand_fn = Reduce(
            pattern="b ... -> (b s) ...", reduction="repeat", s=num_streams
        )

    reduce_fn = Reduce(pattern="(b s) ... -> b ...", reduction="sum", s=num_streams)

    return expand_fn, reduce_fn


def get_init_and_expand_reduce_stream_functions(
    num_streams, dim=None, add_stream_embed=False, disable=None, **kwargs
):
    disable = default(disable, num_streams == 1)

    hyper_conn_klass = GeometricHyperConnections if not disable else Residual

    init_hyper_conn_fn = partial(hyper_conn_klass, num_streams, **kwargs)
    expand_reduce_fns = get_expand_reduce_stream_functions(
        num_streams, add_stream_embed=add_stream_embed, dim=dim, disable=disable
    )

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim=dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)


# Geometric Hyper-Connections

class GeometricHyperConnections(Module):
    """
    几何诱导扩散超连接
    
    核心改动：将 H 矩阵的生成从"全参数预测"改为"几何诱导"
    
    原始方式: RMSNorm -> Linear(d -> n*n) -> Sinkhorn
    新方式:
        1. 坐标投影: u_i (d维) -> p_i (k维)  
        2. 核函数: H_tilde[i,j] = exp(-||p_i - p_j||^2 / (2*sigma^2))
        3. Sinkhorn归一化: H = Sinkhorn(H_tilde)
    """
    
    def __init__(
        self,
        num_residual_streams: int,
        *,
        dim: int,
        branch: Module | None = None,
        layer_index: int | None = None,
        # 几何参数
        manifold_dim: int = 4,  # 流形空间维度 k
        sigma_init: float = 1.0,  # RBF 核的初始温度
        sigma_learnable: bool = True,  # sigma 是否可学习
        # Sinkhorn 参数
        sinkhorn_iters: int = 10,
        sinkhorn_tau: float = 0.05,
        sinkhorn_tolerance: float = 1e-3,  # 提前终止的收敛阈值
        # H生成模式
        H_mode: str = "per-token",  # "per-token", "per-seq", "chunk"
        chunk_size: int = 8,  # 仅在H_mode=="chunk"时生效
        pool_type: str = "mean",  # 池化方式: "mean"或"max"
        # 其他
        channel_first: bool = False,
        dropout: float = 0.0,
        residual_transform: Module | None = None,
        add_branch_out_to_residual: bool = True,
        depth_residual_fn: Callable = add,
        gradient_checkpointing: bool = False,
    ):
        """
        Args:
            num_residual_streams: 残差流数量 n
            dim: 特征维度 d
            manifold_dim: 流形空间维度 k（投影后的坐标维度）
            sigma_init: RBF 核的初始 sigma 值
            sigma_learnable: sigma 是否作为可学习参数
            sinkhorn_iters: Sinkhorn 最大迭代次数
            sinkhorn_tau: Sinkhorn 温度参数
            sinkhorn_tolerance: 提前终止的收敛阈值
        """
        super().__init__()

        assert num_residual_streams > 1, "GeometricHyperConnections requires num_residual_streams > 1"
        
        self.num_residual_streams = num_residual_streams
        self.dim = dim
        self.manifold_dim = manifold_dim
        self.branch = branch
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_tau = sinkhorn_tau
        self.channel_first = channel_first
        self.add_branch_out_to_residual = add_branch_out_to_residual
        self.depth_residual_fn = depth_residual_fn
        self.gradient_checkpointing = gradient_checkpointing
        self.H_mode = H_mode
        self.chunk_size = chunk_size
        self.pool_type = pool_type
        init_residual_index = (
            default(layer_index, randrange(num_residual_streams)) % num_residual_streams
        )

        # ===== 步骤1: 坐标投影头 =====
        # 将每个流从 d 维投影到 k 维流形空间
        self.norm = RMSNorm(dim)
        self.proj_head = nn.Linear(dim, manifold_dim, bias=False)
        
        # 初始化：使用较小的值，让初始坐标比较集中
        nn.init.normal_(self.proj_head.weight, std=0.02)

        # ===== 步骤2: 核函数参数 =====
        # sigma: RBF 核的带宽参数（温度）
        if sigma_learnable:
            self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma_init)))
        else:
            self.register_buffer('log_sigma', torch.tensor(math.log(sigma_init)))

        # ===== H_pre 和 H_post（保持原有设计）=====
        # H_pre: 选择哪个流作为 branch 输入
        H_pre_init = torch.full((num_residual_streams,), -8.0)
        H_pre_init[init_residual_index] = 0.0
        self.H_pre_logits = nn.Parameter(H_pre_init)

        # H_post: 将 branch 输出分配回各流
        if add_branch_out_to_residual:
            self.H_post_logits = nn.Parameter(torch.zeros(num_residual_streams))

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Residual transform
        self.residual_transform = default(residual_transform, nn.Identity())

    @property
    def sigma(self) -> Tensor:
        """获取当前的 sigma 值"""
        return self.log_sigma.exp()

    def compute_geometric_H(self, residuals: Tensor) -> Tensor:
        """
         计算几何诱导的 H 矩阵（位置相关）
        Returns:
            H: (b, T, s, s) for per-token; (b, s, s) for per-seq; (b, num_chunks, s, s) for chunk
            coords: (b, T, s, k) for per-token; (b, s, k) for per-seq; (b, num_chunks, s, k) for chunk
        """
        b, T, s, d = residuals.shape
        H_mode = getattr(self, 'H_mode', 'per-token')
        pool_type = getattr(self, 'pool_type', 'mean')
        chunk_size = getattr(self, 'chunk_size', 8)

        def pool(x, dim):
            if pool_type == 'mean':
                return x.mean(dim=dim, keepdim=True)
            elif pool_type == 'max':
                return x.max(dim=dim, keepdim=True).values
            else:
                raise ValueError(f"Unknown pool_type: {pool_type}")

        if H_mode == 'per-token':
            normed = self.norm(residuals)
            coords = self.proj_head(normed)
            coords_i = coords.unsqueeze(-2)
            coords_j = coords.unsqueeze(-3)
            diff = coords_i - coords_j
            dist_sq = (diff ** 2).sum(dim=-1)
            sigma = self.sigma
            H_tilde_logits = -dist_sq / (2 * sigma ** 2)
            H = self.batched_sinkhorn(H_tilde_logits)
            return H, coords
        elif H_mode == 'per-seq':
            pooled = pool(residuals, dim=1)  # (b, 1, s, d)
            normed = self.norm(pooled)
            coords_seq = self.proj_head(normed).squeeze(1)  # (b, s, k)
            coords_i = coords_seq.unsqueeze(-2)
            coords_j = coords_seq.unsqueeze(-3)
            diff = coords_i - coords_j
            dist_sq = (diff ** 2).sum(dim=-1)
            sigma = self.sigma
            H_tilde_logits = -dist_sq / (2 * sigma ** 2)
            H = self.batched_sinkhorn(H_tilde_logits)  # (b, s, s)
            return H, coords_seq
        elif H_mode == 'chunk':
            num_chunks = (T + chunk_size - 1) // chunk_size
            pad_len = num_chunks * chunk_size - T
            if pad_len > 0:
                pad_shape = list(residuals.shape)
                pad_shape[1] = pad_len
                pad_tensor = torch.zeros(pad_shape, device=residuals.device, dtype=residuals.dtype)
                residuals_padded = torch.cat([residuals, pad_tensor], dim=1)
            else:
                residuals_padded = residuals
            residuals_chunk = residuals_padded.view(b, num_chunks, chunk_size, s, d)
            pooled = pool(residuals_chunk, dim=2)  # (b, num_chunks, 1, s, d)
            normed = self.norm(pooled.squeeze(2))  # (b, num_chunks, s, d)
            coords_chunk = self.proj_head(normed)  # (b, num_chunks, s, k)
            coords_i = coords_chunk.unsqueeze(-2)
            coords_j = coords_chunk.unsqueeze(-3)
            diff = coords_i - coords_j
            dist_sq = (diff ** 2).sum(dim=-1)
            sigma = self.sigma
            H_tilde_logits = -dist_sq / (2 * sigma ** 2)
            H = self.batched_sinkhorn(H_tilde_logits)  # (b, num_chunks, s, s)
            return H, coords_chunk
        else:
            raise ValueError(f"Unknown H_mode: {H_mode}")

    def batched_sinkhorn(self, logits: Tensor) -> Tensor:
        """
        批量 Sinkhorn 归一化（带提前终止优化）

        Args:
            logits: (..., s, s) 任意 batch 维度的 logits

        Returns:
            (..., s, s) 双随机矩阵
        """
        s = logits.shape[-1]
        Z = logits / self.sinkhorn_tau
        log_marginal = torch.full((s,), -math.log(s), device=logits.device, dtype=logits.dtype)

        # u, v: (..., s)
        u = torch.zeros(logits.shape[:-1], device=Z.device, dtype=Z.dtype)
        v = torch.zeros(logits.shape[:-1], device=Z.device, dtype=Z.dtype)

        for _ in range(self.sinkhorn_iters):
            # Z: (..., s, s), v: (..., s) -> v.unsqueeze(-2): (..., 1, s)
            u = log_marginal - torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
            v = log_marginal - torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)

        return torch.exp(Z + u.unsqueeze(-1) + v.unsqueeze(-2)) * s

    def width_connection(self, residuals: Tensor):
        """宽度连接：计算 branch 输入和混合后的残差"""
        streams = self.num_residual_streams
        maybe_transformed_residuals = self.residual_transform(residuals)
        if self.channel_first:
            residuals = rearrange(residuals, "b d ... -> b ... d")
            maybe_transformed_residuals = rearrange(maybe_transformed_residuals, "b d ... -> b ... d")
        residuals = rearrange(residuals, "(b s) T d -> b T s d", s=streams)
        residuals_mixed_source = rearrange(maybe_transformed_residuals, "(b s) T d -> b T s d", s=streams)

        H_mode = getattr(self, 'H_mode', 'per-token')
        if self.gradient_checkpointing and self.training:
            H_res, coords = checkpoint(self.compute_geometric_H, residuals, use_reentrant=False)
        else:
            H_res, coords = self.compute_geometric_H(residuals)

        # 延迟计算：不在此处计算 residuals_mixed，只保存必要的参数
        # depth_connection 中按需计算混合
        mixing_params = dict(
            H_mode=H_mode,
            H_res=H_res,
            residuals_mixed_source=residuals_mixed_source,
            chunk_size=getattr(self, 'chunk_size', 8) if H_mode == 'chunk' else None,
        )

        H_pre = F.softmax(self.H_pre_logits, dim=-1)
        H_post = None
        if self.add_branch_out_to_residual:
            H_post = F.softmax(self.H_post_logits, dim=-1)

        H_pre_expanded = H_pre.view(1, 1, -1, 1)
        branch_input = (residuals * H_pre_expanded).sum(dim=2)

        if getattr(self, "collect_stats", False):
            with torch.no_grad():
                # 统计时需要广播 H_res 来计算
                if H_mode == 'per-seq':
                    H_res_broadcast = H_res.unsqueeze(1).expand(-1, residuals.shape[1], -1, -1)
                elif H_mode == 'chunk':
                    H_res_broadcast = H_res.unsqueeze(2).expand(-1, num_chunks, chunk_size, -1, -1)
                    H_res_broadcast = H_res_broadcast.contiguous().view(H_res.shape[0], num_chunks * chunk_size, H_res.shape[2], H_res.shape[3])
                    H_res_broadcast = H_res_broadcast[:, :residuals.shape[1], :, :]
                else:
                    H_res_broadcast = H_res
                stats = dict(
                    h_res_min=H_res_broadcast.min(),
                    h_res_max=H_res_broadcast.max(),
                    h_res_diag_mean=H_res_broadcast.diagonal(dim1=-2, dim2=-1).mean(),
                    sigma=self.sigma,
                    coords_norm=coords.norm(dim=-1).mean(),
                    h_pre_entropy=-(H_pre * H_pre.log().clamp(min=-100)).sum(),
                )
                if H_post is not None:
                    stats["h_post_entropy"] = -(H_post * H_post.log().clamp(min=-100)).sum()
                self.last_stats = {k: v.detach() for k, v in stats.items()}

        if self.channel_first:
            branch_input = rearrange(branch_input, "b ... d -> b d ...")

        return (
            branch_input,
            maybe_transformed_residuals,
            dict(beta=H_post, mixing_params=mixing_params),
        )

    def depth_connection(
        self, 
        branch_output: Tensor, 
        residuals: Tensor, 
        *, 
        beta: Tensor, 
        mixing_params: dict
    ) -> Tensor:
        """深度连接：将 branch 输出混合回残差流，按需计算 residuals_mixed"""
        assert self.add_branch_out_to_residual
        assert beta is not None
        assert mixing_params is not None

        if self.channel_first:
            branch_output = rearrange(branch_output, "b d ... -> b ... d")

        # 将 branch 输出按 H_post 分配到各个流
        # branch_output: (b, T, d), beta: (s,)
        branch_to_streams = einsum(branch_output, beta, "b T d, s -> b T s d")
        
        # 按需计算 residuals_mixed
        H_mode = mixing_params['H_mode']
        H_res = mixing_params['H_res']
        residuals_mixed_source = mixing_params['residuals_mixed_source']
        streams = self.num_residual_streams
        
        if H_mode == 'per-seq':
            residuals_trans = residuals_mixed_source.transpose(1, 2)  # (b, s, T, d)
            mixed_trans = einsum(H_res, residuals_trans, "b s t, b s T d -> b t T d")
            residuals_mixed = mixed_trans.transpose(1, 2)  # (b, T, s, d)
        elif H_mode == 'chunk':
            chunk_size = mixing_params['chunk_size']
            num_chunks = H_res.shape[1]
            residuals_chunk = residuals_mixed_source.view(residuals_mixed_source.shape[0], num_chunks, chunk_size, streams, -1)
            residuals_mixed_chunk = einsum(H_res, residuals_chunk, "b c s t, b c k s d -> b c k t d")
            residuals_mixed = residuals_mixed_chunk.contiguous().view(residuals_mixed_source.shape[0], num_chunks * chunk_size, streams, -1)
            residuals_mixed = residuals_mixed[:, :residuals_mixed_source.shape[1], :, :]  # 裁剪 padding
        else:
            # per-token: H_res 已经是 (b, T, s, s)
            residuals_mixed = einsum(H_res, residuals_mixed_source, "b T s t, b T s d -> b T t d")
        
        # 与混合后的残差相加
        # residuals_mixed: (b, T, s, d)
        output = residuals_mixed + branch_to_streams
        
        # 合并回 (b*s, T, d) 格式
        output = rearrange(output, "b T s d -> (b s) T d")

        if self.channel_first:
            output = rearrange(output, "b ... d -> b d ...")

        return self.dropout(output)

    def decorate_branch(self, branch: Callable):
        """装饰器模式：包装 branch 函数"""
        assert not exists(self.branch), "branch was already wrapped on init"

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)
            branch_output = branch(branch_input, *args, **kwargs)
            residual = add_residual(branch_output)
            return residual

        return forward_and_add_residual

    def forward(self, residuals: Tensor, *branch_args, **branch_kwargs):
        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            if not self.add_branch_out_to_residual:
                return branch_out

            (branch_out, *rest), tree_spec = tree_flatten(branch_out)
            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)
            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn
        if self.gradient_checkpointing and self.training:
            branch_output = checkpoint(
                self.branch, branch_input, *branch_args, **branch_kwargs, use_reentrant=False
            )
        else:
            branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)
        return add_residual_fn(branch_output)


# 添加静态方法
GeometricHyperConnections.get_expand_reduce_stream_functions = staticmethod(
    get_expand_reduce_stream_functions
)
GeometricHyperConnections.get_init_and_expand_reduce_stream_functions = staticmethod(
    get_init_and_expand_reduce_stream_functions
)

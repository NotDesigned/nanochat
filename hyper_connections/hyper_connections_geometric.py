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
    num_streams, dim=None, add_stream_embed=False, disable=None, gradient_checkpointing=False
):
    disable = default(disable, num_streams == 1)

    hyper_conn_klass = GeometricHyperConnections if not disable else Residual

    init_hyper_conn_fn = partial(hyper_conn_klass, num_streams, gradient_checkpointing=gradient_checkpointing)
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
        self.sinkhorn_tolerance = sinkhorn_tolerance
        self.channel_first = channel_first
        self.add_branch_out_to_residual = add_branch_out_to_residual
        self.depth_residual_fn = depth_residual_fn
        self.gradient_checkpointing = gradient_checkpointing
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
        
        Args:
            residuals: (batch, seq, streams, dim) 残差流
            
        Returns:
            H: (batch, seq, streams, streams) 每个位置独立的双随机矩阵
            coords: (batch, seq, streams, k) 流形坐标
        """
        # residuals: (b, T, s, d)
        b, T, s, d = residuals.shape
        
        # 步骤1: 坐标投影
        # 对每个位置的每个流独立投影到流形空间
        normed = self.norm(residuals)  # (b, T, s, d)
        coords = self.proj_head(normed)  # (b, T, s, k)
        
        # 步骤2: RBF 核函数
        # 计算每个位置上流与流之间的距离
        # coords: (b, T, s, k)
        coords_i = coords.unsqueeze(-2)  # (b, T, s, 1, k)
        coords_j = coords.unsqueeze(-3)  # (b, T, 1, s, k)
        diff = coords_i - coords_j  # (b, T, s, s, k)
        dist_sq = (diff ** 2).sum(dim=-1)  # (b, T, s, s)
        
        # 热核 / RBF: exp(-||p_i - p_j||^2 / (2*sigma^2))
        sigma = self.sigma
        H_tilde_logits = -dist_sq / (2 * sigma ** 2)  # (b, T, s, s)
        
        # 步骤3: Sinkhorn 归一化（对每个位置独立进行）
        H = self.batched_sinkhorn(H_tilde_logits)  # (b, T, s, s)
        
        return H, coords

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

        for iter_num in range(self.sinkhorn_iters):
            u_prev = u
            # Z: (..., s, s), v: (..., s) -> v.unsqueeze(-2): (..., 1, s)
            u = log_marginal - torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
            v = log_marginal - torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)

            # 提前终止检查：在前几次迭代后检查收敛
            if iter_num > 3:
                max_change = (u - u_prev).abs().max()
                if max_change < self.sinkhorn_tolerance:
                    break

        return torch.exp(Z + u.unsqueeze(-1) + v.unsqueeze(-2)) * s

    def width_connection(self, residuals: Tensor):
        """宽度连接：计算 branch 输入和混合后的残差"""
        streams = self.num_residual_streams
        
        maybe_transformed_residuals = self.residual_transform(residuals)

        # 处理 channel first
        if self.channel_first:
            residuals = rearrange(residuals, "b d ... -> b ... d")
            maybe_transformed_residuals = rearrange(maybe_transformed_residuals, "b d ... -> b ... d")

        # 分离出各个流
        # residuals: (b*s, T, d) -> (b, T, s, d)
        residuals = rearrange(residuals, "(b s) T d -> b T s d", s=streams)
        # 同样处理transformed版本，用于后续混合
        residuals_mixed_source = rearrange(maybe_transformed_residuals, "(b s) T d -> b T s d", s=streams)

        # 计算几何诱导的 H_res 矩阵（位置相关）
        # H_res: (b, T, s, s)
        H_res, coords = self.compute_geometric_H(residuals)

        # H_pre: 选择 branch 输入
        H_pre = F.softmax(self.H_pre_logits, dim=-1)

        # H_post: 分配 branch 输出
        H_post = None
        if self.add_branch_out_to_residual:
            H_post = F.softmax(self.H_post_logits, dim=-1)

        # 应用 H_res 混合残差流（位置相关）
        # 使用transformed版本进行混合，以保证residual_transform生效
        # residuals_mixed_source: (b, T, s, d), H_res: (b, T, s, t)
        # 对每个位置独立进行流混合
        residuals_mixed = einsum(H_res, residuals_mixed_source, "b T s t, b T s d -> b T t d")
        
        # 计算 branch 输入
        # H_pre: (s,) 静态选择向量

        # 优化：用矩阵乘法替代einsum
        # 原: branch_input = einsum(H_pre, residuals, "s, b T s d -> b T d")
        # 相当于对s维度做加权求和: sum(H_pre[i] * residuals[:, :, i, :])
        # H_pre: (s,), residuals: (b, T, s, d)
        H_pre_expanded = H_pre.view(1, 1, -1, 1)  # (1, 1, s, 1)
        branch_input = (residuals * H_pre_expanded).sum(dim=2)  # (b, T, d)

        # 收集统计信息（用于调试）
        if getattr(self, "collect_stats", False):
            with torch.no_grad():
                stats = dict(
                    h_res_min=H_res.min(),
                    h_res_max=H_res.max(),
                    h_res_diag_mean=H_res.diagonal(dim1=-2, dim2=-1).mean(),
                    sigma=self.sigma,
                    coords_norm=coords.norm(dim=-1).mean(),
                    h_pre_entropy=-(H_pre * H_pre.log().clamp(min=-100)).sum(),
                )
                if H_post is not None:
                    stats["h_post_entropy"] = -(H_post * H_post.log().clamp(min=-100)).sum()
                self.last_stats = {k: v.detach() for k, v in stats.items()}

        # 恢复 channel first
        if self.channel_first:
            branch_input = rearrange(branch_input, "b ... d -> b d ...")

        return (
            branch_input,
            maybe_transformed_residuals,
            dict(beta=H_post, residuals_mixed=residuals_mixed),
        )

    def depth_connection(
        self, 
        branch_output: Tensor, 
        residuals: Tensor, 
        *, 
        beta: Tensor, 
        residuals_mixed: Tensor
    ) -> Tensor:
        """深度连接：将 branch 输出混合回残差流"""
        assert self.add_branch_out_to_residual
        assert beta is not None
        assert residuals_mixed is not None

        if self.channel_first:
            branch_output = rearrange(branch_output, "b d ... -> b ... d")

        # 将 branch 输出按 H_post 分配到各个流
        # branch_output: (b, T, d), beta: (s,)
        branch_to_streams = einsum(branch_output, beta, "b T d, s -> b T s d")
        
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

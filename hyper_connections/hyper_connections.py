from __future__ import annotations
from typing import Callable

from functools import partial
from random import randrange
import math

import torch
from torch import nn, cat, Tensor
import torch.nn.functional as F
from torch.nn import Module, Sequential
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.utils.checkpoint import checkpoint

from einops import rearrange, repeat, reduce, einsum
from einops.layers.torch import Rearrange, Reduce

from nanochat.common import print0

"""
ein notation:
b - batch
d - feature dimension
s - residual streams
t - residual streams + num branch inputs
v - number of views for branch input
"""

# helper functions


def exists(v):
    return v is not None


def divisible_by(num, den):
    return (num % den) == 0


def default(v, d):
    return v if exists(v) else d


def identity(t):
    return t


def add(x, y):
    return x + y


def sinkhorn_log(logits, num_iters=10, tau=0.05):
    """
    Unified Sinkhorn-Knopp algorithm supporting both 2D and batched inputs.

    Args:
        logits: (n, n) or (..., n, n) - unnormalized log-probabilities
        num_iters: Number of Sinkhorn iterations
        tau: Temperature parameter

    Returns:
        (..., n, n) doubly-stochastic matrix
    """
    n = logits.shape[-1]
    Z = logits / tau
    log_marginal = torch.zeros(
        (n,), device=logits.device, dtype=logits.dtype
    )

    # Initialize u, v with proper batch shape
    batch_shape = logits.shape[:-2]
    u = torch.zeros((*batch_shape, n), device=Z.device, dtype=Z.dtype)
    v = torch.zeros((*batch_shape, n), device=Z.device, dtype=Z.dtype)

    for _ in range(num_iters):
        u = log_marginal - torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
        v = log_marginal - torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)

    return torch.exp(Z + u.unsqueeze(-1) + v.unsqueeze(-2))


def zeropower_via_newtonschulz(X, steps=5, eps=1e-7, coeffs=(3.0, -3.2, 1.2)):
    a, b, c = coeffs

    # Support arbitrary batch dimensions: (..., n, n)
    original_shape = X.shape
    ndim = X.dim()

    if ndim == 2:
        # Single matrix (n, n)
        X = X / (X.norm() + eps)
        transpose = False
        if X.shape[0] > X.shape[1]:
            X = X.T
            transpose = True
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        if transpose:
            X = X.T
        return X

    elif ndim >= 3:
        # Batch of matrices (..., n, n)
        # Flatten all batch dimensions into one
        batch_shape = original_shape[:-2]
        n, m = original_shape[-2:]
        X = X.reshape(-1, n, m)  # (batch_prod, n, m)

        # Normalize
        X_norm = X.norm(dim=(1, 2), keepdim=True) + eps
        X = X / X_norm

        # Handle transpose if needed
        transpose = X.shape[1] > X.shape[2]
        if transpose:
            X = X.transpose(1, 2)

        # Newton-Schulz iterations
        for _ in range(steps):
            A = X @ X.transpose(1, 2)
            B = b * A + c * (A @ A)
            X = a * X + B @ X

        if transpose:
            X = X.transpose(1, 2)

        # Unflatten back to original batch shape
        X = X.reshape(*batch_shape, n, m)
        return X

    else:
        raise ValueError("Input must be at least 2D tensor")


def orthostochastic_project(
    logits, ns_steps=5, ns_eps=1e-7, ns_coeffs=(3.0, -3.2, 1.2)
):
    O = zeropower_via_newtonschulz(logits, steps=ns_steps, eps=ns_eps, coeffs=ns_coeffs)
    return O.square()


# main functions


def get_expand_reduce_stream_functions(
    num_streams, add_stream_embed=False, dim=None, disable=False
):
    if num_streams == 1 or disable:
        return (nn.Identity(), nn.Identity())

    if add_stream_embed:
        assert exists(dim), (
            "`dim` must be passed into get_init_and_expand_reduce_stream_functions for returning an expansion function with stream embeddings added"
        )

        expand_fn = StreamEmbed(num_streams, dim, expand_to_streams=True)
    else:
        expand_fn = Reduce(
            pattern="b ... -> (b s) ...", reduction="repeat", s=num_streams
        )

    reduce_fn = Reduce(pattern="(b s) ... -> b ...", reduction="sum", s=num_streams)

    return expand_fn, reduce_fn


def get_init_and_expand_reduce_stream_functions(
    num_streams,
    dim=None,
    add_stream_embed=False,
    disable=None,
    gradient_checkpointing=False,
    dynamic_H=False,
    # Geometric mode parameters
    hc_geometric=False,
    manifold_dim=4,
    sigma_init=1.0,
    sigma_learnable=True,
    H_mode="per-token",
    pool_type="last",
    # MHC parameters
    mhc=False,
    sinkhorn_iters=10,
    sinkhorn_tau=0.05,
    mhc_h_res_proj="sinkhorn",
    **kwargs  # Catch any other parameters
):
    """
    Unified factory function for creating HyperConnections initialization function.

    Supports three modes:
    - Standard: Original learnable alpha/beta (when mhc=False, hc_geometric=False)
    - MHC: manifold-constraint Hyper-Connections with static/dynamic/Geometry H (when mhc=True)
    """
    if num_streams == 1:
        print0("Warning: num_streams=1, HyperConnections will be disabled.")
        disable = True
    
    hyper_conn_klass = HyperConnections if not disable else Residual

    init_hyper_conn_fn = partial(
        hyper_conn_klass,
        num_streams,
        gradient_checkpointing=gradient_checkpointing,
        dynamic_H=dynamic_H,
        hc_geometric=hc_geometric,
        manifold_dim=manifold_dim,
        sigma_init=sigma_init,
        sigma_learnable=sigma_learnable,
        H_mode=H_mode,
        pool_type=pool_type,
        mhc=mhc,
        sinkhorn_iters=sinkhorn_iters,
        sinkhorn_tau=sinkhorn_tau,
        mhc_h_res_proj=mhc_h_res_proj,
        **kwargs
    )
    expand_reduce_fns = get_expand_reduce_stream_functions(
        num_streams, add_stream_embed=add_stream_embed, dim=dim, disable=disable
    )

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim=dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)


# norms


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * (self.gamma + 1)


# main classes

# residual base class


class Residual(Module):
    def __init__(
        self,
        *args,
        branch: Module | None = None,
        residual_transform: Module | None = None,
        **kwargs,
    ):
        super().__init__()
        self.branch = branch
        self.residual_transform = default(residual_transform, nn.Identity())

    def width_connection(self, residuals):
        return residuals, residuals, dict()

    def depth_connection(
        self,
        branch_output,
        residuals,
    ):
        return branch_output + self.residual_transform(residuals)

    def decorate_branch(self, branch: Callable):
        assert not exists(self.branch), "branch was already wrapped on init"

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)

            branch_output = branch(branch_input, *args, **kwargs)

            residual = add_residual(branch_output)

            return residual

        return forward_and_add_residual

    def forward(self, residuals, *branch_args, **branch_kwargs):
        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            (branch_out, *rest), tree_spec = tree_flatten(branch_out)

            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)

            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)

        return add_residual_fn(branch_output)


# hyper connection residual streams

"""
Hyper Connections:

general steps:

x: inputs (b, ..., d) -> expand to residual streams (b, ..., s, d)

Width connection set the branch input and mix residual streams:
residuals: (b, ..., s, d) -> branch_input: (b, ..., d)

Depth connection adds branch output back to residual streams:
branch_output: (b, ..., d), residuals: (b, ..., s, d) -> new_residuals: (b, ..., s, d)

Then reduce residual streams back to original shape:
new_residuals: (b, ..., s, d) -> (b, ..., d)

There are several modes:
- Standard mode (Static/Dynamic, whether dependent on input or not)
- MHC mode (manifold-constraint Hyper-Connections)
    - Static/Dynamic 
    - Geometric

The standard mode is the original Hyper-Connections paper:
    https://arxiv.org/pdf/2409.19606

They learn two vector and a matrix to control the width and depth connections:
    - Alpha: width connection weights from each residual stream and branch inputs to each residual stream
    - Beta: depth connection weights from branch output back to each residual stream
    - A: residual mixing matrix to mix residual streams before width connection
    
    In the dynamic setting:
    - beta(x) = s_beta \\circ tanh(norm(x) @ W_beta)^T + beta_static : (1, s)
    - alpha(x) = s_alpha \\circ tanh(norm(x) @ W_alpha) + alpha_static : (s, 1)
    - A(x)= s_alpha \\circ tanh(norm(x) @ W_A) + A_static : (s, s)

The MHC mode introduces manifold constraints on the residual mixing matrix A:
    - A is constrained to be a doubly-stochastic matrix on the probability simplex
    - Achieved via Sinkhorn-Knopp algorithm or orthostochastic projection
    
    The static is parameterized as logits, which is easy.
    In the dynamic setting:
        - First, the stream is flattened and RMSNormed as x (..., s, d) -> (..., s * d)
        - Then three linear projections are applied to generate the logits for H_res, H_pre, H_post
        - Finally, the logits are projected to the desired form:
            - H_res: doubly-stochastic matrix via Sinkhorn/orthostochastic
            - H_pre: sigmoid to (0, 1)
            - H_post: sigmoid * 2 to (0, 2)
    
    alpha <-> H_pre
    beta <-> H_post
    A <-> H_res

    Geometric mode:
    - Introduce a manifold projection head to project each residual stream to a low-dimensional manifold
    - Compute pairwise distances between residual streams in the manifold space
    - Use Gaussian kernel to compute similarity logits
    - Apply Sinkhorn-Knopp to get doubly-stochastic H_res
        Choices for granularity: 
            - per-token: each token has its own H_res
            - per-seq: each sequence has its own H_res (pool over "mean", "max", "last")
"""


class HyperConnections(Module):
    def __init__(
        self,
        num_residual_streams,
        *,
        dim,
        branch: Module | None = None,
        layer_index=None,
        tanh=True,
        channel_first=False,
        dropout=0.0,
        residual_transform: Module
        | None = None,  # to support resnet blocks where dimension in not equal to dimension out - usually a residual conv
        add_branch_out_to_residual=True,  # will disable depth connections (weighted residual sum with beta) if set False
        num_input_views=1,  # allow for the branch module to receive multiple input views, dimension placed on the very left (before batch)
        depth_residual_fn=add,
        mhc=False,
        dynamic_H=False,  # enable dynamic H generation for mhc mode
        sinkhorn_iters=10,
        sinkhorn_tau=0.05,
        mhc_h_res_proj="sinkhorn",
        ns_steps=5,
        ns_eps=1e-7,
        ns_coeffs=(3.0, -3.2, 1.2),
        gradient_checkpointing=False,
        # Geometric mode parameters
        hc_geometric=False,  # enable geometric-induced HC
        manifold_dim=4,  # manifold dimension for geometric projection
        sigma_init=1.0,  # initial RBF bandwidth
        sigma_learnable=True,  # whether sigma is learnable
        H_mode="per-token",  # geometric H granularity: per-token/per-seq
        pool_type="last",  # pooling type: mean/max/last
    ):
        """
        Appendix J, Algorithm2 in - https://arxiv.org/abs/2409.19606
        """
        super().__init__()

        self.branch = branch

        self.act = nn.Tanh() if tanh else nn.Identity()

        # RMSNorm for standard mode
        if not mhc and not hc_geometric:
            self.norm = RMSNorm(dim)

        assert num_residual_streams > 0, "`num_residual_streams` must be greater than 0"

        self.num_residual_streams = num_residual_streams
        init_residual_index = (
            default(layer_index, randrange(num_residual_streams)) % num_residual_streams
        )  # just choose one random residual stream if layer index not given

        # width num residual streams

        assert num_input_views >= 1
        self.num_input_views = num_input_views

        # width connection

        if not mhc and not hc_geometric:
            init_alpha0 = torch.zeros((num_residual_streams, num_input_views))
            init_alpha0[init_residual_index, :] = 1.0

            self.static_alpha = nn.Parameter(
                cat((init_alpha0, torch.eye(num_residual_streams)), dim=1)
            )

            self.dynamic_alpha_fn = nn.Parameter(
                torch.zeros(dim, num_residual_streams + num_input_views)
            )
            self.dynamic_alpha_scale = nn.Parameter(torch.ones(()) * 1e-2)

        # depth connection related (beta)

        self.add_branch_out_to_residual = add_branch_out_to_residual

        if add_branch_out_to_residual and not mhc and not hc_geometric:
            self.static_beta = nn.Parameter(torch.ones(num_residual_streams))
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(dim))
            self.dynamic_beta_scale = nn.Parameter(torch.ones(()) * 1e-2)

        self.dropout = nn.Dropout(dropout)
        self.channel_first = channel_first
        self.residual_transform = default(residual_transform, nn.Identity())

        # maybe custom depth connection residual function
        # this is to prepare for gating the addition of the branch outputs to the residual streams
        # needed for memory lanes a la RMT / LMM
        self.depth_residual_fn = depth_residual_fn

        self.mhc = mhc
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_tau = sinkhorn_tau
        self.mhc_h_res_proj = mhc_h_res_proj
        self.ns_steps = ns_steps
        self.ns_eps = ns_eps
        self.ns_coeffs = ns_coeffs
        self.gradient_checkpointing = gradient_checkpointing
        
        # Geometric 
        self.hc_geometric = hc_geometric

        if mhc:
            assert num_input_views == 1, "mhc currently requires num_input_views = 1"
            assert mhc_h_res_proj in (
                "sinkhorn",
                "orthostochastic",
            ), "mhc_h_res_proj must be 'sinkhorn' or 'orthostochastic'"

            if not hc_geometric:
                # 静态 H 参数（作为 beta，即 bias）
                H_res_init = torch.full((num_residual_streams, num_residual_streams), -8.0)
                H_res_init.fill_diagonal_(0.0)
                self.H_res_beta = nn.Parameter(H_res_init)

            H_pre_init = torch.full((num_residual_streams,), -8.0)
            H_pre_init[init_residual_index] = 0.0
            self.H_pre_beta = nn.Parameter(H_pre_init)

            if add_branch_out_to_residual:
                self.H_post_beta = nn.Parameter(torch.zeros(num_residual_streams))

            # 动态 H 生成（可选）
            self.dynamic_H = dynamic_H
            if self.dynamic_H:
                # RMSNorm for flattened input
                self.H_norm = RMSNorm(num_residual_streams * dim)

                # 三个线性投影层
                self.H_res_proj = nn.Linear(
                    num_residual_streams * dim,
                    num_residual_streams * num_residual_streams,
                    bias=False
                )
                self.H_pre_proj = nn.Linear(
                    num_residual_streams * dim,
                    num_residual_streams,
                    bias=False
                )
                if add_branch_out_to_residual:
                    self.H_post_proj = nn.Linear(
                        num_residual_streams * dim,
                        num_residual_streams,
                        bias=False
                    )

                # Alpha (scalar weights) - 控制动态部分的强度
                self.H_res_alpha = nn.Parameter(torch.ones(()) * 1e-2)
                self.H_pre_alpha = nn.Parameter(torch.ones(()) * 1e-2)
                if add_branch_out_to_residual:
                    self.H_post_alpha = nn.Parameter(torch.ones(()) * 1e-2)

                # 初始化投影层为小值
                nn.init.normal_(self.H_res_proj.weight, std=0.02)
                nn.init.normal_(self.H_pre_proj.weight, std=0.02)
                if add_branch_out_to_residual:
                    nn.init.normal_(self.H_post_proj.weight, std=0.02)
                    
            elif self.hc_geometric:

                # Geometric parameters
                self.manifold_dim = manifold_dim
                self.H_mode = H_mode
                self.pool_type = pool_type

                # RMSNorm and manifold projection
                self.norm = RMSNorm(dim)
                self.proj_head = nn.Linear(dim, manifold_dim, bias=False)
                nn.init.normal_(self.proj_head.weight, std=0.02)

                if sigma_learnable:
                    self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma_init)))
                else:
                    self.register_buffer('log_sigma', torch.tensor(math.log(sigma_init)))

    @property
    def sigma(self) -> Tensor:
        """Get current sigma value (for geometric mode)"""
        return self.log_sigma.exp()

    def _pool(self, x, dim):
        """Pooling helper for geometric mode"""
        if self.pool_type == 'mean':
            return x.mean(dim=dim, keepdim=True)
        elif self.pool_type == 'max':
            return x.max(dim=dim, keepdim=True).values
        elif self.pool_type == 'last':
            return x.narrow(dim, x.shape[dim] - 1, 1)
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

    def _compute_geometric_H(self, residuals):
        """
        Compute geometry-induced H matrix.
        Args: residuals after rearrange: (b, T, s, d)
        Returns: H_res, H_pre, H_post
        """
        if residuals.ndim == 3:
            # (b, s, d) - add time dimension
            residuals = residuals.unsqueeze(1)  # (b, 1, s, d)

        b, T, s, d = residuals.shape

        if self.H_mode == 'per-token':
            normed = self.norm(residuals)
            coords = self.proj_head(normed)  # (b, T, s, k)
            dist_sq = torch.cdist(coords, coords, p=2) ** 2 # (b, T, s, s)
            H_res_logits = -dist_sq / (2 * self.sigma ** 2)

        elif self.H_mode == 'per-seq':
            pooled = self._pool(residuals, dim=1)  # (b, 1, s, d)
            normed = self.norm(pooled)
            coords_seq = self.proj_head(normed).squeeze(1)  # (b, s, k)
            dist_sq = torch.cdist(coords_seq, coords_seq, p=2) ** 2
            H_res_logits = -dist_sq / (2 * self.sigma ** 2)
        else:
            raise ValueError(f"Unknown H_mode: {self.H_mode}")

        # Apply projection to get doubly-stochastic H_res
        # Support both sinkhorn and orthostochastic projections
        if self.mhc_h_res_proj == "sinkhorn":
            H_res = sinkhorn_log(H_res_logits, self.sinkhorn_iters, self.sinkhorn_tau)
        elif self.mhc_h_res_proj == "orthostochastic":
            H_res = orthostochastic_project(
                H_res_logits,
                ns_steps=self.ns_steps,
                ns_eps=self.ns_eps,
                ns_coeffs=self.ns_coeffs,
            )
        else:
            raise ValueError(f"Unknown mhc_h_res_proj: {self.mhc_h_res_proj}")

        H_pre = F.softmax(self.H_pre_beta, dim=-1)
        H_post = F.softmax(self.H_post_beta, dim=-1) if self.add_branch_out_to_residual else None

        return H_res, H_pre, H_post

    def _compute_dynamic_H(self, residuals):
        """
        Per-token dynamic H generation
        Args: residuals (b, ..., s, d)
        Returns: H_res (b, ..., s, s), H_pre (b, ..., s), H_post (b, ..., s) or None
        """
        s = self.num_residual_streams

        # 1. Flatten: (b, ..., s, d) -> (b, ..., s*d)
        residuals_flat = rearrange(residuals, "... s d -> ... (s d)")

        # 2. RMSNorm
        normed = self.H_norm(residuals_flat)

        # 3. Linear projections
        H_res_proj = self.H_res_proj(normed)  # (b, ..., s*s)
        H_pre_proj = self.H_pre_proj(normed)  # (b, ..., s)
        H_post_proj = self.H_post_proj(normed) if self.add_branch_out_to_residual else None

        # 4. Modulate: alpha * dynamic + beta
        H_res_logits = (
            H_res_proj.view(*H_res_proj.shape[:-1], s, s) * self.H_res_alpha
            + self.H_res_beta
        )
        H_pre_logits = H_pre_proj * self.H_pre_alpha + self.H_pre_beta
        if H_post_proj is not None:
            H_post_logits = H_post_proj * self.H_post_alpha + self.H_post_beta
        else:
            H_post_logits = None

        # 5. Activations
        # H_res: Sinkhorn normalization (works directly on logits)
        if self.mhc_h_res_proj == "sinkhorn":
            H_res = sinkhorn_log(H_res_logits, self.sinkhorn_iters, self.sinkhorn_tau)
        elif self.mhc_h_res_proj == "orthostochastic":
            H_res = orthostochastic_project(
                H_res_logits,
                ns_steps=self.ns_steps,
                ns_eps=self.ns_eps,
                ns_coeffs=self.ns_coeffs,
            )
        else:
            raise ValueError(f"Unknown mhc_h_res_proj: {self.mhc_h_res_proj}")

        # H_pre: sigmoid
        H_pre = torch.sigmoid(H_pre_logits)

        # H_post: sigmoid * 2
        H_post = torch.sigmoid(H_post_logits) * 2 if H_post_logits is not None else None

        return H_res, H_pre, H_post

    def width_connection(self, residuals):
        streams = self.num_residual_streams

        maybe_transformed_residuals = self.residual_transform(residuals)
        # (b, ..., d) or (b, d, ...)

        # width connection

        # handle channel first

        if self.channel_first:
            residuals = rearrange(residuals, "b d ... -> b ... d")

        # split out streams

        residuals = rearrange(residuals, "(b s) ... d -> b ... s d", s=streams)

        if self.hc_geometric or self.mhc:
            residuals_mixed_source = maybe_transformed_residuals

            if self.channel_first:
                residuals_mixed_source = rearrange(
                    residuals_mixed_source, "b d ... -> b ... d"
                )
            residuals_mixed_source = rearrange(
                residuals_mixed_source, "(b s) ... d -> b ... s d", s=streams
            )
            if self.hc_geometric:
                # Compute geometric H matrices
                if self.gradient_checkpointing and self.training:
                    H_res, H_pre, H_post = checkpoint(
                        self._compute_geometric_H, residuals, use_reentrant=False
                    )
                else:
                    H_res, H_pre, H_post = self._compute_geometric_H(residuals)
            else:
                if self.dynamic_H:
                    # Per-token dynamic H
                    H_res, H_pre, H_post = self._compute_dynamic_H(residuals)
                else:
                    # 静态 H (原始实现)
                    if self.mhc_h_res_proj == "orthostochastic":
                        H_res = orthostochastic_project(
                            self.H_res_beta,
                            ns_steps=self.ns_steps,
                            ns_eps=self.ns_eps,
                            ns_coeffs=self.ns_coeffs,
                        )
                    else:
                        H_res = sinkhorn_log(
                            self.H_res_beta, self.sinkhorn_iters, self.sinkhorn_tau
                        )
                        
                    H_pre = F.softmax(self.H_pre_beta, dim=-1)
                    H_post = F.softmax(self.H_post_beta, dim=-1) if self.add_branch_out_to_residual else None

            if H_res.ndim == 2:
                # Static H_res: (s, s)
                residuals_mixed = einsum(
                    H_res, residuals_mixed_source, "s t, b ... s d -> b ... t d"
                )
                branch_input = einsum(
                    H_pre, residuals, "s, b ... s d -> b ... d"
                )
            elif H_res.ndim == 3: # per-seq and geometric
                # per-seq
                residuals_mixed = einsum(
                    H_res, residuals_mixed_source, "b s t, b ... s d -> b ... t d"
                )
                branch_input = einsum(
                    H_pre, residuals, "s, b ... s d -> b ... d"
                )
            else:
                # per-token
                residuals_mixed = einsum(
                    H_res, residuals_mixed_source, "b ... s t, b ... s d -> b ... t d"
                )
                if self.hc_geometric: # geometric H_pre is static
                    branch_input = einsum(
                        H_pre, residuals, "s, b ... s d -> b ... d"
                    )
                else:
                    branch_input = einsum(
                        H_pre, residuals, "b ... s, b ... s d -> b ... d"
                    )
                

            if self.channel_first:
                branch_input = rearrange(branch_input, "b ... d -> b d ...")

            return (
                branch_input, 
                maybe_transformed_residuals, 
                dict(beta=H_post, residuals_mixed=residuals_mixed)
            )

        # --- standard mode ---
        
        # norm

        normed = self.norm(residuals)

        # alpha for weighted sum of residuals going into branch

        dynamic_alpha = self.act(normed @ self.dynamic_alpha_fn) * self.dynamic_alpha_scale
        alpha = dynamic_alpha + self.static_alpha

        # beta for weights from branch output back to residual streams
        beta = None
        if self.add_branch_out_to_residual:
            dynamic_beta = self.act(normed @ self.dynamic_beta_fn) * self.dynamic_beta_scale
            beta = dynamic_beta + self.static_beta
        
        mix_h = einsum(alpha, residuals, "... s t, ... s d -> ... t d")

        if self.num_input_views == 1:
            branch_input, residuals_next = mix_h[..., 0, :], mix_h[..., 1:, :]
        else:
            branch_input, residuals_next = (
                mix_h[..., : self.num_input_views, :],
                mix_h[..., self.num_input_views :, :],
            )
            branch_input = rearrange(branch_input, "b ... v d -> v b ... d")
            # Residuals next shape: (b, T, s, d)

        if self.channel_first:
            branch_input = rearrange(branch_input, "b ... d -> b d ...")

        return branch_input, maybe_transformed_residuals, dict(beta=beta, residuals_mixed=residuals_next)

    def depth_connection(self, branch_output, residuals, *, beta, residuals_mixed=None):
        assert self.add_branch_out_to_residual

        # 'depth' connection

        if self.channel_first:
            branch_output = rearrange(branch_output, "b d ... -> b ... d")
            
        # --- Standard mode ---
        # beta shape: (s,) or dynamically computed
        if beta.ndim == 1:
            # Static beta: (s,)
            branch_to_streams = einsum(branch_output, beta, "b ... d, s -> b ... s d")
        else:
            # Dynamic beta: (b, ..., s)
            branch_to_streams = einsum(branch_output, beta, "b ... d, b ... s -> b ... s d")

        output = self.depth_residual_fn(branch_to_streams, residuals_mixed)
        
        output = rearrange(output, "b ... s d -> (b s) ... d")

        if self.channel_first:
            output = rearrange(output, "b ... d -> b d ...")

        return self.dropout(output)

    def decorate_branch(self, branch: Callable):
        assert not exists(self.branch), "branch was already wrapped on init"

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)

            branch_output = branch(branch_input, *args, **kwargs)

            residual = add_residual(branch_output)

            return residual

        return forward_and_add_residual

    def forward(self, residuals, *branch_args, **branch_kwargs):
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


HyperConnections.get_expand_reduce_stream_functions = staticmethod(
    get_expand_reduce_stream_functions
)
HyperConnections.get_init_and_expand_reduce_stream_functions = staticmethod(
    get_init_and_expand_reduce_stream_functions
)

# stream embed


class StreamEmbed(Module):
    def __init__(self, num_streams, dim, channel_first=False, expand_to_streams=False):
        super().__init__()
        self.channel_first = channel_first
        self.num_streams = num_streams

        self.expand_to_streams = expand_to_streams
        self.stream_embed = nn.Parameter(torch.zeros(num_streams, dim))

    def forward(self, residuals):
        if self.expand_to_streams:
            residuals = repeat(residuals, "b ... -> (b s) ...", s=self.num_streams)

        if self.channel_first:
            residuals = rearrange(
                residuals, "(b s) d ... -> b ... s d", s=self.num_streams
            )
        else:
            residuals = rearrange(
                residuals, "(b s) ... d -> b ... s d", s=self.num_streams
            )

        residuals = residuals + self.stream_embed

        if self.channel_first:
            residuals = rearrange(
                residuals, "b ... s d -> (b s) d ...", s=self.num_streams
            )
        else:
            residuals = rearrange(
                residuals, "b ... s d -> (b s) ... d", s=self.num_streams
            )

        return residuals


# attention pool - taken from Enformer https://www.nature.com/articles/s41592-021-01252-x , in turn taken from somewhere else


class AttentionPoolReduceStream(Module):
    def __init__(self, num_streams, dim, channel_first=False):
        super().__init__()
        self.num_streams = num_streams
        self.channel_first = channel_first

        self.to_attn_logits = nn.Linear(dim, dim, bias=False)
        self.to_attn_logits.weight.data.copy_(torch.eye(dim))

    def forward(self, residuals):
        if self.channel_first:
            residuals = rearrange(
                residuals, "(b s) d ... -> b ... s d", s=self.num_streams
            )
        else:
            residuals = rearrange(
                residuals, "(b s) ... d -> b ... s d", s=self.num_streams
            )

        attn_logits = self.to_attn_logits(residuals)
        attn = attn_logits.softmax(dim=-2)

        residuals = reduce(residuals * attn, "b ... s d -> b ... d", "sum")

        if self.channel_first:
            residuals = rearrange(residuals, "b ... d -> b d ...")

        return residuals

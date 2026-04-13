from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.flash_attention import flash_attn


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    """Match nanochat's explicit compute-dtype casting behavior."""

    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


class InterleavedSelfAttention(nn.Module):
    """
    Training-oriented Interleaved Head Attention for nanochat.

    This path currently supports full-sequence forward passes used in pretraining
    and loss evaluation. KV-cache decode is left unsupported so inference does
    not silently produce incorrect outputs.
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        if config.n_kv_head != config.n_head:
            raise ValueError("IHA currently requires n_kv_head == n_head in nanochat")
        if config.iha_num_pseudo_heads <= 0:
            raise ValueError("iha_num_pseudo_heads must be positive")
        if config.iha_collapse_mode not in {"per_head", "global"}:
            raise ValueError("iha_collapse_mode must be 'per_head' or 'global'")
        if config.iha_mask_mode not in {"flat_causal", "token_causal", "none"}:
            raise ValueError("iha_mask_mode must be 'flat_causal', 'token_causal', or 'none'")

        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.num_pseudo_heads = config.iha_num_pseudo_heads
        self.collapse_mode = config.iha_collapse_mode
        self.mask_mode = config.iha_mask_mode

        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)

        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

        mix_shape = (self.n_head, self.n_head, self.num_pseudo_heads)
        self.alpha_q = nn.Parameter(torch.empty(mix_shape))
        self.alpha_k = nn.Parameter(torch.empty(mix_shape))
        self.alpha_v = nn.Parameter(torch.empty(mix_shape))

        if self.collapse_mode == "per_head":
            collapse_shape = (self.n_head, self.num_pseudo_heads)
        else:
            collapse_shape = (self.n_head, self.n_head * self.num_pseudo_heads)
        self.collapse = nn.Parameter(torch.empty(collapse_shape))

    @torch.no_grad()
    def reset_iha_parameters(self):
        identity = torch.eye(self.n_head, dtype=self.alpha_q.dtype, device=self.alpha_q.device)
        identity = identity.unsqueeze(-1).expand(-1, -1, self.num_pseudo_heads)
        self.alpha_q.copy_(identity)
        self.alpha_k.copy_(identity)
        self.alpha_v.copy_(identity)

        if self.collapse_mode == "per_head":
            self.collapse.fill_(1.0 / self.num_pseudo_heads)
        else:
            self.collapse.zero_()
            for head_idx in range(self.n_head):
                self.collapse[head_idx, head_idx * self.num_pseudo_heads] = 1.0

    def _mix_heads(self, states, alpha):
        return torch.einsum("mhp,btmd->bthpd", alpha.to(dtype=states.dtype), states)

    def _merge_pseudo(self, states):
        bsz, seq_len, num_heads, num_pseudo, head_dim = states.shape
        return (
            states.permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(bsz, seq_len * num_pseudo, num_heads, head_dim)
        )

    def _reshape_output(self, states, seq_len):
        bsz, total_seq, num_heads, head_dim = states.shape
        assert total_seq == seq_len * self.num_pseudo_heads
        return states.view(bsz, seq_len, self.num_pseudo_heads, num_heads, head_dim).permute(0, 1, 3, 2, 4)

    def _collapse_pseudo(self, states):
        if self.collapse_mode == "per_head":
            return torch.einsum("hp,bthpd->bthd", self.collapse.to(dtype=states.dtype), states)

        bsz, seq_len, _, _, head_dim = states.shape
        flat_states = states.contiguous().view(
            bsz,
            seq_len,
            self.n_head * self.num_pseudo_heads,
            head_dim,
        )
        return torch.einsum("ho,btod->bthd", self.collapse.to(dtype=states.dtype), flat_states)

    def _build_attention_mask(self, seq_len, window_tokens, device):
        total_seq = seq_len * self.num_pseudo_heads
        if self.mask_mode == "none" and window_tokens is None:
            return torch.ones(total_seq, total_seq, dtype=torch.bool, device=device)

        virtual_positions = torch.arange(total_seq, device=device)
        token_positions = virtual_positions // self.num_pseudo_heads

        if self.mask_mode == "flat_causal":
            mask = virtual_positions[:, None] >= virtual_positions[None, :]
        elif self.mask_mode == "token_causal":
            mask = token_positions[:, None] >= token_positions[None, :]
        else:
            mask = torch.ones(total_seq, total_seq, dtype=torch.bool, device=device)

        if window_tokens is not None:
            token_delta = token_positions[:, None] - token_positions[None, :]
            mask = mask & (token_delta >= 0) & (token_delta < window_tokens)

        return mask

    def _run_attention(self, q, k, v, seq_len, window_size):
        window_tokens = window_size[0]
        if window_tokens < 0 or window_tokens >= seq_len:
            window_tokens = None

        if self.mask_mode == "flat_causal" and window_tokens is None:
            return flash_attn.flash_attn_func(
                q,
                k,
                v,
                causal=True,
                window_size=(q.size(1), 0),
            )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        mask = self._build_attention_mask(seq_len, window_tokens, q.device)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return y.transpose(1, 2)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        if kv_cache is not None:
            raise NotImplementedError(
                "nanochat IHA does not yet support KV-cache decode; use full-sequence forwards"
            )

        bsz, seq_len, _ = x.size()

        q = self.c_q(x).view(bsz, seq_len, self.n_head, self.head_dim)
        k = self.c_k(x).view(bsz, seq_len, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(bsz, seq_len, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(bsz, seq_len, self.n_kv_head, self.head_dim)
            assert self.ve_gate is not None
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        q = self._merge_pseudo(self._mix_heads(q, self.alpha_q))
        k = self._merge_pseudo(self._mix_heads(k, self.alpha_k))
        v = self._merge_pseudo(self._mix_heads(v, self.alpha_v))

        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q = q * 1.2
        k = k * 1.2

        y = self._run_attention(q, k, v, seq_len, window_size)
        y = self._reshape_output(y, seq_len)
        y = self._collapse_pseudo(y).contiguous().view(bsz, seq_len, self.n_embd)
        return self.c_proj(y)

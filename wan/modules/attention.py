import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    if q.device.type == 'xpu' or not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        return _attention_sdpa_varlen(
            q=q, k=k, v=v, q_lens=q_lens, k_lens=k_lens,
            dropout_p=dropout_p, softmax_scale=softmax_scale,
            q_scale=q_scale, causal=causal, window_size=window_size,
            dtype=dtype,
        )
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic).unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def _attention_sdpa_varlen(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
                           softmax_scale=None, q_scale=None, causal=False,
                           window_size=(-1, -1), dtype=torch.bfloat16):
    """SDPA fallback for the flash_attention varlen interface (XPU / no-flash builds).
    q, k, v: [B, L, N, C]; q_lens/k_lens: optional per-batch valid lengths."""
    import torch.nn.functional as F
    if q_scale is not None:
        q = q * q_scale
    out_dtype = q.dtype
    half_dtypes = (torch.float16, torch.bfloat16)
    if q.dtype not in half_dtypes:
        q = q.to(dtype)
    if k.dtype not in half_dtypes:
        k = k.to(dtype)
    if v.dtype not in half_dtypes:
        v = v.to(dtype)
    q = q.to(v.dtype)
    k = k.to(v.dtype)
    if q_lens is not None:
        assert q.size(0) == 1, "SDPA varlen fallback supports q_lens only for batch=1"
        q = q[:, :q_lens[0]]
    lq = q.size(1)
    lk = k.size(1)
    left, right = window_size
    attn_mask = None
    if k_lens is not None or left >= 0 or right >= 0 or causal:
        key_positions = torch.arange(lk, device=q.device)
        query_positions = torch.arange(lq, device=q.device) + max(lk - lq, 0)
        attn_mask = torch.ones((lq, lk), device=q.device, dtype=torch.bool)
        if k_lens is not None:
            valid_keys = key_positions[None, :] < k_lens.to(q.device)[:, None]
            attn_mask = attn_mask.unsqueeze(0) & valid_keys[:, None, :]
        else:
            attn_mask = attn_mask.unsqueeze(0)
        if causal:
            attn_mask = attn_mask & (key_positions[None, None, :] <= query_positions[None, :, None])
        if left >= 0:
            attn_mask = attn_mask & (key_positions[None, None, :] >= query_positions[None, :, None] - left)
        if right >= 0:
            attn_mask = attn_mask & (key_positions[None, None, :] <= query_positions[None, :, None] + right)
        attn_mask = attn_mask[:, None, :, :]
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=attn_mask, is_causal=False, dropout_p=dropout_p,
        scale=softmax_scale)
    return out.transpose(1, 2).contiguous().to(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        q = q.to(dtype)
        k = k.to(dtype)
        v = v.to(dtype)
        return _attention_sdpa_varlen(
            q=q, k=k, v=v, q_lens=q_lens, k_lens=k_lens,
            dropout_p=dropout_p, softmax_scale=softmax_scale,
            q_scale=q_scale, causal=causal, window_size=window_size,
            dtype=dtype)

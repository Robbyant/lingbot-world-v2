import torch


def get_accelerator(allow_cpu=False):
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch, 'xpu', None) is not None and torch.xpu.is_available():
        return 'xpu'
    if allow_cpu:
        return 'cpu'
    raise RuntimeError('Neither CUDA nor XPU is available.')


def _device_type(device=None):
    if device is None:
        return get_accelerator()
    return torch.device(device).type


def empty_cache(device=None):
    device_type = _device_type(device)
    if device_type == 'cuda':
        torch.cuda.empty_cache()
    elif device_type == 'xpu':
        torch.xpu.empty_cache()


def synchronize(device=None):
    device_type = _device_type(device)
    if device_type == 'cuda':
        torch.cuda.synchronize()
    elif device_type == 'xpu':
        torch.xpu.synchronize()


def ac(*args, **kwargs):
    """torch.amp.autocast with 'cuda' rewritten to 'xpu' on XPU-only hosts.

    This repo targets CUDA. On the XPU build autocast('cuda') is a silent
    no-op, which breaks fp32/bf16 mixed-dtype blocks (VAE emits fp32 latents,
    the DiT is bf16). Rewriting keeps CUDA behavior identical.
    """
    if (args and args[0] == 'cuda'
            and getattr(torch, 'xpu', None) is not None
            and torch.xpu.is_available()
            and not torch.cuda.is_available()):
        dtype = kwargs.get('dtype')
        if dtype is None and len(args) > 1:
            dtype = args[1]
        if dtype is torch.float32:
            # fp32 autocast is unsupported on XPU; the mixed-dtype blocks
            # are handled by keeping their submodules in fp32 (see
            # image2video.py). Disable autocast for the block so the
            # fp32 submodules are not upcast by an outer bf16 autocast.
            return torch.amp.autocast('xpu', enabled=False)
        args = ('xpu',) + tuple(args[1:])
    return torch.amp.autocast(*args, **kwargs)

"""YOLO11 network in MLX, built from an Ultralytics PyTorch model and its weights (PRD 5.2 MLX
evaluation, PROMPTS step 22).

Only for the micro-benchmark: forward + backward of the network with a fixed surrogate loss, to
compare MLX with PyTorch MPS on a Mac. It is NOT a training engine: Ultralytics' detection loss
(task-aligned assignment, CIoU, DFL) is not ported — that is the large job the benchmark is meant
to justify or not (MLX must be >= 1.5x faster than MPS).

Layout: MLX convolutions are NHWC, PyTorch NCHW. Weights are transposed on import; everything
that depends on channel order (splits, concat, attention reshapes) is written so the numbers
match PyTorch exactly (tests/test_bench_apple.py checks outputs and gradients on the CPU).

Supported blocks = everything in yolo11{n,s,m}.yaml detect models: Conv, DWConv, Bottleneck,
C3k2 (C2f + C3k), SPPF, C2PSA (PSABlock, Attention), Upsample, Concat, Detect.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _np(t) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float32)


class Conv(nn.Module):
    """Conv2d (no bias) + BatchNorm + SiLU, or a plain Conv2d with bias (Detect's last layers)."""

    def __init__(self, conv, bn=None, act: bool = True) -> None:
        super().__init__()
        self.weight = mx.array(_np(conv.weight).transpose(0, 2, 3, 1))  # (O, I/g, kh, kw) -> (O, kh, kw, I/g)
        if conv.bias is not None:
            self.bias = mx.array(_np(conv.bias))
        self.stride, self.padding = conv.stride[0], conv.padding[0]
        self.dilation, self.groups = conv.dilation[0], conv.groups
        self.has_bn, self.act = bn is not None, act
        if bn is not None:
            self.gamma, self.beta = mx.array(_np(bn.weight)), mx.array(_np(bn.bias))
            self._mean, self._var = mx.array(_np(bn.running_mean)), mx.array(_np(bn.running_var))  # "_" = not trained
            self._eps = bn.eps

    @classmethod
    def from_ultralytics(cls, m) -> Conv:
        """Ultralytics Conv / DWConv (conv + bn + SiLU or Identity)."""
        return cls(m.conv, m.bn, act=type(m.act).__name__ == "SiLU")

    def __call__(self, x: mx.array, train: bool) -> mx.array:
        y = mx.conv2d(x, self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation,
                      groups=self.groups)
        if "bias" in self:
            y = y + self.bias
        if self.has_bn:
            if train:  # batch statistics, biased variance (as PyTorch normalises in training)
                mean = mx.mean(y, axis=(0, 1, 2))
                var = mx.var(y, axis=(0, 1, 2))
            else:
                mean, var = self._mean, self._var
            y = (y - mean) * mx.rsqrt(var + self._eps) * self.gamma + self.beta
        return nn.silu(y) if self.act else y


class Bottleneck(nn.Module):
    def __init__(self, m) -> None:
        super().__init__()
        self.cv1, self.cv2 = Conv.from_ultralytics(m.cv1), Conv.from_ultralytics(m.cv2)
        self.add = bool(m.add)

    def __call__(self, x, train):
        y = self.cv2(self.cv1(x, train), train)
        return x + y if self.add else y


class C3k(nn.Module):
    """C3: cv3(cat(m(cv1(x)), cv2(x)))."""

    def __init__(self, m) -> None:
        super().__init__()
        self.cv1, self.cv2, self.cv3 = (Conv.from_ultralytics(c) for c in (m.cv1, m.cv2, m.cv3))
        self.m = [Bottleneck(b) for b in m.m]

    def __call__(self, x, train):
        y = self.cv1(x, train)
        for b in self.m:
            y = b(y, train)
        return self.cv3(mx.concatenate([y, self.cv2(x, train)], axis=-1), train)


class Attention(nn.Module):
    def __init__(self, m) -> None:
        super().__init__()
        self.qkv, self.proj, self.pe = (Conv.from_ultralytics(c) for c in (m.qkv, m.proj, m.pe))
        self.num_heads, self.key_dim, self.head_dim, self.scale = m.num_heads, m.key_dim, m.head_dim, m.scale

    def __call__(self, x, train):
        B, H, W, C = x.shape
        N = H * W
        # PyTorch views the NCHW qkv tensor as (B, heads, 2*kd+hd, N): go channel-major first
        qkv = self.qkv(x, train).transpose(0, 3, 1, 2).reshape(B, self.num_heads, 2 * self.key_dim + self.head_dim, N)
        q = qkv[:, :, : self.key_dim]
        k = qkv[:, :, self.key_dim: 2 * self.key_dim]
        v = qkv[:, :, 2 * self.key_dim:]
        attn = mx.softmax((q * self.scale).transpose(0, 1, 3, 2) @ k, axis=-1)
        out = (v @ attn.transpose(0, 1, 3, 2)).reshape(B, C, H, W).transpose(0, 2, 3, 1)
        pe = self.pe(v.reshape(B, C, H, W).transpose(0, 2, 3, 1), train)
        return self.proj(out + pe, train)


class PSABlock(nn.Module):
    def __init__(self, m) -> None:
        super().__init__()
        self.attn = Attention(m.attn)
        self.ffn = [Conv.from_ultralytics(c) for c in m.ffn]
        self.add = bool(m.add)

    def __call__(self, x, train):
        x = x + self.attn(x, train) if self.add else self.attn(x, train)
        y = self.ffn[1](self.ffn[0](x, train), train)
        return x + y if self.add else y


def _inner(m):
    """Blocks inside C3k2.m: Bottleneck, C3k, or Sequential(Bottleneck, PSABlock)."""
    name = type(m).__name__
    if name == "Bottleneck":
        return Bottleneck(m)
    if name == "C3k":
        return C3k(m)
    if name == "Sequential":
        return _Seq([_inner(c) if type(c).__name__ != "PSABlock" else PSABlock(c) for c in m])
    raise NotImplementedError(f"MLX port: {name} in C3k2")


class _Seq(nn.Module):
    def __init__(self, layers) -> None:
        super().__init__()
        self.layers = layers

    def __call__(self, x, train):
        for layer in self.layers:
            x = layer(x, train)
        return x


class C2f(nn.Module):
    """C2f / C3k2: split cv1 into two halves, chain the inner blocks, concat, cv2."""

    def __init__(self, m) -> None:
        super().__init__()
        self.cv1, self.cv2 = Conv.from_ultralytics(m.cv1), Conv.from_ultralytics(m.cv2)
        self.c = m.c
        self.m = [_inner(b) for b in m.m]

    def __call__(self, x, train):
        y = self.cv1(x, train)
        ys = [y[..., : self.c], y[..., self.c:]]
        for b in self.m:
            ys.append(b(ys[-1], train))
        return self.cv2(mx.concatenate(ys, axis=-1), train)


class SPPF(nn.Module):
    def __init__(self, m) -> None:
        super().__init__()
        self.cv1, self.cv2 = Conv.from_ultralytics(m.cv1), Conv.from_ultralytics(m.cv2)
        self.k, self.n = m.m.kernel_size, getattr(m, "n", 3)
        self.add = bool(getattr(m, "add", False))

    def _pool(self, x):
        p = self.k // 2
        x = mx.pad(x, [(0, 0), (p, p), (p, p), (0, 0)], constant_values=-float("inf"))
        return nn.MaxPool2d(self.k, stride=1)(x)

    def __call__(self, x, train):
        ys = [self.cv1(x, train)]
        for _ in range(self.n):
            ys.append(self._pool(ys[-1]))
        y = self.cv2(mx.concatenate(ys, axis=-1), train)
        return y + x if self.add else y


class C2PSA(nn.Module):
    def __init__(self, m) -> None:
        super().__init__()
        self.cv1, self.cv2 = Conv.from_ultralytics(m.cv1), Conv.from_ultralytics(m.cv2)
        self.c = m.c
        self.m = [PSABlock(b) for b in m.m]

    def __call__(self, x, train):
        y = self.cv1(x, train)
        a, b = y[..., : self.c], y[..., self.c:]
        for blk in self.m:
            b = blk(b, train)
        return self.cv2(mx.concatenate([a, b], axis=-1), train)


class Upsample(nn.Module):
    def __init__(self, m) -> None:
        super().__init__()
        self.s = int(m.scale_factor)

    def __call__(self, x, train):
        return mx.repeat(mx.repeat(x, self.s, axis=1), self.s, axis=2)


class Concat(nn.Module):
    def __call__(self, xs, train):
        return mx.concatenate(xs, axis=-1)


def _branch(seq) -> list:
    """Detect branch: Conv | Sequential(DWConv, Conv) | Conv2d(bias) -> flat list of MLX Convs."""
    out = []
    for m in seq:
        name = type(m).__name__
        if name in ("Conv", "DWConv"):
            out.append(Conv.from_ultralytics(m))
        elif name == "Sequential":
            out.extend(_branch(m))
        elif name == "Conv2d":
            out.append(Conv(m, None, act=False))
        else:
            raise NotImplementedError(f"MLX port: {name} in Detect")
    return out


class Detect(nn.Module):
    """Training-mode head: (boxes (B, N, 4*reg_max), scores (B, N, nc)), anchors ordered like
    PyTorch's view(B, C, -1) per level (row-major H, W), levels concatenated."""

    def __init__(self, m) -> None:
        super().__init__()
        if getattr(m, "one2one_cv2", None) is not None:
            raise NotImplementedError("MLX port: end-to-end (one2one) heads")
        self.box = [_branch(s) for s in m.cv2]
        self.cls = [_branch(s) for s in m.cv3]

    def __call__(self, xs, train):
        boxes, scores = [], []
        for x, bb, cb in zip(xs, self.box, self.cls):
            b, c = x, x
            for layer in bb:
                b = layer(b, train)
            for layer in cb:
                c = layer(c, train)
            boxes.append(b.reshape(b.shape[0], -1, b.shape[-1]))
            scores.append(c.reshape(c.shape[0], -1, c.shape[-1]))
        return mx.concatenate(boxes, axis=1), mx.concatenate(scores, axis=1)


BLOCKS = {"Conv": Conv.from_ultralytics, "DWConv": Conv.from_ultralytics, "C3k2": C2f, "C2f": C2f,
          "C3k": C3k, "SPPF": SPPF, "C2PSA": C2PSA, "Upsample": Upsample, "Concat": lambda m: Concat(),
          "Detect": Detect}


class YOLO(nn.Module):
    """Whole network; layer routing (`f`) copied from the Ultralytics model."""

    def __init__(self, torch_model) -> None:
        super().__init__()
        layers, self._routes = [], []
        for m in torch_model.model:
            name = type(m).__name__
            if name not in BLOCKS:
                raise NotImplementedError(f"MLX port: {name}")
            layers.append(BLOCKS[name](m))
            self._routes.append(m.f)
        self.layers = layers

    def __call__(self, x: mx.array, train: bool = True):
        """x: NHWC float32 in 0..1. Returns (boxes, scores) of the Detect head."""
        outputs = []
        for layer, f in zip(self.layers, self._routes):
            if f != -1:
                x = outputs[f] if isinstance(f, int) else [x if j == -1 else outputs[j] for j in f]
            x = layer(x, train)
            outputs.append(x)
        return x


def surrogate_loss_mlx(model: YOLO, images: mx.array, box_target: mx.array, cls_target: mx.array) -> mx.array:
    """Stand-in for the detection loss: MSE on box logits + BCE on class logits (same in PyTorch)."""
    boxes, scores = model(images, train=True)
    bce = mx.mean(mx.logaddexp(0.0, scores) - scores * cls_target)
    return mx.mean((boxes - box_target) ** 2) + bce


def surrogate_loss_torch(model, images, box_target, cls_target):
    """PyTorch twin of surrogate_loss_mlx (model in train mode; targets laid out as (B, N, C))."""
    import torch.nn.functional as F

    out = model(images)
    boxes, scores = out["boxes"].transpose(1, 2), out["scores"].transpose(1, 2)
    return F.mse_loss(boxes, box_target) + F.binary_cross_entropy_with_logits(scores, cls_target)

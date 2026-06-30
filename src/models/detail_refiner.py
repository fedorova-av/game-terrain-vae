"""DetailRefiner — второй этап пайплайна генерации рельефа.

CVAE даёт корректную структуру (где горы/равнины, нужного типа и масштаба),
но сглаживает микрорельеф (свойство VAE: MSE минимизируется усреднением).
Refiner — НЕ VAE, а детерминированная residual-CNN, которая добавляет
высокочастотную деталь поверх сглаженного входа.

Ключевые отличия от VAE, из-за которых это работает (а вторая VAE — нет):
  - нет латентного бутылочного горлышка и KL -> ничего не усредняется в ноль;
  - лосс с упором на градиенты/шероховатость, а не чистый MSE -> поощряется
    высокочастотная текстура, а не гладкость;
  - residual-связь: модель учит только ДОБАВКУ детали, базовая структура
    проходит насквозь без искажения.

Вход/выход — тот же контракт: [B, 1, 256, 256], значения в [-1, 1].
Refiner обучается восстанавливать реальную плитку из её сглаженной версии,
поэтому на инференсе он «доводит» гладкий выход CVAE до реалистичной фактуры.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _gn(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)


class ResBlock(nn.Module):
    """Residual-блок без понижения разрешения (деталь добавляется в полном масштабе)."""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            _gn(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            _gn(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.body(x))


class DetailRefiner(nn.Module):
    """Residual-CNN, дорисовывает деталь поверх сглаженного heightmap.

    Условие опционально: можно подать класс рельефа, чтобы фактура зависела
    от типа (у гор резкие гребни, у равнин почти ничего). По умолчанию включено.

    forward(x_coarse, label=None) -> x_refined в [-1, 1]
    Возвращает x_coarse + delta, где delta — выученная высокочастотная добавка.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 48,
        num_blocks: int = 6,
        num_classes: int = 3,
        use_class_cond: bool = True,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.use_class_cond = bool(use_class_cond)
        self.residual_scale = float(residual_scale)
        cond_ch = 1 if self.use_class_cond else 0
        if self.use_class_cond:
            # класс рельефа -> 1 доп. канал (нормированный id), broadcast по H×W
            self.class_embed = nn.Embedding(num_classes, 1)

        self.head = nn.Conv2d(in_channels + cond_ch, base_channels, 3, 1, 1)
        self.blocks = nn.Sequential(*[ResBlock(base_channels) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(base_channels, in_channels, 3, 1, 1)
        nn.init.zeros_(self.tail.weight)   # старт с нулевой добавки -> сначала refiner = identity
        nn.init.zeros_(self.tail.bias)

    def forward(self, x_coarse: torch.Tensor, label: torch.Tensor | None = None) -> torch.Tensor:
        inp = x_coarse
        if self.use_class_cond:
            if label is None:
                label = torch.zeros(x_coarse.size(0), dtype=torch.long, device=x_coarse.device)
            emb = self.class_embed(label)[:, :, None, None].expand(-1, -1, x_coarse.size(2), x_coarse.size(3))
            inp = torch.cat([x_coarse, emb], dim=1)
        h = self.head(inp)
        h = self.blocks(h)
        delta = self.tail(h)
        return torch.clamp(x_coarse + self.residual_scale * delta, -1.0, 1.0)


# ----------------------------- лосс рефайнера -----------------------------
def _grad(x):
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return dx, dy


def _roughness(x):
    # Same local-variation idea as roughness_value in evaluation metrics.
    dx, dy = _grad(x)
    return 0.5 * (dx.abs().mean(dim=(1, 2, 3)) + dy.abs().mean(dim=(1, 2, 3)))


def refiner_loss(pred, target, w_pixel=1.0, w_grad=1.0, w_rough=0.5):
    """Лосс с упором на высокие частоты, а НЕ чистый MSE.

    pixel  — слабый якорь, чтобы не уезжала база (низкий вес);
    grad   — L1 по градиентам, главный драйвер фактуры;
    rough  — match шероховатости: |roughness(pred) - roughness(target)| -> 0,
             ровно та величина, что в общей метрике roughness_diff.
    """
    pixel = F.l1_loss(pred, target)
    dxp, dyp = _grad(pred)
    dxt, dyt = _grad(target)
    grad = 0.5 * (F.l1_loss(dxp, dxt) + F.l1_loss(dyp, dyt))
    rough = (_roughness(pred) - _roughness(target)).abs().mean()
    total = w_pixel * pixel + w_grad * grad + w_rough * rough
    return total, {"pixel": float(pixel.detach()), "grad": float(grad.detach()),
                   "rough": float(rough.detach()), "total": float(total.detach())}


def coarsen(x, kernel: int = 9, sigma: float = 2.5):
    """Сглаживание реальной плитки -> вход для обучения refiner (имитация выхода VAE).

    Гауссово размытие. Refiner учится восстанавливать x из coarsen(x), поэтому
    на инференсе доводит сглаженный выход CVAE до реалистичной фактуры.
    """
    device = x.device
    ax = torch.arange(kernel, device=device) - (kernel - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).to(x.dtype)
    kx = g.view(1, 1, 1, kernel)
    ky = g.view(1, 1, kernel, 1)
    pad = kernel // 2
    x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="reflect"), kx)
    x = F.conv2d(F.pad(x, (0, 0, pad, pad), mode="reflect"), ky)
    return x

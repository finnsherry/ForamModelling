"""Binary morphology on torch masks: dilation, connectivity and a truncated distance transform."""
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F

from .parameters import DTYPE, FLOOD_CHECK_EVERY


def _dilate_along(mask, dim):
    """OR each pixel with its two neighbours along dim."""
    out = mask.clone()
    n = mask.shape[dim]
    out.narrow(dim, 1, n - 1).logical_or_(mask.narrow(dim, 0, n - 1))
    out.narrow(dim, 0, n - 1).logical_or_(mask.narrow(dim, 1, n - 1))
    return out


def dilate(mask, kernel: Literal["cross", "square"]="cross"):
    """Binary dilation by boolean shifts; pixels outside the grid count as False.

    "cross" is the plus-shaped 3x3 kernel (4-connectivity). "square" is the full
    3x3 kernel (8-connectivity); it is separable, so it dilates along the rows and
    then along the columns of the result.
    """
    if kernel == "square":
        return _dilate_along(_dilate_along(mask, 0), 1)
    rows, columns = _dilate_along(mask, 0), _dilate_along(mask, 1)
    return rows | columns


def _flood(mask, region, kernel, stop_at=None):
    """Grow region by dilation inside mask until it stops growing, or reaches stop_at pixels."""
    count = int(region.sum())
    while True:
        for _ in range(FLOOD_CHECK_EVERY):
            region = dilate(region, kernel) & mask
        grown = int(region.sum())
        if grown == count or grown == stop_at:
            return region
        count = grown


def component_containing(mask, index, kernel: Literal["cross", "square"]="cross"):
    """The connected part of mask that contains pixel `index`, by flood fill.

    The number of dilations is about the region's diameter in pixels.
    """
    region = torch.zeros_like(mask)
    region[index] = mask[index]
    return _flood(mask, region, kernel)


def is_connected(mask, kernel: Literal["cross", "square"]="cross"):
    """True if mask is a single nonempty connected component."""
    total = int(mask.sum())
    if total == 0:
        return False
    region = torch.zeros(mask.shape, dtype=torch.bool, device=mask.device)
    region.view(-1)[int(mask.reshape(-1).to(torch.uint8).argmax())] = True
    # Stop as soon as the fill covers the whole mask: then it is connected.
    return int(_flood(mask, region, kernel, stop_at=total).sum()) == total


def distance_to(mask, h, max_distance):
    """Euclidean distance from every pixel to the nearest True pixel of mask, up to max_distance."""
    R = math.ceil(max_distance / h)
    d2 = torch.full(mask.shape, math.inf, dtype=DTYPE, device=mask.device).masked_fill(mask, 0.0)
    for dim in (0, 1):
        n = d2.shape[dim]
        padded = F.pad(d2, (0, 0, R, R) if dim == 0 else (R, R), value=math.inf)
        d2 = torch.stack([padded.narrow(dim, R + k, n) + k**2 for k in range(-R, R + 1)]).amin(0)
    distance = d2.sqrt() * h
    # Beyond R the square window may have missed a nearer pixel; values there are not exact.
    distance[distance > max_distance] = math.inf
    return distance

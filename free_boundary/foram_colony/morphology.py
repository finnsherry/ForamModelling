"""Binary morphology on torch masks: dilation, connectivity and a truncated distance transform."""
from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn.functional as F

CROSS = ((0, 1, 0), (1, 1, 1), (0, 1, 0))   # 4-connectivity
SQUARE = ((1, 1, 1), (1, 1, 1), (1, 1, 1))  # 8-connectivity


@lru_cache
def _weights(kernel, device):
    return torch.tensor(kernel, dtype=torch.float32, device=device)[None, None]


def dilate(mask, kernel=CROSS):
    """Binary dilation; pixels outside the grid count as False."""
    return F.conv2d(mask[None, None].float(), _weights(kernel, mask.device), padding=1)[0, 0] > 0


def component_containing(mask, index, kernel=CROSS):
    """The connected part of mask that contains pixel `index`, by flood fill.

    Grows the region one dilation at a time, staying inside mask, until it stops
    changing. The number of iterations is the region's diameter in pixels.
    """
    region = torch.zeros_like(mask)
    region[index] = mask[index]
    while True:
        grown = dilate(region, kernel) & mask
        if torch.equal(grown, region):
            return region
        region = grown


def is_connected(mask, kernel=CROSS):
    """True if mask is a single nonempty connected component."""
    pixels = torch.nonzero(mask)
    if len(pixels) == 0:
        return False
    return torch.equal(component_containing(mask, tuple(pixels[0].tolist()), kernel), mask)


def distance_to(mask, h, max_distance):
    """Euclidean distance from every pixel to the nearest True pixel of mask, up to max_distance.

    Exact wherever the distance is at most max_distance; inf beyond it. Squared
    Euclidean distance is separable in min-plus algebra, so it is computed as two
    1D passes, d2(i) = min_k [f(i+k) + k^2] over |k| <= R, first along columns
    and then along rows.
    """
    R = math.ceil(max_distance / h)
    d2 = torch.where(mask, 0.0, math.inf)
    for dim in (0, 1):
        n = d2.shape[dim]
        padded = F.pad(d2, (0, 0, R, R) if dim == 0 else (R, R), value=math.inf)
        d2 = torch.stack([padded.narrow(dim, R + k, n) + k**2 for k in range(-R, R + 1)]).amin(0)
    distance = d2.sqrt() * h
    # Beyond R the square window may have missed a nearer pixel; values there are not exact.
    distance[distance > max_distance] = math.inf
    return distance

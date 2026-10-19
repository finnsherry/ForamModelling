"""
SIM2EikonalSolverPyTorch
========================

Compute Riemannian distance maps on SIM(2), using PyTorch.
"""

import torch
import torch.nn.functional as F
from math import sqrt, log
from functools import lru_cache
from typing import Literal
from tqdm.auto import tqdm


# Distance map


def eikonal_solver(
    cost: torch.Tensor,
    source_points: tuple[tuple[int, int, int, int], ...],
    a_min: float,
    a_max: float,
    ws: tuple[float, float, float, float],
    dxy: float = 1,
    n_max: float = 1e3,
    tol: float = 1e-3,
    initial_condition: float = 1e8,
    device: str | torch.DeviceObjType = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    τ_min, τ_max = log(a_min), log(a_max)
    scales = torch.exp(torch.linspace(τ_min, τ_max, cost.shape[0], device=device))
    dscale_forward = scales[1] / scales[0] - 1.
    dscale_backward = 1. - scales[0] / scales[1]
    dscale_factor = min(dscale_forward, dscale_backward)
    dθ = 2 * torch.pi / (cost.shape[1] - 1)
    ε = 1 / sqrt(1 / (ws[0] * dxy)**2 + 1 / (ws[1] * dxy)**2 + 1 / (ws[2] * dθ)**2 + 1 / (ws[3] * dscale_factor)**2) * cost.min() / 4.
    cost = cost.to(device)
    u = torch.ones(cost.shape, device=device) * initial_condition
    boundary_condition = torch.ones_like(cost)
    for p in source_points:
        boundary_condition[*p] = 0.0

    u = u * boundary_condition

    error = torch.inf
    is_converged = False
    with tqdm(range(int(n_max))) as pbar:
        for n in pbar:
            u, du_dt = step_u(u, cost, dxy, dθ, scales, ws, ε, pad_value=10*initial_condition)
            u = u * boundary_condition
            error = check_convergence(du_dt, boundary_condition)
            if n % 50 == 0:
                pbar.set_postfix({"error": error.item()})
            if error < tol:
                is_converged = True
                print(f"Converged after {n} steps!")
                break
    if not is_converged:
        print(f"Hamiltonian did not converge to tolerance {tol}!")

    grad_u = distance_gradient(u, cost, dxy, dθ, scales, ws, pad_value=10*initial_condition)

    return u.cpu(), grad_u.cpu()


def step_u(
    u: torch.Tensor,
    cost: torch.Tensor,
    dxy: float,
    dθ: float,
    scales: torch.Tensor,
    ws: tuple[float, float, float, float],
    ε: float,
    pad_value: float = 1e3,
) -> tuple[torch.Tensor, torch.Tensor]:
    A1, A2, A3, A4 = upwinds(u, dxy,  dθ, scales, pad_value=pad_value)

    du_dt = 1 - ((A1 / ws[0])**2 + (A2 / ws[1])**2 + (A3 / ws[2])**2 + (A4 / ws[3])**2).sqrt() / cost
    return u + du_dt * ε, du_dt


def distance_gradient(
    u: torch.Tensor,
    cost: torch.Tensor,
    dxy: float,
    dθ: float,
    scales: torch.Tensor,
    ws: tuple[float, float, float, float],
    pad_value: float = 1e3,
) -> torch.Tensor:
    A1, A2, A3, A4 = upwinds(u, dxy,  dθ, scales, pad_value=pad_value)

    grad_u = (
        torch.stack(
            (
                A4/ws[3]**2,
                A3/ws[2]**2,
                A2/ws[1]**2,
                A1/ws[0]**2,
            ),
            dim=-1,
        )
        / cost[..., None] ** 2
    )
    return grad_u


def upwinds(
    u: torch.Tensor,
    dxy: float,
    dθ: float,
    scales: torch.Tensor,
    pad_value: float = 1e3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    Or = u.shape[1]

    (
        I_A1_forward,
        I_A1_backward,
        I_A2_forward,
        I_A2_backward,
    ) = get_A1_A2_kernels(Or, u.device)

    u_pad = F.pad(u, (1, 1, 1, 1), mode="constant", value=pad_value)

    A1_forward = (
        F.conv2d(
            u_pad,
            I_A1_forward,
            groups=Or,
        )
        / dxy
    ) * scales[:, None, None, None]
    A1_backward = (
        F.conv2d(
            u_pad,
            -I_A1_backward,
            groups=Or,
        )
        / dxy
    ) * scales[:, None, None, None]

    A2_forward = (
        F.conv2d(
            u_pad,
            I_A2_forward,
            groups=Or,
        )
        / dxy
    ) * scales[:, None, None, None]
    A2_backward = (
        F.conv2d(
            u_pad,
            -I_A2_backward,
            groups=Or,
        )
        / dxy
    ) * scales[:, None, None, None]

    dθ = 2 * torch.pi / (u.shape[1] - 1)

    u_padded = F.pad(u, pad=(0, 0, 0, 0, 1, 1), mode="circular")
    A3_forward = (u_padded[:, 2:] - u_padded[:, 1:-1]) / dθ
    A3_backward = (u_padded[:, 1:-1] - u_padded[:, :-2]) / dθ

    u_padded = F.pad(u.permute(dims=(1, 2, 3, 0)), pad=(1, 1), mode="constant", value=pad_value) # Or, H, W, S
    dscale_forward = scales[1] / scales[0] - 1.
    dscale_backward = 1. - scales[0] / scales[1]
    A4_forward = (u_padded[..., 2:] - u_padded[..., 1:-1]) / dscale_forward
    A4_backward = (u_padded[..., 1:-1] - u_padded[..., :-2]) / dscale_backward

    A1 = select_upwind_derivative(A1_forward, A1_backward)
    A2 = select_upwind_derivative(A2_forward, A2_backward)
    A3 = select_upwind_derivative(A3_forward, A3_backward)
    A4 = select_upwind_derivative(A4_forward, A4_backward).permute(3, 0, 1, 2) # S, Or, H, W

    return A1, A2, A3, A4


@lru_cache
def get_A1_A2_kernels(Or, device):
    θs = torch.arange(Or, device=device) * 2 * torch.pi / Or

    cos = torch.cos(θs)
    sin = torch.sin(θs)

    def bilinear_shift_kernel(dx, dy):
        k = torch.zeros(Or, 1, 3, 3, device=device)

        x_neg = (-dx).clamp(min=0)
        x_pos = dx.clamp(min=0)

        y_neg = (-dy).clamp(min=0)
        y_pos = dy.clamp(min=0)

        x_ctr = 1.0 - x_neg - x_pos
        y_ctr = 1.0 - y_neg - y_pos

        wx = torch.stack([x_neg, x_ctr, x_pos], dim=-1)
        wy = torch.stack([y_neg, y_ctr, y_pos], dim=-1)

        k[:, 0] = wy[:, :, None] * wx[:, None, :]
        k[..., 1, 1] += -1.0
        too_small = k.abs() < 1e-4
        k[too_small] = 0.

        return k

    A1_forward = bilinear_shift_kernel(cos, sin)
    A1_backward = bilinear_shift_kernel(-cos, -sin)

    A2_forward = bilinear_shift_kernel(-sin, cos)
    A2_backward = bilinear_shift_kernel(sin, -cos)

    return A1_forward, A1_backward, A2_forward, A2_backward


def select_upwind_derivative(
    d_forward: torch.Tensor, d_backward: torch.Tensor
) -> torch.Tensor:
    min_d_forward = -d_forward
    maximum = torch.maximum(min_d_forward, d_backward)
    return torch.where(
        maximum > 0,
        torch.where(min_d_forward >= d_backward, -maximum, maximum),
        0.0,
    )


def check_convergence(du_dt: torch.Tensor, boundary_condition) -> bool:
    du_dt = du_dt * boundary_condition
    return du_dt.abs().max()


# Backtracking


def geodesic_back_tracking(
    grad_W: torch.Tensor,
    cost: torch.Tensor,
    source_points: tuple[tuple[int, int, int, int], ...],
    target_point: tuple[int, int, int, int],
    a_min: float,
    a_max: float,
    ws: tuple[float, float, float, float],
    dt: float = 1.0,
    n_max: int = 1000,
    tol: float = 0.1,
):
    S, Or = cost.shape[:2]

    point = torch.tensor(target_point, dtype=torch.float)
    γ = [point]
    source_points = torch.tensor(source_points, dtype=torch.float)
    min_distance = torch.inf
    with tqdm(range(1, int(n_max))) as pbar:
        for n in pbar:
            point = geodesic_back_tracking_step(grad_W, cost, point, ws, dt, Or, S, a_min, a_max)
            min_distance, index = distance_in_pixels(point, source_points, Or, S)
            γ.append(point)
            if n % 10 == 0:
                pbar.set_postfix({"error": min_distance.item()})
            if min_distance < tol:
                pbar.set_postfix({"error": min_distance.item()})
                break
    print(f"Geodesic consists of {len(γ)} points.")
    γ[-1] = source_points[index]
    return torch.stack(γ)


def geodesic_back_tracking_step(
    grad_W: torch.Tensor,
    cost: torch.Tensor,
    point: torch.Tensor,
    ws: tuple[float, float, float, float],
    dt: float,
    Or: int,
    S: int,
    a_min: float,
    a_max: float,
) -> torch.Tensor:
    gradient_at_point_LI = vectorfield_quadrilinear_interpolate(grad_W, point, ws, cost)
    θ = point[1] * 2 * torch.pi / Or
    a = a_min * (a_max/a_min)**(point[0] / S)
    # Get gradient with respect to static frame.
    gradient_at_point = vector_LI_to_static(gradient_at_point_LI, θ, a)
    new_point = (
        point
        - dt
        * gradient_at_point
        / (gradient_at_point**2).sum(dim=-1, keepdim=True).sqrt()
    )
    return new_point


# Tesselation

def tesselate_volume(
    shape: tuple[int, int, int, int],
    seeds: tuple[tuple[int, int, int, int], ...],
    a_min: float,
    a_max: float,
    ws: tuple[float, float, float, float],
    dxy: float = 1,
    n_distance_max: float = 1e3,
    n_tracking: int = 256,
    tol: float = 1e-3,
    initial_condition: float = 1e3,
    device="cuda"
) -> torch.Tensor:
    # Compute distance to any component.
    print("Computing distance map and gradient.")
    _, grad_u = eikonal_solver(
        torch.ones(shape, device=device),
        seeds,
        a_min,
        a_max,
        ws,
        dxy=dxy,
        n_max=n_distance_max,
        tol=tol,
        initial_condition=initial_condition,
        device=device,
    )
    grad_u = grad_u.to(device)
    for p in seeds:
        grad_u[*p] = 0.0

    # Find regions.
    print("Back tracking to find tesselation.")
    regions = parallel_back_tracking(
        seeds,
        grad_u,
        a_min,
        a_max,
        dt=1.0,
        n=n_tracking,
    )

    return regions


def parallel_back_tracking(
    seeds: tuple[tuple[int, int, int, int], ...],
    grad_W: torch.Tensor,
    a_min: float,
    a_max: float,
    dt: float = 1.0,
    n: int = 1000,
) -> torch.Tensor:
    S, Or, H, W = grad_W.shape[:4]
    device = grad_W.device
    combined = torch.zeros(S, Or, H, W, device=device)
    for i, seed in enumerate(seeds):
        combined[seed] = i + 1

    points = torch.stack(
        torch.meshgrid(
            torch.arange(S, device=device),
            torch.arange(Or, device=device),
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )[::-1],
        dim=-1,
    ).float()

    for _ in tqdm(range(n)):
        A4_at_points, A3_at_points, A2_at_points, A1_at_points = grid_sample_4d(
            grad_W.permute(4, 0, 1, 2, 3)[None], # 1, 4, S, Or, H, W
            rescale_point(points, W, H, Or, S)[None], # 1, S, Or, H, W, 4
            mode="bilinear",
            align_corners=True,
        )[0] # 4, S, Or, H, W

        a = a_min * (a_max / a_min) ** points[..., 3]
        θ = points[..., 2] * 2 * torch.pi / Or
        cos = torch.cos(θ)
        sin = torch.sin(θ)
        dx_at_points = (cos * A1_at_points - sin * A2_at_points) / a
        dy_at_points = (sin * A1_at_points + cos * A2_at_points) / a
        dθ_at_points = A3_at_points
        da_at_points = A4_at_points / a
        gradient_at_points = torch.stack(
            (dx_at_points, dy_at_points, dθ_at_points, da_at_points), dim=-1
        )

        displacement = (
            dt
            * gradient_at_points
            / ((gradient_at_points**2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)
        )
        points = points - displacement
        points[..., 0].clamp_(0, W - 1)
        points[..., 1].clamp_(0, H - 1)
        points[..., 2] = points[..., 2] % Or
        points[..., 3].clamp_(0, S - 1)

    regions = grid_sample_4d(
        combined[None, None],
        rescale_point(points, W, H, Or, S)[None],
        mode="nearest",
        align_corners=True,
    )[0, 0]
    regions = neighbour_mode_filtering(regions, radius=1)

    return regions


# Helper functions

def vectorfield_quadrilinear_interpolate(
    vectorfield: torch.Tensor,
    index: torch.Tensor,
    ws: tuple[float, float, float, float],
    cost_field: torch.Tensor,
) -> torch.Tensor:
    shape = vectorfield.shape[:-1]
    r = torch.frac(index)
    f = torch.floor(index)
    f = sanitize_index(f, shape)
    c = torch.ceil(index)
    c = sanitize_index(c, shape)

    u0000, v0000, w0000, x0000 = vectorfield[f[0], f[1], f[2], f[3]]
    u0001, v0001, w0001, x0001 = vectorfield[f[0], f[1], f[2], c[3]]
    u0010, v0010, w0010, x0010 = vectorfield[f[0], f[1], c[2], f[3]]
    u0011, v0011, w0011, x0011 = vectorfield[f[0], f[1], c[2], c[3]]
    u0100, v0100, w0100, x0100 = vectorfield[f[0], c[1], f[2], f[3]]
    u0101, v0101, w0101, x0101 = vectorfield[f[0], c[1], f[2], c[3]]
    u0110, v0110, w0110, x0110 = vectorfield[f[0], c[1], c[2], f[3]]
    u0111, v0111, w0111, x0111 = vectorfield[f[0], c[1], c[2], c[3]]
    u1000, v1000, w1000, x1000 = vectorfield[c[0], f[1], f[2], f[3]]
    u1001, v1001, w1001, x1001 = vectorfield[c[0], f[1], f[2], c[3]]
    u1010, v1010, w1010, x1010 = vectorfield[c[0], f[1], c[2], f[3]]
    u1011, v1011, w1011, x1011 = vectorfield[c[0], f[1], c[2], c[3]]
    u1100, v1100, w1100, x1100 = vectorfield[c[0], c[1], f[2], f[3]]
    u1101, v1101, w1101, x1101 = vectorfield[c[0], c[1], f[2], c[3]]
    u1110, v1110, w1110, x1110 = vectorfield[c[0], c[1], c[2], f[3]]
    u1111, v1111, w1111, x1111 = vectorfield[c[0], c[1], c[2], c[3]]

    u = quadrilinear_interpolate(u0000, u0001, u0010, u0011, u0100, u0101, u0110, u0111, u1000, u1001, u1010, u1011, u1100, u1101, u1110, u1111, r)
    v = quadrilinear_interpolate(v0000, v0001, v0010, v0011, v0100, v0101, v0110, v0111, v1000, v1001, v1010, v1011, v1100, v1101, v1110, v1111, r)
    w = quadrilinear_interpolate(w0000, w0001, w0010, w0011, w0100, w0101, w0110, w0111, w1000, w1001, w1010, w1011, w1100, w1101, w1110, w1111, r)
    x = quadrilinear_interpolate(x0000, x0001, x0010, x0011, x0100, x0101, x0110, x0111, x1000, x1001, x1010, x1011, x1100, x1101, x1110, x1111, r)

    cost = scalar_quadrilinear_interpolate(cost_field, index)

    return normalise_LI(torch.Tensor([u, v, w, x]), ws, cost)


def scalar_quadrilinear_interpolate(input: torch.Tensor, index: torch.Tensor) -> float:
    shape = input.shape
    r = torch.frac(index)
    f = torch.floor(index)
    f = sanitize_index(f, shape)
    c = torch.ceil(index)
    c = sanitize_index(c, shape)

    v0000 = input[f[0], f[1], f[2], f[3]]
    v0001 = input[f[0], f[1], f[2], c[3]]
    v0010 = input[f[0], f[1], c[2], f[3]]
    v0011 = input[f[0], f[1], c[2], c[3]]
    v0100 = input[f[0], c[1], f[2], f[3]]
    v0101 = input[f[0], c[1], f[2], c[3]]
    v0110 = input[f[0], c[1], c[2], f[3]]
    v0111 = input[f[0], c[1], c[2], c[3]]
    v1000 = input[c[0], f[1], f[2], f[3]]
    v1001 = input[c[0], f[1], f[2], c[3]]
    v1010 = input[c[0], f[1], c[2], f[3]]
    v1011 = input[c[0], f[1], c[2], c[3]]
    v1100 = input[c[0], c[1], f[2], f[3]]
    v1101 = input[c[0], c[1], f[2], c[3]]
    v1110 = input[c[0], c[1], c[2], f[3]]
    v1111 = input[c[0], c[1], c[2], c[3]]

    v = quadrilinear_interpolate(v0000, v0001, v0010, v0011, v0100, v0101, v0110, v0111, v1000, v1001, v1010, v1011, v1100, v1101, v1110, v1111, r)

    return v


def quadrilinear_interpolate(
    v0000: float,
    v0001: float,
    v0010: float,
    v0011: float,
    v0100: float,
    v0101: float,
    v0110: float,
    v0111: float,
    v1000: float,
    v1001: float,
    v1010: float,
    v1011: float,
    v1100: float,
    v1101: float,
    v1110: float,
    v1111: float,
    r: torch.Tensor,
) -> float:
    v000 = linear_interpolate(v0000, v1000, r[0])
    v001 = linear_interpolate(v0001, v1001, r[0])
    v010 = linear_interpolate(v0010, v1010, r[0])
    v011 = linear_interpolate(v0011, v1011, r[0])
    v100 = linear_interpolate(v0100, v1100, r[0])
    v101 = linear_interpolate(v0101, v1101, r[0])
    v110 = linear_interpolate(v0110, v1110, r[0])
    v111 = linear_interpolate(v0111, v1111, r[0])

    v00 = linear_interpolate(v000, v100, r[1])
    v01 = linear_interpolate(v001, v101, r[1])
    v10 = linear_interpolate(v010, v110, r[1])
    v11 = linear_interpolate(v011, v111, r[1])
    
    v0 = linear_interpolate(v00, v10, r[1])
    v1 = linear_interpolate(v01, v11, r[1])

    v = linear_interpolate(v0, v1, r[2])

    return v


def linear_interpolate(v0: float, v1: float, r: float) -> float:
    return v0 * (1.0 - r) + v1 * r


def normalise_LI(vec: torch.Tensor, ws: tuple[float, float, float, float], cost: float) -> torch.Tensor:
    norm_LI = (
        vec[3] ** 2 * ws[3]**2
        + vec[2] ** 2 * ws[2]**2
        + vec[1] ** 2 * ws[1]**2
        + vec[0] ** 2 * ws[0]**2
    ).sqrt() * cost
    return vec / norm_LI


def sanitize_index(index: torch.Tensor, shape: tuple[int, int, int, int]) -> torch.Tensor:
    return (
        torch.tensor(
            [
                torch.clamp(index[0], 0, shape[0] - 1),
                index[1] % shape[1],
                torch.clamp(index[2], 0, shape[2] - 1),
                torch.clamp(index[3], 0, shape[3] - 1),
            ]
        )
        .round()
        .int()
    )


def vector_LI_to_static(
    vector_LI: torch.Tensor,
    θ: float,
    a: float,
) -> torch.Tensor:
    # A1 = [a cos(θ),a sin(θ),0,0],
    # A2 = [-a sin(θ),a cos(θ),0,0],
    # A3 = [0,0,1,0],
    # A4 = [0,0,0,a], whence
    # ω1 = [cos(θ)/a,-sin(θ)/a,0],
    # ω2 = [sin(θ)/a,cos(θ)/a,0],
    # ω3 = [0,0,1,0],
    # ω4 = [0,0,0,1/a].

    cos = torch.cos(θ)
    sin = torch.sin(θ)

    return torch.tensor(
        [
            vector_LI[0] / a,
            vector_LI[1],
            sin * vector_LI[3] / a + cos * vector_LI[2] / a,
            cos * vector_LI[3] / a - sin * vector_LI[2] / a,
        ]
    )


def distance_in_pixels(
    point: torch.Tensor,
    source_points: torch.Tensor,
    Or: int,
    S: int,
) -> tuple[float, int]:
    distance_vec = point - source_points

    distance = (
        distance_vec[..., 0] ** 2
        + mod_offset(distance_vec[..., 1], Or, -(Or // 2)) ** 2
        + distance_vec[..., 2] ** 2
        + distance_vec[..., 3] ** 2
    ).sqrt()
    min_distance = distance.min()
    index = distance.argmin()

    return min_distance, index


def mod_offset(
    x: torch.Tensor,
    period: float,
    offset: float,
) -> torch.Tensor:
    return x - (x - offset) // period * period


def neighbour_mode_filtering(tensor: torch.Tensor, radius: int = 1) -> torch.Tensor:
    shifts_list = []
    for j in range(-radius, radius + 1):
        for k in range(-radius, radius + 1):
            for l in range(-radius, radius + 1):
                shifts_list.append(torch.roll(tensor, shifts=[j, k, l], dims=(1, 2, 3)))
    shifts_tensor = torch.stack(shifts_list, dim=0).to(torch.long)
    max_val = shifts_tensor.max()

    counts = torch.nn.functional.one_hot(shifts_tensor, num_classes=max_val + 1).sum(dim=0)
    counts[..., 0] = 0

    filtered = counts.argmax(dim=-1)
    no_nonzero = counts.sum(dim=-1) == 0
    filtered[no_nonzero] = 0.0

    return filtered


def rescale_point(points: torch.Tensor, W: int, H: int, Or: int, S: int) -> torch.Tensor:
    rescaled_points = torch.zeros_like(points)
    rescaled_points[..., 0] = 2.0 * (points[..., 0] / (W - 1.0)) - 1.0
    rescaled_points[..., 1] = 2.0 * (points[..., 1] / (H - 1.0)) - 1.0
    rescaled_points[..., 2] = 2.0 * (points[..., 2] / (Or - 1.0)) - 1.0
    rescaled_points[..., 3] = 2.0 * (points[..., 3] / (S - 1.0)) - 1.0
    return rescaled_points


def grid_sample_4d(
    input: torch.Tensor,
    grid: torch.Tensor,
    mode: Literal["bilinear", "nearest"] = "bilinear",
    padding_mode: Literal["zeros", "border", "reflection"] = "zeros",
    align_corners: bool = False,
) -> torch.Tensor:
    """
    4D analogue of `torch.nn.functional.grid_sample`.

    Args:
        `input`: tensor of shape [N, C, S_in, D_in, H_in, W_in].
        `grid`: tensor of shape [N, S_out, D_out, H_out, W_out, 4], with
          normalised coordinates in [-1, 1], ordered (x, y, z, s), i.e.
          indexing (W, H, D, S), as in `torch.nn.functional.grid_sample`.
      Optional:
        `mode`: "bilinear" for quadrilinear interpolation, or "nearest".
        `padding_mode`: "zeros", "border", or "reflection".
        `align_corners`: same meaning as in
          `torch.nn.functional.grid_sample`.

    Returns:
        tensor of shape [N, C, S_out, D_out, H_out, W_out].

    Notes:
        The first three coordinates are handled by a single 3D `grid_sample`
        call on the input with the S dimension folded into the channels. The
        fourth coordinate is then interpolated by gathering the (at most) two
        relevant S slices for every output point. Memory scales as
        N * C * S_in * S_out * D_out * H_out * W_out.
    """
    N, C, S_in, D_in, H_in, W_in = input.shape
    N_grid, S_out, D_out, H_out, W_out, v = grid.shape
    assert N == N_grid, "batch dimensions must match for input and grid!"
    assert v == 4, "grid coordinates must be 4 dimensional!"
    assert mode in ("bilinear", "nearest"), f"mode {mode} not supported!"
    P = S_out * D_out * H_out * W_out

    grid_3D = grid[..., :3].reshape(N, S_out * D_out, H_out, W_out, 3)
    sampled_3D = F.grid_sample(
        input.reshape(N, C * S_in, D_in, H_in, W_in),
        grid_3D,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    ).view(N, C, S_in, P)

    s = grid[..., 3].reshape(N, 1, 1, P)
    if align_corners:
        s = (s + 1.0) / 2.0 * (S_in - 1)
    else:
        s = ((s + 1.0) * S_in - 1.0) / 2.0
    if padding_mode == "border":
        s = s.clamp(0, S_in - 1)
    elif padding_mode == "reflection":
        if align_corners:
            s = _reflect_coordinates(s, 0.0, S_in - 1.0)
        else:
            s = _reflect_coordinates(s, -0.5, S_in - 0.5)
        s = s.clamp(0, S_in - 1)

    def gather_slice(index: torch.Tensor) -> torch.Tensor:
        """Pick slice `index` per output point; zero if out of bounds."""
        in_bounds = (index >= 0) & (index <= S_in - 1)
        index_clamped = index.clamp(0, S_in - 1).expand(N, C, 1, P)
        values = sampled_3D.gather(2, index_clamped)
        return values * in_bounds

    if mode == "nearest":
        out = gather_slice(torch.round(s).long())
    else:
        s_0 = torch.floor(s)
        w_1 = s - s_0
        s_0 = s_0.long()
        out = (1.0 - w_1) * gather_slice(s_0) + w_1 * gather_slice(s_0 + 1)

    return out.view(N, C, S_out, D_out, H_out, W_out)


def _reflect_coordinates(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Reflect `x` into [`low`, `high`], as in PyTorch's `grid_sample`."""
    span = high - low
    if span <= 0:
        return torch.zeros_like(x)
    x = (x - low).abs()
    flips = torch.floor(x / span)
    extra = x - flips * span
    return torch.where(flips % 2 == 0, low + extra, high - extra)

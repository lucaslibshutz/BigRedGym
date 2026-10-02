import math
from types import SimpleNamespace

import pytest
import torch

from gym.envs.go2.go2trot import Go2Trot
from gym.envs.go2.go2trot_config import Go2TrotCfg
from gym.utils.helpers import class_to_dict
from learning.utils import kernel_diagnostics as kd

SIGMA = 0.25


@pytest.mark.parametrize(
    "p, expected",
    [(2, (0.354, 1.716, 0.113, 0.865)), (4, (0.658, 2.153, 0.337, 0.930))],
)
def test_kernel_constants_match_hand_derivation(p, expected):
    assert kd.kernel_costs(p, SIGMA) == pytest.approx(expected, abs=1e-3)


def test_e_star_is_where_the_slope_peaks():
    for p in (2, 4):
        e_star, s_max, _, _ = kd.kernel_costs(p, SIGMA)
        grid = torch.linspace(1e-4, 2.0, 200_001, dtype=torch.float64)
        slope = kd.slope_abs(grid, p, SIGMA)
        assert grid[slope.argmax()].item() == pytest.approx(e_star, abs=1e-4)
        assert slope.max().item() == pytest.approx(s_max, rel=1e-6)


@pytest.mark.parametrize("s", [0.05, 0.2, math.sqrt(SIGMA / 2), 0.6, 1.5])
def test_utilization_matches_gaussian_closed_form(s):
    # e ~ N(0, s^2), p = 2:  U(s) = 2 sqrt(e sigma / pi) * s / (sigma + 2 s^2)
    gen = torch.Generator().manual_seed(0)
    e_abs = (s * torch.randn(1_000_000, generator=gen)).abs()
    expected = 2 * math.sqrt(math.e * SIGMA / math.pi) * s / (SIGMA + 2 * s**2)
    assert kd.stats(e_abs, 2, SIGMA)["U"] == pytest.approx(expected, abs=2e-3)


def test_histogram_density_integrates_to_one_minus_overflow():
    gen = torch.Generator().manual_seed(0)
    e_abs = (0.8 * torch.randn(200_000, generator=gen)).abs()
    density, edges, overflow = kd.histogram(e_abs, 2, SIGMA)
    area = (density * (edges[1] - edges[0])).sum().item()
    assert overflow > 0
    assert area == pytest.approx(1 - overflow, abs=1e-4)


def _go2trot_with_random_state(n=512):
    gen = torch.Generator().manual_seed(0)
    rand = lambda *shape, scale=1.0: scale * torch.randn(*shape, generator=gen)
    task = Go2Trot.__new__(Go2Trot)
    task.cfg = SimpleNamespace(
        reward_settings=SimpleNamespace(tracking_sigma=SIGMA, base_height_target=0.36)
    )
    task.scales = {
        "base_ang_vel": 0.3,
        "base_height": 0.3,
        "dof_vel": torch.tensor(4 * [2.0, 2.0, 4.0]),
        "dof_pos_obs": torch.tensor(4 * [1.0472, 2.53075, 0.94247]),
    }
    task.reward_scales = class_to_dict(Go2TrotCfg.reward_settings.reward_scales, "cpu")
    task.commands = rand(n, 3, scale=1.5)
    task.base_lin_vel = rand(n, 3, scale=1.5)
    task.base_ang_vel = rand(n, 3, scale=1.5)
    task.projected_gravity = rand(n, 3, scale=0.3)
    task.base_height = 0.36 + rand(n, 1, scale=0.1)
    task.dof_pos = rand(n, 12, scale=0.5)
    task.default_dof_pos = rand(1, 12, scale=0.5)
    task.dof_vel = rand(n, 12, scale=3.0)
    return task


def test_every_error_term_reproduces_its_reward():
    """reward == sum_j k(e_j) guards against _error_* and _reward_* drifting apart."""
    task = _go2trot_with_random_state()
    names = [m.replace("_error_", "") for m in dir(task) if m.startswith("_error_")]
    assert set(names) == {
        "ang_vel_xy", "dof_near_home", "dof_vel", "min_base_height",
        "orientation", "tracking_ang_vel", "tracking_lin_vel",
    }
    for name in names:
        e = getattr(task, f"_error_{name}")()
        p = task.kernel_p.get(name, 2)
        k = kd.kernel(e.abs(), p, SIGMA)
        rebuilt = k.sum(dim=1) if k.dim() == 2 else k
        reward = getattr(task, f"_reward_{name}")()
        torch.testing.assert_close(rebuilt, reward, rtol=1e-5, atol=1e-6, msg=name)


def test_error_terms_do_not_write_into_state():
    task = _go2trot_with_random_state()
    before = task.base_height.clone()
    task._error_min_base_height()
    task._reward_min_base_height()
    torch.testing.assert_close(task.base_height, before)


def test_overlay_figure_renders():
    gen = torch.Generator().manual_seed(0)
    e_abs = (0.3 * torch.randn(10_000, generator=gen)).abs()
    stats = kd.stats(e_abs, 2, SIGMA)
    density, edges, overflow = kd.histogram(e_abs, 2, SIGMA)
    fig = kd.overlay_figure("test", density, edges, 2, SIGMA, stats, overflow, 25)
    assert len(fig.axes) == 2


def test_default_reward_scales_reproduce_original_rewards():
    """Regression: the pre-split reward formulas, which read `scaling` directly."""
    task = _go2trot_with_random_state()
    sc, sigma = task.scales, SIGMA
    sq = lambda x: torch.exp(-torch.square(x) / sigma)
    lin = (task.commands[:, :2] - task.base_lin_vel[:, :2]) / (
        1.0 + task.commands[:, :2].abs()
    )
    height = torch.clamp((task.base_height - 0.36) / sc["base_height"], max=0)
    original = {
        "ang_vel_xy": sq(task.base_ang_vel[:, :2] / sc["base_ang_vel"]).sum(1),
        "orientation": torch.exp(
            -torch.square(task.projected_gravity[:, :2]) / sigma
        ).sum(1),
        "min_base_height": sq(height.flatten()),
        "tracking_lin_vel": torch.exp(-lin.square().sum(1) / sigma),
        "tracking_ang_vel": sq(
            torch.square((task.commands[:, 2] - task.base_ang_vel[:, 2]) / 2.5)
        ),
        "dof_vel": sq(task.dof_vel / sc["dof_vel"]).sum(1),
        "dof_near_home": sq(
            (task.dof_pos - task.default_dof_pos) / sc["dof_pos_obs"]
        ).sum(1),
    }
    for name, expected in original.items():
        reward = getattr(task, f"_reward_{name}")()
        torch.testing.assert_close(reward, expected, rtol=1e-5, atol=1e-6, msg=name)


def test_reward_scales_actually_rescale_the_error():
    task = _go2trot_with_random_state()
    base = task._error_ang_vel_xy().clone()
    task.reward_scales["ang_vel_xy"] = task.reward_scales["ang_vel_xy"] * 2
    torch.testing.assert_close(task._error_ang_vel_xy(), base / 2)

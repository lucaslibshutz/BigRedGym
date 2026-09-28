import math
import torch
from matplotlib.figure import Figure

SAT, DEAD = 0.95, 0.05

def kernel_costs(p, sigma):
    e_star = ((p-1) * sigma / p) ** (1 / p)
    s_max = (p / sigma) * e_star ** (p - 1) * math.exp(-(p-1) / p)
    e_sat = (-sigma * math.log(SAT)) ** (1 / p)
    e_dead = (-sigma * math.log(DEAD)) ** (1 / p)
    return e_star, s_max, e_sat, e_dead

def kernel(e_abs, p, sigma):
    return torch.exp(-e_abs**p / sigma)

def slope_abs(e_abs, p, sigma):
    return (p / sigma) * e_abs ** (p - 1) * kernel(e_abs, p, sigma)

def stats(e_abs, p, sigma):
    e_star, s_max, e_sat, e_dead = kernel_costs(p, sigma)
    q = torch.quantile(e_abs[:100_000], torch.tensor([0.1, 0.5, 0.9], device=e_abs.device))
    return {
        "U": (slope_abs(e_abs, p, sigma).mean() / s_max).item(),
        "f_sat": (e_abs < e_sat).float().mean().item(),
        "f_dead": (e_abs > e_dead).float().mean().item(),
        "q10": q[0].item(),
        "q50": q[1].item(),
        "q90": q[2].item(),
    }

def histogram(e_abs, p, sigma, bins=60):
    e_max = 2 * kernel_costs(p, sigma)[3]
    counts = torch.histc(e_abs, bins=bins, min=0, max=e_max)
    width = e_max / bins
    density = counts / (e_abs.numel() * width) # divide by ALL n, not just in range
    edges = torch.linspace(0, e_max, bins + 1)
    overflow = (e_abs > e_max).float().mean().item()
    return density.cpu(), edges, overflow

def overlay_figure(name, density, edges, p, sigma, stats_dict, overflow, iteration):
    """Histogram of visited |e| with k(e), |k'(e)|/s_max and signal density on top."""
    e_star, s_max, e_sat, e_dead = kernel_costs(p, sigma)
    centers = 0.5 * (edges[:-1] + edges[1:])
    grid = torch.linspace(0, float(edges[-1]), 400)
    slope = (slope_abs(centers, p, sigma) / s_max).numpy()
    density, centers = density.numpy(), centers.numpy()
    k_grid = kernel(grid, p, sigma).numpy()
    slope_grid = (slope_abs(grid, p, sigma) / s_max).numpy()
    grid = grid.numpy()

    fig = Figure(figsize=(7, 4), dpi=100)
    ax = fig.add_subplot()
    ax.bar(centers, density, width=float(edges[1] - edges[0]), color="0.75",
           label=r"$\hat h(e)$ visited")
    ax.fill_between(centers, density * slope, step="mid", alpha=0.5,
                    color="tab:orange", label=r"$\hat h \cdot |k'|/s_{max}$ signal")
    ax.set_xlabel(r"$|e|$ (scaled error)")
    ax.set_ylabel("density")

    ax2 = ax.twinx()
    ax2.plot(grid, k_grid, color="tab:blue", label=r"$k(e)$")
    ax2.plot(grid, slope_grid, color="tab:red",
             label=r"$|k'(e)|/s_{max}$")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("kernel / normalized slope")
    for x, style in ((e_sat, ":"), (e_star, "--"), (e_dead, ":")):
        ax2.axvline(x, color="k", linestyle=style, linewidth=0.8)

    ax.set_title(
        f"{name} (p={p})  it {iteration}   U={stats_dict['U']:.2f}  "
        f"sat={stats_dict['f_sat']:.2f}  dead={stats_dict['f_dead']:.2f}  "
        f"overflow={overflow:.2f}",
        fontsize=9,
    )
    handles = ax.get_legend_handles_labels()
    handles2 = ax2.get_legend_handles_labels()
    ax.legend(handles[0] + handles2[0], handles[1] + handles2[1],
              fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig

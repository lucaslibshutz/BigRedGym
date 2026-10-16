# Running BigRedGym

## Setup
```
uv sync --frozen
```
Always pass `--frozen` to `uv run`: without it, uv tries to resolve the optional
Unitree SDK checkout (`thirdparty/unitree_sdk2_python`) and fails when it's absent.

## Train
```
uv run --frozen scripts/train.py --task=go2trot --device=cuda:0 --headless
```
- `--device` defaults to `cpu`; pass `cuda:0` for the GPU (MuJoCo-Warp).
- Checkpoints land in `logs/<experiment_name>/<run>/`.
- `go2trot` sets its own `num_envs` in `gym/envs/go2/go2trot_config.py`;
  editing `LeggedRobotCfg.env.num_envs` in the base config has no effect on it.
- Steps per env per iteration = `algorithm.rollout_size // num_envs`
  (`learning/runners/on_policy_runner.py`). Scale `rollout_size` with `num_envs`.

## VSim backend (optional, licensed)
Setup, system libraries (`libczmq.so.4`, `libassimp.so.5`; Arch needs AUR
workarounds) and license activation: [`thirdparty/README.md`](thirdparty/README.md).
```
uv sync --frozen --extra vsim
uv run --frozen --extra vsim --env-file .env.vsim scripts/train.py --task mini_cheetah \
    --backend vsim --device cuda:0 --num_envs 4096 --headless
```
- Play (vsim is CUDA-only, so `--device cuda:0` is required; the vlearn window
  needs `thirdparty/vlearn/shaders/`):
  `uv run --frozen --extra vsim --env-file .env.vsim scripts/play.py --task mini_cheetah --backend vsim --device cuda:0`
  — loads the latest run/checkpoint under `logs/<experiment_name>/`; pick one with
  `--load_run <dir> --checkpoint <iter>`. Keyboard teleop is on by default.
- Upgrading the engine wheel: drop the new `.whl` into `thirdparty/vlearn/`, update
  the `vlearn` path in `[tool.uv.sources]`, then `uv lock --upgrade-package vlearn`.
  `--frozen` sync fails with "Distribution not found" until the lock is updated.
- `pytest -m vsim` errors at setup by design (the `Q2_VSIM_TESTS` gate is deferred).
  Smoke-test with a short training run instead (`--num_envs 256 --max_iterations 3
  --disable_wandb`).
- vlearn docs: there's a prebuilt Sphinx site at `thirdparty/vlearn/docs/vlearn-docs/html/`
  (unzipped from `vlearn-docs.zip`). `uv run vsim-docs` serves it at http://127.0.0.1:8765
  and opens a browser (`--port N`, `--no-browser`; Ctrl+C stops it). Serve rather than
  open the files directly: Sphinx search fails over `file://`. Type stubs for IDE completion:
  `thirdparty/vlearn/docs/stubs/vlearn_bindings.pyi`.

## Play
```
uv run --frozen scripts/play.py --task=go2trot
```

## Weights & Biases
- One-time login (the key is stored in `~/.netrc`):
  `uv run --frozen wandb login --relogin` — paste the key once; the prompt doesn't echo.
- `user/wandb_config.json` (gitignored) sets the entity and project; the template is
  `user/wandb_config_default.json`. Override with `--wandb_entity` / `--wandb_project`,
  or turn it off with `--disable_wandb`.
- Logged per iteration: `rewards/<term>` for each reward weight, `rewards/total_rewards`,
  `algorithm/{mean_value_loss, mean_surrogate_loss, learning_rate}`,
  `actor/{action_std, entropy}`.
- Gotcha: the run's hyperparameters are **not** uploaded (`wandb.config` stays empty);
  only the `gym/` source is saved under the run's Code tab.
- Check the login: `.venv/bin/python -c "import wandb; print(wandb.Api().viewer.username)"`.

## Tests
```
uv run --frozen python -m pytest -q
```

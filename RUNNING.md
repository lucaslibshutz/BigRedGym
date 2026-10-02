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

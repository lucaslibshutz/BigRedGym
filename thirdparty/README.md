# thirdparty/vlearn — vsim engine drop zone

The vsim backend's engine (`vlearn`) is closed-source and licensed; its
binaries cannot be committed (100 KB file cap + licensing). The `vlearn/`
subdirectory holds the machine-local pieces, all gitignored:

- `vlearn-0.3.15+cu130-cp311-cp311-linux_x86_64.whl` — the selected engine wheel
  (from the vendor / the vlearn SDK repo). `[tool.uv.sources]` in
  pyproject.toml points here; install with `uv sync --frozen --extra vsim`.
  The CUDA 13.0 variant matches the locked Linux PyTorch build.
- `License.key` — your node-locked license key (one line); found via
  `VL_LICENSE_KEY_PATH` (set in `.env.vsim`).
- `TurboActivate.dat` — vendor-shipped product-definition file (copied from
  the vlearn SDK repo; version-independent); found via
  `VL_TURBO_ACTIVATE_PATH`.
- `cache/` — the engine's cache tree (`cache/tmp/` + marker file
  `cache/donotremove.txt`); required by `vsim::Path::findCachePath`.
  `VSimBackend.setup()` self-heals it if missing.
- `shaders/` + `assets/{VsimTile,Skybox}.png` — renderer files, needed only
  for the viewer (copied from the vlearn SDK repo root). Headless runs work
  without them.

## System libraries

The engine's native libs link against two system libraries the wheel does not
bundle: `libczmq.so.4` (`libgym.so`, `librender.so`) and `libassimp.so.5`
(`libvsimdynamics.so`, `libgym.so`, `librender.so`). Without them,
`import vlearn` fails with `OSError: libassimp.so.5: cannot open shared object
file`. Check with:

```bash
ldconfig -p | grep -E 'libczmq.so.4|libassimp.so.5'
```

- Ubuntu 24.04: `sudo apt install libczmq4 libassimp5`.
- Arch: both come from the AUR. The `assimp` package in `extra` is 6.x
  (`libassimp.so.6`) and does not satisfy the `.so.5` soname.
  - `czmq`: its `check()` flakily times out in the `zsock` test; build with
    `yay -S --mflags --nocheck czmq`.
  - `libassimp5-compat` (5.4.3-1): fails with `fatal error: unzip.h` because
    Arch's `minizip` installs headers under `/usr/include/minizip/` and its
    `.pc` exports no `-I`. Add `-DASSIMP_BUILD_MINIZIP=ON` to the PKGBUILD's
    cmake call (assimp's bundled `contrib/unzip`), then
    `makepkg -f && sudo pacman -U libassimp5-compat-5.4.3-1-x86_64.pkg.tar.zst`.

The vendor zip (~4.4 GB, zip64) trips Info-ZIP `unzip`'s overlapped-components
"zip bomb" check falsely; extract it with `7z x` or `bsdtar -xf`.

## Install and activate

From the repository root:

```bash
uv sync --frozen --extra vsim
```

Activate after the first install and whenever `License.key` is replaced or
renewed. Internet access is required during activation:

```bash
# Run from the repository root.
(
  cd thirdparty/vlearn
  LD_LIBRARY_PATH=../../.venv/lib/python3.11/site-packages/vlearn/lib \
  VL_WORKING_DIRECTORY="$PWD" \
  VL_TURBO_ACTIVATE_PATH="$PWD/TurboActivate.dat" \
  VL_LICENSE_KEY_PATH="$PWD/License.key" \
  ../../.venv/bin/python -c \
    'import vlearn as v; v.create_gym(with_render=False, with_window=False); v.delete_gym(); print("vsim activation probe succeeded")'
)
```

The final line should be `vsim activation probe succeeded`. This checks engine
activation only. The VSim backend code, optional extra, and marked tests remain
in the repository, but the local test launcher and support workflow are deferred.
See [validation scope](../genAI_skills/README_MUJOCO.md#limitations-and-validation-scope).

Run normal commands from the repository root:

```bash
uv run --frozen --extra vsim --env-file .env.vsim scripts/train.py --task mini_cheetah \
    --backend vsim --device cuda:0 --num_envs 4096 --headless
```

`--frozen` installs the selected locked dependencies without requiring the
unrelated optional Unitree checkout. The Go2 hardware SDK has its own gitignored
directory, `thirdparty/unitree_sdk2_python/`, populated by
`uv run --frozen python scripts/fetch_unitree_sdk.py`. See
[`README_DEPLOY.md`](../README_DEPLOY.md) for its native DDS prerequisite and
editable installation.

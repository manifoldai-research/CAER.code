<h1 align="center">CAER</h1>

<p align="center"><b>Causal Action Effect Reweighting for World Model Training</b></p>

<p align="center">
  <a href="docs/TRAINING.md">Training Guide</a> · <a href="LICENSE">License</a>
</p>

Training code for **CAER**: an online, annotation-free objective that reallocates a fixed gradient budget toward tokens whose predicted future is sensitive to the action. Drop-in replacement for space–time-uniform MSE on action-conditioned world models.

<p align="center">
  <img src="assets/teaser.png" alt="CAER overview" width="100%">
</p>

## Repository

```text
.
├── config/paths.env.example  # copy to paths.env and fill in local paths
├── scripts/                  # train / preflight entry points
├── docs/TRAINING.md
└── sources/
    ├── cap/                  # Arm / Camera / PoseAnything
    └── libero/               # LIBERO
```

Machine paths go in `config/paths.env` (gitignored). Do not hard-code absolute paths.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r sources/cap/requirements.txt   # or sources/libero/requirements.txt

cp config/paths.env.example config/paths.env
# edit every /path/to/... value

bash scripts/preflight.sh all

bash scripts/train_libero.sh s_max1
bash scripts/train_arm.sh s_only
bash scripts/train_camera.sh s_only
bash scripts/train_poseanything.sh s_only
```

`s_only` / `s_max1` are the CAER variants. Pass `uniform` for the matched MSE baseline.

```bash
DRY_RUN=1 bash scripts/train_camera.sh s_only
SMOKE=1   bash scripts/train_poseanything.sh s_only
```

Data contracts, resume, and hyperparameters: [docs/TRAINING.md](docs/TRAINING.md).

Loss implementation: `sources/cap/videox_fun/training/method1_focused_loss.py`.

## Citation

```bibtex
@article{fang2026caer,
  title   = {CAER: Causal Action Effect Reweighting for World Model Training},
  author  = {Fang, Jianjie and Liu, Xvyuan and Wang, Ziyou and Tang, Rongze
             and Wang, Zhaolu and Li, Zhuohang and Zhang, Xin and Su, Haisheng
             and Gao, Chen and Wu, Wei and Chen, Xinlei and Li, Yong},
  year    = {2026}
}
```

## License

[Apache License 2.0](LICENSE). Model weights and datasets keep their original licenses. See [NOTICE](NOTICE).

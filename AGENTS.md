# Repository Guidelines

## Project Structure & Module Organization
- `project.md` captures the lightweight face recognition requirements; keep it updated as specs evolve.
- Place production code under `src/` with subpackages such as `src/models/` for architectures, `src/pipelines/` for preprocessing/postprocessing, and `src/utils/` for shared helpers.
- Staging experiments live in `notebooks/` (Jupyter) and `scripts/` for CLI workflows; migrate stable logic into `src/` before release.
- Tests track the package layout in `tests/`, while small reference images and embeddings go in `assets/`. Do not version checkpoints; store them under `artifacts/` ignored by Git.

## Build, Test, and Development Commands
- `python -m pip install -r requirements.txt` - install pinned runtime and tooling dependencies.
- `pre-commit run --all-files` - apply formatters (`black`, `ruff`, `isort`) and static checks before pushing.
- `pytest` - run the default unit test suite; add `-m slow` for performance coverage.
- `python scripts/train.py --config configs/mobile.yaml` - train the mobile-friendly backbone; reuse configs for experiments.
- `python scripts/export.py --weights artifacts/latest.pt --device cpu` - create exportable inference graphs for deployment tests.

## Coding Style & Naming Conventions
- Target Python 3.10+, 4-space indentation, type hints on public functions, and NumPy-style docstrings for modules interacting with tensors.
- Modules, functions, and files use `snake_case`; classes use `PascalCase`; constants in `UPPER_SNAKE_CASE`.
- Prefer TorchScript-friendly constructs; avoid dynamic control flow that breaks ONNX export.

## Testing Guidelines
- Write `pytest` cases under `tests/<module>/test_<feature>.py` mirroring `src`.
- Cover input edge cases (lighting, occlusion) with parametrized tests and synthetic tensors.
- Keep fast unit tests under 1s; gate longer silicon benchmarks with `@pytest.mark.slow`.
- Maintain >=85% statement coverage reported via `pytest --cov=src --cov-report=term-missing`.

## Commit & Pull Request Guidelines
- Follow Conventional Commits (`feat:`, `fix:`, `refactor:`) with <=72-char subject and concise body bullets for context.
- Reference issue IDs in the footer (`Refs #123`) and note model size and latency deltas when relevant.
- Pull requests must include a short narrative, before/after metrics, test results, and updated docs/configs.
- Attach screenshots or tensorboard exports when UI, training curves, or performance dashboards change.

## Model & Data Handling
- Keep personal or sensitive datasets out of the repo; document required public sets in `docs/datasets.md`.
- Record hardware/software assumptions in `configs/hardware.yaml` and update when kernels or quantization settings change.
- When introducing new weights, publish SHA256 hashes in release notes and upload binaries to the shared model registry.

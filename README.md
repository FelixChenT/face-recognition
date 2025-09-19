# Lightweight Face Recognition

A lightweight, real-time face recognition stack targeting mobile and edge deployments.

## Project Layout
- src/ – production modules (models, pipelines, utilities).
- scripts/ – CLI entry points for training, evaluation, and export workflows.
- configs/ – YAML configurations for experiments and hardware assumptions.
- 	ests/ – pytest suites mirroring the source tree.
- ssets/ – small sample images and embeddings for unit tests.
- 
otebooks/ – exploratory experiments before uplifting to src/.
- docs/ – additional documentation (datasets, research notes).

## Quickstart
1. Create a virtual environment targeting Python 3.10+.
2. Install dependencies: python -m pip install -r requirements.txt.
3. Run the unit tests: pytest.
4. Launch training with the default config: python scripts/train.py --config configs/mobile.yaml.

## Contributing
Follow the repository guidelines in AGENTS.md for code style, testing, and documentation expectations.

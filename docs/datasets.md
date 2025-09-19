# Datasets

Document the public datasets required for training and evaluation. Avoid checking personal or sensitive data into the repository.

## Recommended Public Sources
- [MS1M-ArcFace](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo) – large-scale identities for representation learning.
- [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) – facial attributes, occlusion cases, and pose variety.
- [LFW](http://vis-www.cs.umass.edu/lfw/) – standard verification benchmark for lightweight setups.

## Data Preparation
1. Download the dataset archives to a secure location outside of the repository.
2. Normalize image resolutions to 112x112 with aligned faces when possible.
3. Generate train/validation splits via configuration under configs/.
4. Cache preprocessed tensors under rtifacts/ (ignored by git).

## Ethics & Compliance
- Ensure datasets meet regional privacy regulations.
- Provide model cards summarizing known biases before shipping models.

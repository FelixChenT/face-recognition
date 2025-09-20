# 数据集

记录训练和评估所需的公共数据集。避免将个人或敏感数据检入存储库。

## 推荐的公共来源
- [MS1M-ArcFace](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo) – 用于表示学习的大规模身份。
- [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) – 面部属性、遮挡案例和姿势多样性。
- [LFW](http://vis-www.cs.umass.edu/lfw/) – 用于轻量级设置的标准验证基准。

## 数据准备
1. 将数据集存档下载到存储库之外的安全位置。
2. 尽可能将图像分辨率标准化为 112x112，并对齐人脸。
3. 通过 `configs/` 下的配置生成训练/验证拆分。
4. 将预处理的张量缓存在 `rtifacts/` 下（被 git 忽略）。

## 道德与合规
- 确保数据集符合地区隐私法规。
- 在发布模型之前，提供总结已知偏差的模型卡。
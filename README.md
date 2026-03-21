# graduation_project_feature_extra

本仓库用于多模态对话特征提取与评估，主流程运行在 `conda` 的 `security` 环境中。

## 1. 快速迁移（新机器）

```bash
git clone https://github.com/JDLu2003/graduation_project_feature_extra.git
cd graduation_project_feature_extra
git submodule update --init --recursive
```

## 2. 环境准备

推荐先创建并激活 `security` 环境（Python 3.10）：

```bash
conda create -n security python=3.10 -y
conda activate security
```

然后安装三个代码仓库中的依赖：

```bash
# 主仓库依赖（按需安装）
pip install pyyaml numpy torch torchvision tqdm pillow opencv-python clip-anytorch

# 子模块：face_name_id
pip install -r face_name_id/requirements.txt

# 子模块：talknet_asd
pip install -r talknet_asd/requirement.txt

# 子模块：audio_opensmile
pip install -e audio_opensmile
```

如果你在 Mac 上使用 `ffmpeg`，可先安装：

```bash
brew install ffmpeg
```

## 3. 数据与模型准备

请确保以下资源在本机存在，并在 `config.yaml` 中配置正确路径：

1. `paths.dev_txt`
2. `paths.video_dir`
3. `paths.feat_out`
4. `face_scene_fr_config.face_checkpoint`
5. `talknet_asd/pretrain_TalkSet.model`

提示：`config.yaml` 当前使用的是相对路径（如 `../data_set/...`），迁移后请按你的目录结构调整。

## 4. 最小可用验证

先查看可用命令：

```bash
make help
```

执行一个轻量验证（角色统计）：

```bash
make role-stats
```

执行 TalkNet 小样本验证：

```bash
make talknet-smoke
```

## 5. 子模块说明

本仓库依赖以下子模块：

1. `face_name_id` -> `https://github.com/JDLu2003/graduation_project_face_name_train.git`
2. `talknet_asd` -> `https://github.com/JDLu2003/TalkNet-ASD.git`
3. `audio_opensmile` -> `https://github.com/JDLu2003/audio_opensmile.git`

若子模块 commit 更新，拉取主仓库后执行：

```bash
git submodule update --init --recursive
```

## 6. 常见问题

1. 子模块为空目录  
执行：`git submodule update --init --recursive`

2. 模型或数据找不到  
检查 `config.yaml` 的路径是否与本机目录一致。

3. CUDA/MPS 不可用  
本项目默认可在 CPU 下运行，必要时在配置中将 device 设为 `cpu`。

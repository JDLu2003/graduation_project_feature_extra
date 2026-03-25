SHELL := /bin/zsh

ROOT ?= $(CURDIR)
CONDA_ENV ?= security
PYTHON ?= python
RUN := conda run --no-capture-output -n $(CONDA_ENV) $(PYTHON) -u

CONFIG ?= $(ROOT)/config.yaml
FACE_TRAIN_CONFIG ?= $(ROOT)/face_name_id/configs/facenet_fr.yaml
MAX_DIALOGUES ?= 2

ROLE_SCRIPT := $(ROOT)/scripts/stat_role_frequencies.py
FACE_BOX_SCRIPT := $(ROOT)/scripts/extract_faces_with_ffmpeg_mtcnn.py
TRAIN_FACE_SCRIPT := $(ROOT)/face_name_id/scripts/train_facenet_fr.py
MAIN_SCRIPT := $(ROOT)/main.py

ROLE_SORT ?= utterances
ROLE_OUT_CSV ?= $(ROOT)/logs/role_frequencies.csv
ROLE_OUT_JSON ?= $(ROOT)/logs/role_frequencies.json
VIDEO ?=
FPS ?= 25
DEVICE ?= auto
MIN_CONF ?= 0.95

.PHONY: help config-help train-face-model train-face-model-skip-split role-stats role-stats-export face-box-video smoke full verify-and-full one-click

help:
	@echo "Project quickstart:"
	@echo "  1) git clone <repo> && git submodule update --init --recursive"
	@echo "  2) conda create -n $(CONDA_ENV) python=3.10 -y"
	@echo "  3) conda activate $(CONDA_ENV)"
	@echo "  4) pip install -r face_name_id/requirements.txt"
	@echo "  5) pip install pyyaml numpy torch torchvision tqdm pillow opencv-python clip-anytorch facenet-pytorch"
	@echo "  6) 配置文件字段见: make config-help"
	@echo "  7) 先训练人物识别模型: make train-face-model"
	@echo "  8) 再跑 face_scene_fr 冒烟: make smoke"
	@echo ""
	@echo "Common targets:"
	@echo "  make train-face-model            # 训练 face_name_id 的 FaceNet-FR 模型"
	@echo "  make smoke                       # face_scene_fr 冒烟，默认前 $(MAX_DIALOGUES) 个 dialogue"
	@echo "  make full                        # face_scene_fr 全量提取"
	@echo "  make role-stats                  # 统计角色频次"
	@echo "  make face-box-video VIDEO=/abs.mp4"
	@echo ""
	@echo "Override examples:"
	@echo "  make smoke CONFIG=$(ROOT)/config.yaml MAX_DIALOGUES=3"
	@echo "  make train-face-model FACE_TRAIN_CONFIG=$(ROOT)/face_name_id/configs/facenet_fr.yaml"

config-help:
	@echo "[必须配置] 主仓库文件: config.yaml"
	@echo "  paths.dev_txt                    # dev.txt 路径"
	@echo "  paths.video_dir                  # 语句视频根目录"
	@echo "  paths.feat_out                   # 特征输出目录"
	@echo "  extractor.active_type            # 需要设置为 face_scene_fr"
	@echo "  face_scene_fr_config.device      # 推荐 auto，新机有 NVIDIA 时会自动使用 cuda"
	@echo "  face_scene_fr_config.face_checkpoint"
	@echo "                                   # 训练完成后的 best.pt 路径"
	@echo ""
	@echo "[必须配置] 子仓库文件: face_name_id/configs/facenet_fr.yaml"
	@echo "  paths.image_root                 # 训练图片目录"
	@echo "  paths.rejected_root              # 过滤图片目录，可为空目录"
	@echo "  paths.split_dir                  # 训练/验证/测试划分输出目录"
	@echo "  paths.output_dir                 # checkpoint 输出目录"
	@echo ""
	@echo "[推荐检查]"
	@echo "  face_name_id/configs/facenet_fr.yaml -> data.batch_size / train.epochs"
	@echo "  config.yaml -> face_scene_fr_config.person_num_frames / face_batch_size"

train-face-model:
	@echo "[train-face-model] CONFIG=$(FACE_TRAIN_CONFIG)"
	$(RUN) $(TRAIN_FACE_SCRIPT) --config $(FACE_TRAIN_CONFIG)

train-face-model-skip-split:
	@echo "[train-face-model-skip-split] CONFIG=$(FACE_TRAIN_CONFIG)"
	$(RUN) $(TRAIN_FACE_SCRIPT) --config $(FACE_TRAIN_CONFIG) --skip-split

role-stats:
	$(RUN) $(ROLE_SCRIPT) --config $(CONFIG) --sort $(ROLE_SORT)

role-stats-export:
	$(RUN) $(ROLE_SCRIPT) --config $(CONFIG) --sort $(ROLE_SORT) --out-csv $(ROLE_OUT_CSV) --out-json $(ROLE_OUT_JSON)

face-box-video:
	@test -n "$(VIDEO)" || (echo "VIDEO is required, e.g. make face-box-video VIDEO=/abs/path/to/video.mp4" && exit 1)
	$(RUN) $(FACE_BOX_SCRIPT) "$(VIDEO)" --fps $(FPS) --device $(DEVICE) --min-confidence $(MIN_CONF)

smoke:
	@echo "[smoke] face_scene_fr smoke run with CONFIG=$(CONFIG), MAX_DIALOGUES=$(MAX_DIALOGUES)"
	$(RUN) $(MAIN_SCRIPT) --config $(CONFIG) --smoke --max-dialogues $(MAX_DIALOGUES)

full:
	@echo "[full] face_scene_fr full run with CONFIG=$(CONFIG)"
	$(RUN) $(MAIN_SCRIPT) --config $(CONFIG)

verify-and-full: smoke full

one-click: verify-and-full

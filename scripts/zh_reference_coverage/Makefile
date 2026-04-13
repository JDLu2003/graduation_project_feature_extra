SHELL := /bin/zsh

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
CONDA_ENV ?= EL_LJD
PYTHON ?= python
RUN := conda run --no-capture-output -n $(CONDA_ENV) $(PYTHON) -u

SCRIPT := $(ROOT)/scripts/zh_reference_coverage/run_reference_coverage_eval.py
PIPELINE_SCRIPT := $(ROOT)/scripts/zh_reference_coverage/run_facenet_cluster_pipeline.py
TXT_PATH ?= /data16T_1/sunshengzhe/lujiading/data_zh/dev/dev.txt
VIDEO_DIR ?= /data16T_1/sunshengzhe/lujiading/data_zh/dev/Video_dev
REFERENCE_ROOT ?= /data16T_2/sunshengzhe/reference_face_zh
OUTPUT_ROOT ?= $(ROOT)/scripts/zh_reference_coverage/artifacts
RUN_NAME ?= latest
THRESHOLDS ?= 1 3 5 10 15 20
PIPELINE_RUN_NAME ?= pipeline_latest
DEVICE ?= cuda
NUM_FRAMES ?= 6
DETECT_BATCH_SIZE ?= 8
EMBED_BATCH_SIZE ?= 256
REFERENCE_EMBED_BATCH_SIZE ?= 256
CLUSTER_THRESHOLD ?= 0.72
REFERENCE_MATCH_THRESHOLD ?= 0.80
REFERENCE_MATCH_MARGIN ?= 0.03
FACE_VERIFY_THRESHOLD ?= 0.75
MAX_FACES_PER_FRAME ?= 8
USE_FP16 ?= 1
SMOKE_MAX_DIALOGUES ?= 8

.PHONY: help run pipeline pipeline-smoke clean

help:
	@echo "zh_reference_coverage targets:"
	@echo "  make run                          # 静态 reference 覆盖率评估"
	@echo "  make pipeline                     # MTCNN + FaceNet + 聚类 + reference 检索评估"
	@echo "  make pipeline-smoke               # 只跑少量 dialogue 的冒烟评估"
	@echo "  make clean                        # 删除 latest 输出"
	@echo ""
	@echo "Override examples:"
	@echo "  make run CONDA_ENV=EL_LJD"
	@echo "  make pipeline PIPELINE_RUN_NAME=run_20260413 DEVICE=cuda"
	@echo ""
	@echo "A6000 10G 推荐起始参数:"
	@echo "  DETECT_BATCH_SIZE=8"
	@echo "  EMBED_BATCH_SIZE=256"
	@echo "  REFERENCE_EMBED_BATCH_SIZE=256"
	@echo "  USE_FP16=1"
	@echo "  FACE_VERIFY_THRESHOLD=0.75"

run:
	$(RUN) $(SCRIPT) \
		--txt-path $(TXT_PATH) \
		--reference-root $(REFERENCE_ROOT) \
		--output-root $(OUTPUT_ROOT) \
		--run-name $(RUN_NAME) \
		$(foreach thr,$(THRESHOLDS),--threshold $(thr)) \
		--overwrite

pipeline:
	$(RUN) $(PIPELINE_SCRIPT) \
		--txt-path $(TXT_PATH) \
		--video-dir $(VIDEO_DIR) \
		--reference-root $(REFERENCE_ROOT) \
		--output-root $(OUTPUT_ROOT) \
		--run-name $(PIPELINE_RUN_NAME) \
		--device $(DEVICE) \
		--num-frames $(NUM_FRAMES) \
		--detect-batch-size $(DETECT_BATCH_SIZE) \
		--embed-batch-size $(EMBED_BATCH_SIZE) \
		--reference-embed-batch-size $(REFERENCE_EMBED_BATCH_SIZE) \
		--cluster-threshold $(CLUSTER_THRESHOLD) \
		--reference-match-threshold $(REFERENCE_MATCH_THRESHOLD) \
		--reference-match-margin $(REFERENCE_MATCH_MARGIN) \
		--face-verify-threshold $(FACE_VERIFY_THRESHOLD) \
		--max-faces-per-frame $(MAX_FACES_PER_FRAME) \
		$(if $(filter 1,$(USE_FP16)),--use-fp16,) \
		--overwrite

pipeline-smoke:
	$(RUN) $(PIPELINE_SCRIPT) \
		--txt-path $(TXT_PATH) \
		--video-dir $(VIDEO_DIR) \
		--reference-root $(REFERENCE_ROOT) \
		--output-root $(OUTPUT_ROOT) \
		--run-name $(PIPELINE_RUN_NAME)_smoke \
		--device $(DEVICE) \
		--num-frames $(NUM_FRAMES) \
		--detect-batch-size $(DETECT_BATCH_SIZE) \
		--embed-batch-size $(EMBED_BATCH_SIZE) \
		--reference-embed-batch-size $(REFERENCE_EMBED_BATCH_SIZE) \
		--cluster-threshold $(CLUSTER_THRESHOLD) \
		--reference-match-threshold $(REFERENCE_MATCH_THRESHOLD) \
		--reference-match-margin $(REFERENCE_MATCH_MARGIN) \
		--face-verify-threshold $(FACE_VERIFY_THRESHOLD) \
		--max-faces-per-frame $(MAX_FACES_PER_FRAME) \
		--max-dialogues $(SMOKE_MAX_DIALOGUES) \
		$(if $(filter 1,$(USE_FP16)),--use-fp16,) \
		--overwrite

clean:
	rm -rf $(OUTPUT_ROOT)/latest $(OUTPUT_ROOT)/pipeline_latest $(OUTPUT_ROOT)/pipeline_latest_smoke

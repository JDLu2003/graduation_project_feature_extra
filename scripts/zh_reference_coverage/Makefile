SHELL := /bin/zsh

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
CONDA_ENV ?= EL_LJD
PYTHON ?= python
RUN := conda run --no-capture-output -n $(CONDA_ENV) $(PYTHON) -u

SCRIPT := $(ROOT)/scripts/zh_reference_coverage/run_reference_coverage_eval.py
TXT_PATH ?= /data16T_1/sunshengzhe/lujiading/data_zh/dev/dev.txt
REFERENCE_ROOT ?= /data16T_2/sunshengzhe/reference_face_zh
OUTPUT_ROOT ?= $(ROOT)/scripts/zh_reference_coverage/artifacts
RUN_NAME ?= latest
THRESHOLDS ?= 1 3 5 10 15 20

.PHONY: help run clean

help:
	@echo "zh_reference_coverage targets:"
	@echo "  make run                          # 评估 reference 脸库覆盖率"
	@echo "  make clean                        # 删除 latest 输出"
	@echo ""
	@echo "Override examples:"
	@echo "  make run CONDA_ENV=EL_LJD"
	@echo "  make run RUN_NAME=run_20260413"

run:
	$(RUN) $(SCRIPT) \
		--txt-path $(TXT_PATH) \
		--reference-root $(REFERENCE_ROOT) \
		--output-root $(OUTPUT_ROOT) \
		--run-name $(RUN_NAME) \
		$(foreach thr,$(THRESHOLDS),--threshold $(thr)) \
		--overwrite

clean:
	rm -rf $(OUTPUT_ROOT)/latest

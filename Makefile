SHELL := /bin/zsh

ENV_NAME := security
PY := conda run --no-capture-output -n $(ENV_NAME) python

ROOT := /Users/jdlu/Project/graduation_project/feature_extra
CFG := $(ROOT)/config.yaml

ROLE_SCRIPT := $(ROOT)/scripts/stat_role_frequencies.py
FACE_BOX_SCRIPT := $(ROOT)/scripts/extract_faces_with_ffmpeg_mtcnn.py
TALKNET_EVAL_SCRIPT := $(ROOT)/talknet_asd/scripts/evaluate_feature_extra_talknet_asd.py

ROLE_SORT ?= utterances
ROLE_OUT_CSV ?= $(ROOT)/logs/role_frequencies.csv
ROLE_OUT_JSON ?= $(ROOT)/logs/role_frequencies.json

VIDEO ?=
FPS ?= 25
DEVICE ?= cpu
MIN_CONF ?= 0.95

MAX_UTT ?= 2
MAX_DIA ?= 0
ASD_TH ?= 0.0
TRACK_SAMPLE_FRAMES ?= 8
DIALOGUE_IDS ?=
CASE_TAGS ?=

.PHONY: help role-stats role-stats-export face-box-video talknet-smoke talknet-case talknet-full talknet-full-keep

help:
	@echo "Targets:"
	@echo "  make role-stats                     # Print role frequency table from parser/config."
	@echo "  make role-stats-export              # Export role stats CSV/JSON to logs."
	@echo "  make face-box-video VIDEO=/abs.mp4  # ffmpeg frame extract + MTCNN bbox + rebuild video."
	@echo "  make talknet-smoke                  # TalkNet ASD smoke eval (default first 2 utterances)."
	@echo "  make talknet-case CASE_TAGS=C_18_U_1,C_18_U_2  # Evaluate selected utterances."
	@echo "  make talknet-full                   # Full dataset TalkNet ASD eval (tmp cleaned)."
	@echo "  make talknet-full-keep              # Full eval and keep tmp artifacts."
	@echo ""
	@echo "Common vars:"
	@echo "  CFG=$(CFG)"
	@echo "  ROLE_SORT=$(ROLE_SORT) ROLE_OUT_CSV=$(ROLE_OUT_CSV) ROLE_OUT_JSON=$(ROLE_OUT_JSON)"
	@echo "  VIDEO=<abs_path> FPS=$(FPS) DEVICE=$(DEVICE) MIN_CONF=$(MIN_CONF)"
	@echo "  MAX_UTT=$(MAX_UTT) MAX_DIA=$(MAX_DIA) ASD_TH=$(ASD_TH) TRACK_SAMPLE_FRAMES=$(TRACK_SAMPLE_FRAMES)"
	@echo "  DIALOGUE_IDS=$(DIALOGUE_IDS) CASE_TAGS=$(CASE_TAGS)"

role-stats:
	@$(PY) $(ROLE_SCRIPT) --config $(CFG) --sort $(ROLE_SORT)

role-stats-export:
	@$(PY) $(ROLE_SCRIPT) --config $(CFG) --sort $(ROLE_SORT) --out-csv $(ROLE_OUT_CSV) --out-json $(ROLE_OUT_JSON)

face-box-video:
	@test -n "$(VIDEO)" || (echo "VIDEO is required, e.g. make face-box-video VIDEO=/abs/path/to/video.mp4" && exit 1)
	@$(PY) $(FACE_BOX_SCRIPT) "$(VIDEO)" --fps $(FPS) --device $(DEVICE) --min-confidence $(MIN_CONF)

talknet-smoke:
	@$(PY) $(TALKNET_EVAL_SCRIPT) --config $(CFG) --max-utterances $(MAX_UTT) --max-dialogues $(MAX_DIA) --dialogue-ids $(DIALOGUE_IDS) --asd-threshold $(ASD_TH) --track-sample-frames $(TRACK_SAMPLE_FRAMES) --keep-tmp

talknet-case:
	@test -n "$(CASE_TAGS)" || (echo "CASE_TAGS is required, e.g. make talknet-case CASE_TAGS=C_18_U_1,C_18_U_2" && exit 1)
	@$(PY) $(TALKNET_EVAL_SCRIPT) --config $(CFG) --dialogue-ids $(DIALOGUE_IDS) --case-tags $(CASE_TAGS) --asd-threshold $(ASD_TH) --track-sample-frames $(TRACK_SAMPLE_FRAMES) --keep-tmp

talknet-full:
	@$(PY) $(TALKNET_EVAL_SCRIPT) --config $(CFG) --dialogue-ids $(DIALOGUE_IDS) --asd-threshold $(ASD_TH) --track-sample-frames $(TRACK_SAMPLE_FRAMES)

talknet-full-keep:
	@$(PY) $(TALKNET_EVAL_SCRIPT) --config $(CFG) --dialogue-ids $(DIALOGUE_IDS) --asd-threshold $(ASD_TH) --track-sample-frames $(TRACK_SAMPLE_FRAMES) --keep-tmp

SHELL := /bin/zsh

# ===== Runtime config (can be overridden) =====
CONDA_ENV ?= security
PYTHON ?= python
CONFIG ?= config.yaml
MAX_DIALOGUES ?= 2

RUN := conda run --no-capture-output -n $(CONDA_ENV) $(PYTHON) -u

.PHONY: help main-info smoke full verify-and-full one-click

help:
	@echo "Usage:"
	@echo "  make main-info        # Show what main.py does"
	@echo "  make smoke            # Validation run (smoke test)"
	@echo "  make full             # Full feature extraction"
	@echo "  make verify-and-full  # smoke -> full"
	@echo "  make one-click        # Alias of verify-and-full"
	@echo ""
	@echo "Override examples:"
	@echo "  make smoke MAX_DIALOGUES=3"
	@echo "  make full CONFIG=config.yaml CONDA_ENV=security"

main-info:
	@echo "main.py core flow:"
	@echo "1) Load config from --config (default: config.yaml)"
	@echo "2) Parse dev.txt into dialogue records"
	@echo "3) If --smoke: only process first --max-dialogues dialogues"
	@echo "4) Init extractor by extractor.active_type (visual_clip / face_scene_fr)"
	@echo "5) Run pipeline: speaker + listeners feature extraction, then save .pt"

smoke:
	@echo "[smoke] Running validation extraction with MAX_DIALOGUES=$(MAX_DIALOGUES), CONFIG=$(CONFIG)"
	$(RUN) main.py --config $(CONFIG) --smoke --max-dialogues $(MAX_DIALOGUES)

full:
	@echo "[full] Running full extraction with CONFIG=$(CONFIG)"
	$(RUN) main.py --config $(CONFIG)

verify-and-full: smoke full

one-click: verify-and-full
